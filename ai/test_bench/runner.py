"""Core inference and tracking pipeline for the CCTV Test Bench.

Integrates:
- OpenCV YuNet (face detection + 5 landmarks)
- OpenCV SFace (face alignment + 128-d embeddings)
- Gallery cosine matching + sigmoid confidence + blur penalty
- Frigate weighted average voting
- RapidOCR text detection + CLAHE night contrast boost
- Indian plate format validation & positional OCR repair
- Frigate Jaro-Winkler greedy clustering (_get_cluster_rep)
"""

from __future__ import annotations

import base64
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from rapidfuzz.distance import Levenshtein

from plugins.face_recognition.engine import FaceEngine, iou
from plugins.face_recognition.voting import FaceVote
from plugins.vehicle_lpr import plates
from plugins.vehicle_lpr.clustering import PlateRead, best_plate, cluster_reads
from plugins.vehicle_lpr.ocr import PlateOCR, enhance_contrast

log = logging.getLogger("cctv_ai.test_bench")


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def crop_data_url(
    frame: np.ndarray,
    box: list[int] | tuple[int, int, int, int],
    pad_frac: float = 0.3,
    max_w: int = 200,
) -> str | None:
    """Padded crop of `box` from `frame` as a small JPEG data URL.

    Used for video-analysis snapshots: one thumbnail per detection so the UI
    can show where in the clip the face/plate appeared.
    """
    h, w = frame.shape[:2]
    x, y, bw, bh = (int(v) for v in box)
    px, py = int(bw * pad_frac), int(bh * pad_frac)
    x1, y1 = max(0, x - px), max(0, y - py)
    x2, y2 = min(w, x + bw + px), min(h, y + bh + py)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.shape[1] > max_w:
        scale = max_w / crop.shape[1]
        crop = cv2.resize(crop, (max_w, max(1, int(crop.shape[0] * scale))))
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


@dataclass
class FaceDetectionResult:
    box: list[int]  # [x, y, w, h]
    landmarks: list[list[float]]  # 5 points: [[rx, ry], [lx, ly], [nx, ny], [rmx, rmy], [lmx, lmy]]
    score: float
    name: str | None  # shown name: only set when the confidence clears the threshold
    match_name: str | None  # gallery match at the SFace match level (pre-threshold)
    confidence: float
    similarity: float
    blur_penalty: float
    is_known: bool
    usable: bool  # quality gate passed (sharp, near-frontal)
    embedding_preview: list[float]  # first 8 values for telemetry display


@dataclass
class PlateDetectionResult:
    box: list[int]  # [x, y, w, h]
    raw_text: str
    repaired_plate: str | None
    confidence: float
    is_valid_indian: bool
    plate_type: str | None  # "standard", "bharat", or "unrecognized"
    is_two_line: bool
    known_as: str | None


@dataclass
class ProcessResult:
    width: int
    height: int
    faces: list[FaceDetectionResult]
    plates: list[PlateDetectionResult]
    telemetry: dict[str, float]
    events: list[dict[str, Any]]


class VisionPipeline:
    def __init__(self) -> None:
        self.root = get_repo_root()
        models_dir = self.root / "ai" / "models"
        self.face_engine = FaceEngine(models_dir, detection_threshold=0.6)
        self.ocr = PlateOCR()
        self.models_loaded = False
        self.known_plates: dict[str, list[str]] = {}
        self.face_lock = threading.RLock()  # YuNet + SFace not thread-safe
        self.plate_lock = threading.RLock()  # RapidOCR not thread-safe
        self.load()

    def load(self) -> None:
        self.face_engine.load_models()
        self.ocr.load()
        self.reload_gallery()
        self.reload_known_plates()
        self.models_loaded = self.face_engine.ready and (self.ocr.engine is not None)

    def reload_gallery(self) -> int:
        gallery_path = self.root / "data" / "faces.npz"
        if gallery_path.exists():
            try:
                count = self.face_engine.load_gallery(gallery_path)
                log.info("Loaded face gallery with %d identities", count)
                return count
            except Exception as e:
                log.warning("Could not load gallery: %s", e)
        return 0

    def reload_known_plates(self) -> dict[str, list[str]]:
        plates_file = self.root / "data" / "known_plates.yaml"
        if not plates_file.exists():
            self.known_plates = {}
            return self.known_plates
        try:
            raw = yaml.safe_load(plates_file.read_text(encoding="utf-8")) or {}
            self.known_plates = {
                str(owner): [plates.clean_text(p) for p in p_list]
                for owner, p_list in raw.items()
            }
        except Exception as e:
            log.warning("Could not load known_plates.yaml: %s", e)
            self.known_plates = {}
        return self.known_plates

    def get_known_owner(self, plate: str, max_dist: int = 1) -> str | None:
        if not self.known_plates:
            self.reload_known_plates()
        for owner, plate_list in self.known_plates.items():
            for known in plate_list:
                if Levenshtein.distance(plate, known) <= max_dist:
                    return owner
        return None

    def process_bgr(
        self,
        bgr: np.ndarray,
        mode: str = "both",
        contrast_boost: bool = False,
        min_face_area: int = 750,
        face_det_thresh: float = 0.6,
        face_rec_thresh: float = 0.8,
        min_plate_area: int = 1000,
        min_plate_conf: float = 0.80,
    ) -> ProcessResult:
        with self.face_lock, self.plate_lock:
            return self._process_bgr(
                bgr, mode, contrast_boost, min_face_area, face_det_thresh,
                face_rec_thresh, min_plate_area, min_plate_conf,
            )

    def process_faces_bgr(
        self,
        bgr: np.ndarray,
        contrast_boost: bool = False,
        min_face_area: int = 750,
        face_det_thresh: float = 0.6,
        face_rec_thresh: float = 0.8,
        track: bool = True,
    ) -> ProcessResult:
        """Face detection + recognition only (fast ~20-40 ms).

        Uses *face_lock* and *track=True* to cache facial embeddings across
        consecutive frames, delivering realtime 15 FPS video with minimal latency.
        """
        with self.face_lock:
            return self._process_bgr(
                bgr, mode="faces", contrast_boost=contrast_boost,
                min_face_area=min_face_area, face_det_thresh=face_det_thresh,
                face_rec_thresh=face_rec_thresh, track=track,
            )

    def process_plates_bgr(
        self,
        bgr: np.ndarray,
        contrast_boost: bool = False,
        min_plate_area: int = 1000,
        min_plate_conf: float = 0.80,
    ) -> ProcessResult:
        """Plate OCR only (slow ~500-1000 ms on CPU).

        Uses *plate_lock* so it can run concurrently with
        :meth:`process_faces_bgr` on a different thread.
        """
        with self.plate_lock:
            return self._process_bgr(
                bgr, mode="plates", contrast_boost=contrast_boost,
                min_plate_area=min_plate_area, min_plate_conf=min_plate_conf,
                track=False,
            )

    def _process_bgr(
        self,
        bgr: np.ndarray,
        mode: str = "both",
        contrast_boost: bool = False,
        min_face_area: int = 750,
        face_det_thresh: float = 0.6,
        face_rec_thresh: float = 0.8,
        min_plate_area: int = 1000,
        min_plate_conf: float = 0.80,
        track: bool = False,
    ) -> ProcessResult:
        h, w = bgr.shape[:2]
        t_start = time.perf_counter()
        telemetry: dict[str, float] = {}

        face_results: list[FaceDetectionResult] = []
        plate_results: list[PlateDetectionResult] = []
        events: list[dict[str, Any]] = []

        # 1. Face Recognition
        if mode in ("faces", "both"):
            t_face0 = time.perf_counter()
            self.face_engine.set_detection_threshold(face_det_thresh)
            matches = self.face_engine.recognise_all(
                bgr, min_area=min_face_area, max_dim=400, track=track
            )
            telemetry["face_detection_ms"] = round((time.perf_counter() - t_face0) * 1000, 2)

            t_embed0 = time.perf_counter()
            for match in matches:
                face = match.face
                preview = [round(float(v), 4) for v in match.embedding[:8]] if match.embedding is not None else []

                # a face is "known" only when the quality gate passed and the
                # confidence (sigmoid of similarity minus blur penalty) clears
                # the recognition threshold — never compare raw cosine here
                is_known = (match.usable and match.name is not None
                            and match.confidence >= face_rec_thresh)
                name = match.name if is_known else None

                # 5 landmarks from YuNet row
                row = face.row
                landmarks = [
                    [float(row[4]), float(row[5])],   # Right eye
                    [float(row[6]), float(row[7])],   # Left eye
                    [float(row[8]), float(row[9])],   # Nose
                    [float(row[10]), float(row[11])], # Right mouth
                    [float(row[12]), float(row[13])], # Left mouth
                ]

                face_results.append(
                    FaceDetectionResult(
                        box=[int(face.box[0]), int(face.box[1]), int(face.box[2]), int(face.box[3])],
                        landmarks=landmarks,
                        score=round(face.score, 3),
                        name=name,
                        match_name=match.name,
                        confidence=round(match.confidence, 3),
                        similarity=round(match.similarity, 3),
                        blur_penalty=round(match.blur, 3),
                        is_known=is_known,
                        usable=match.usable,
                        embedding_preview=preview,
                    )
                )

                events.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "type": "known_person" if is_known else "unknown_person",
                    "label": name or "Unknown Person",
                    "confidence": round(match.confidence, 3),
                    "box": [int(x) for x in face.box],
                })
            telemetry["face_embedding_ms"] = round((time.perf_counter() - t_embed0) * 1000, 2)

        # 2. Vehicle LPR
        if mode in ("plates", "both"):
            t_ocr0 = time.perf_counter()
            prep_bgr = enhance_contrast(bgr) if contrast_boost else bgr
            lines = self.ocr.detect_text(prep_bgr)
            telemetry["ocr_inference_ms"] = round((time.perf_counter() - t_ocr0) * 1000, 2)

            t_post0 = time.perf_counter()
            boxes = [b for b, _, _ in lines]
            merged_boxes = plates.merge_two_line(boxes) if boxes else []

            for x, y, bw, bh in merged_boxes:
                assembled = plates.text_in_box(lines, (x, y, bw, bh), margin=2.0)
                if assembled is None:
                    continue
                box_text, avg_conf, _count, two_line = assembled

                if not plates.is_plate_shaped(bw, bh, min_plate_area, two_line=two_line):
                    continue

                if avg_conf < min_plate_conf:
                    continue

                repaired = plates.normalize(box_text, min_length=plates.MIN_PLATE_LENGTH)
                is_valid = repaired is not None

                plate_type = "unrecognized"
                if is_valid and repaired:
                    if plates.BHARAT_PLATE_RE.fullmatch(repaired):
                        plate_type = "bharat"
                    elif plates.INDIAN_PLATE_RE.fullmatch(repaired):
                        plate_type = "standard"

                known_owner = self.get_known_owner(repaired) if repaired else None

                plate_results.append(
                    PlateDetectionResult(
                        box=[int(x), int(y), int(bw), int(bh)],
                        raw_text=box_text,
                        repaired_plate=repaired,
                        confidence=round(avg_conf, 3),
                        is_valid_indian=is_valid,
                        plate_type=plate_type,
                        is_two_line=two_line,
                        known_as=known_owner,
                    )
                )

                if is_valid and repaired:
                    events.append({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "type": "vehicle_plate",
                        "label": repaired,
                        "confidence": round(avg_conf, 3),
                        "known_as": known_owner,
                        "box": [int(x), int(y), int(bw), int(bh)],
                    })
            telemetry["plate_postprocessing_ms"] = round((time.perf_counter() - t_post0) * 1000, 2)

        telemetry["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)

        return ProcessResult(
            width=w,
            height=h,
            faces=face_results,
            plates=plate_results,
            telemetry=telemetry,
            events=events,
        )

    def process_video_frames(
        self,
        video_path: Path,
        stride: int = 5,
        mode: str = "both",
        max_frames: int = 150,
        min_face_area: int = 750,
        min_plate_conf: float = 0.80,
        face_rec_thresh: float = 0.8,
        contrast_boost: bool = False,
    ) -> dict[str, Any]:
        """Process video frames with per-face-track voting (Frigate-style) and
        per-clip plate clustering, capturing a snapshot of the best frame of
        every face track and every plate read.

        Faces are assigned to tracks by box overlap (IoU), so two people in
        frame never mix their votes; each track votes only on quality-gated
        attempts, using 0..1 confidences (never raw cosine similarity). The
        returned `detections` list carries one entry per face track / plate
        with a timestamp, box and thumbnail so the UI can seek the player
        to each spot.
        """
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Cannot open video {video_path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_results: list[dict[str, Any]] = []
        plate_reads: list[PlateRead] = []
        plate_snapshots: dict[str, dict[str, Any]] = {}  # plate -> best snapshot
        tracks: list[dict[str, Any]] = []  # face tracks: {"id", "vote", "box", ...}
        next_track_id = 0
        MATCH_IOU = 0.3  # below this overlap it is a different face
        MAX_GAP = 3  # processed frames a track survives without a match

        def matched_track(box: list[int], frame_idx: int) -> dict[str, Any] | None:
            best, best_iou = None, MATCH_IOU
            for track in tracks:
                if frame_idx - track["last_seen"] > MAX_GAP:
                    continue
                overlap = iou(track["box"], box)
                if overlap > best_iou:
                    best, best_iou = track, overlap
            return best

        processed_count = 0
        frame_idx = 0
        # Lock removed: process_video_frames runs sequentially from a video file
        # with only local variables; no shared mutable state needs serialization.
        while frame_idx < total_frames and processed_count < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break

                if frame_idx % stride == 0:
                    with self.face_lock, self.plate_lock:
                        res = self._process_bgr(
                            frame,
                            mode=mode,
                            contrast_boost=contrast_boost,
                            min_face_area=min_face_area,
                            min_plate_conf=min_plate_conf,
                        )

                    # Update per-track Frigate voting + best-attempt snapshot
                    for f in sorted(res.faces, key=lambda r: r.box[2] * r.box[3], reverse=True):
                        area = f.box[2] * f.box[3]
                        track = matched_track(f.box, frame_idx)
                        if track is None:
                            next_track_id += 1
                            track = {
                                "id": next_track_id,
                                "vote": FaceVote(unknown_score=0.8,
                                                 recognition_threshold=face_rec_thresh,
                                                 min_faces=3),
                                "box": f.box,
                                "first_seen": frame_idx,
                                "last_seen": frame_idx,
                                "best_conf": -1.0,
                                "snapshot": None,
                            }
                            tracks.append(track)
                        track["box"] = f.box
                        track["last_seen"] = frame_idx
                        if f.usable:
                            track["vote"].add(f.match_name, f.confidence, area)
                            if f.confidence > track["best_conf"]:
                                track["best_conf"] = f.confidence
                                track["snapshot"] = {
                                    "snapshot": crop_data_url(frame, f.box, pad_frac=0.35),
                                    "box": f.box,
                                    "confidence": round(f.confidence, 3),
                                    "match_name": f.match_name,
                                    "similarity": f.similarity,
                                    "timestamp_sec": round(frame_idx / fps, 2),
                                    "frame_index": frame_idx,
                                }

                    # Update plate reads + best snapshot per plate text
                    for p in res.plates:
                        if p.repaired_plate:
                            plate_reads.append(PlateRead(p.repaired_plate, p.confidence, p.box[2] * p.box[3]))
                            current = plate_snapshots.get(p.repaired_plate)
                            if current is None or p.confidence > current["confidence"]:
                                plate_snapshots[p.repaired_plate] = {
                                    "snapshot": crop_data_url(frame, p.box, pad_frac=0.5),
                                    "box": p.box,
                                    "confidence": round(p.confidence, 3),
                                    "raw_text": p.raw_text,
                                    "plate_type": p.plate_type,
                                    "known_as": p.known_as,
                                    "timestamp_sec": round(frame_idx / fps, 2),
                                    "frame_index": frame_idx,
                                }

                    frame_results.append({
                        "frame_index": frame_idx,
                        "timestamp_sec": round(frame_idx / fps, 2),
                        "faces": [asdict(f) for f in res.faces],
                        "plates": [asdict(p) for p in res.plates],
                        "telemetry": res.telemetry,
                    })
                    processed_count += 1

                frame_idx += 1

        capture.release()

        # One decision per face track; the summary shows the strongest one
        track_decisions: list[dict[str, Any]] = []
        for track in tracks:
            name, score = track["vote"].decision()
            track_decisions.append({
                "track": track["id"],
                "first_frame": track["first_seen"],
                "last_frame": track["last_seen"],
                "decision_name": name,
                "confidence": round(score, 3),
                "total_attempts": len(track["vote"].attempts),
                "voting_history": track["vote"].attempts,
            })
        best = max(
            track_decisions,
            key=lambda d: (d["decision_name"] is not None
                           and d["confidence"] >= face_rec_thresh, d["confidence"]),
            default=None,
        )

        best_plate_res = best_plate(plate_reads, threshold=0.85)

        # detection timeline: one entry per face track / plate, sorted by time
        detections: list[dict[str, Any]] = []
        for track, decision in zip(tracks, track_decisions):
            snap = track.get("snapshot")
            if not snap or not snap.get("snapshot"):
                continue
            detections.append({
                "kind": "face",
                "label": snap["match_name"] or "Unknown",
                "is_known": snap["match_name"] is not None,
                "confidence": snap["confidence"],
                "vote_name": decision["decision_name"],
                "vote_confidence": decision["confidence"],
                "timestamp_sec": snap["timestamp_sec"],
                "frame_index": snap["frame_index"],
                "box": snap["box"],
                "snapshot": snap["snapshot"],
            })
        for plate_text, snap in plate_snapshots.items():
            if not snap.get("snapshot"):
                continue
            detections.append({
                "kind": "plate",
                "label": plate_text,
                "is_known": snap["known_as"] is not None,
                "known_as": snap["known_as"],
                "confidence": snap["confidence"],
                "raw_text": snap["raw_text"],
                "plate_type": snap["plate_type"],
                "timestamp_sec": snap["timestamp_sec"],
                "frame_index": snap["frame_index"],
                "box": snap["box"],
                "snapshot": snap["snapshot"],
            })
        detections.sort(key=lambda d: d["timestamp_sec"])

        summary = {
            "total_video_frames": total_frames,
            "processed_frames": processed_count,
            "fps": round(fps, 1),
            "resolution": [width, height],
            "stride": stride,
            "face_vote_summary": {
                "decision_name": best["decision_name"] if best else None,
                "confidence": round(best["confidence"], 3) if best else 0.0,
                "total_attempts": best["total_attempts"] if best else 0,
                "voting_history": best["voting_history"] if best else [],
                "tracks": [{k: v for k, v in d.items() if k != "voting_history"}
                           for d in track_decisions],
            },
            "plate_cluster_summary": {
                "best_plate": best_plate_res[0] if best_plate_res else None,
                "confidence": round(best_plate_res[1], 3) if best_plate_res else 0.0,
                "agreeing_reads": best_plate_res[2] if best_plate_res else 0,
                "total_reads": len(plate_reads),
                "known_owner": self.get_known_owner(best_plate_res[0]) if best_plate_res else None,
            },
            "detections": detections,
            "frame_results": frame_results,
        }
        return summary
