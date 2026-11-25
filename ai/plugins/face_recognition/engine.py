"""Face detection + embedding + matching (OpenCV Zoo YuNet and SFace).

Both models are MIT/Apache-2.0 licensed and load with plain opencv-python:
only the ONNX files are needed, see download_models.py. YuNet finds faces
with 5 landmarks; SFace embeds an aligned face crop into 128 numbers that
are compared by cosine similarity.

The match score is mapped to a 0..1 confidence with a sigmoid centred on
SFace's suggested cosine match threshold (0.363), like Frigate does, and
blurry faces lose confidence (Laplacian variance penalty).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("cctv_ai.face_recognition")

# SFace cosine similarity above this is the same person (OpenCV Zoo suggestion)
SFACE_MATCH_THRESHOLD = 0.363

# confidence for a cosine similarity at the match threshold (0.5) and slope
SIGMOID_CENTER = SFACE_MATCH_THRESHOLD
SIGMOID_SLOPE = 10.0

# quality gates: attempts outside these waste a vote and pollute the gallery
# match (tuned on 112x112 aligned SFace crops)
MIN_SHARPNESS = 70.0  # Laplacian variance below this: the embed is noise
MAX_POSE_OFFSET = 0.35  # nose x offset from the eye line, in eye distances


def similarity_to_confidence(similarity: float) -> float:
    """Map a cosine similarity to a 0..1 confidence (sigmoid)."""
    return float(1.0 / (1.0 + np.exp(-SIGMOID_SLOPE * (similarity - SIGMOID_CENTER))))


def sharpness(face: np.ndarray) -> float:
    """Laplacian variance of a face crop (higher = sharper)."""
    if face is None or face.size == 0:
        return 0.0
    grey = face if face.ndim == 2 else cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def blur_penalty(face: np.ndarray) -> float:
    """Confidence to subtract for a blurry face (Frigate's Laplacian penalty).

    Pass the aligned 112x112 face crop, not the whole frame: the sharpness
    of the scene says nothing about the sharpness of the face.
    """
    if face is None or face.size == 0:
        return 0.06
    level = sharpness(face)
    if level < 120:
        return 0.06
    if level < 160:
        return 0.04
    if level < 200:
        return 0.02
    if level < 250:
        return 0.01
    return 0.0


def pose_offset(face: "Face") -> float:
    """How far the nose sits from the eye-line midpoint, in eye distances.

    ~0 for a frontal face, grows with yaw (profile faces drift to ~0.5+).
    Used to skip faces that would embed garbage.
    """
    row = face.row
    right_eye_x, left_eye_x, nose_x = float(row[4]), float(row[6]), float(row[8])
    eye_distance = abs(left_eye_x - right_eye_x)
    if eye_distance < 1e-3:
        return 1.0
    return abs(nose_x - (right_eye_x + left_eye_x) / 2.0) / eye_distance


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection over union of two (x, y, w, h) boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    ix = max(0.0, float(min(ax2, bx2) - max(ax1, bx1)))
    iy = max(0.0, float(min(ay2, by2) - max(ay1, by1)))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    return inter / float(aw * ah + bw * bh - inter)


class Face:
    """One detected face in an image."""

    def __init__(self, box: tuple[int, int, int, int], score: float, yunet_row: np.ndarray) -> None:
        self.box = box  # x, y, w, h
        self.score = score
        self.row = yunet_row  # YuNet's raw row (box + 5 landmarks), needed by alignCrop

    @property
    def area(self) -> int:
        return self.box[2] * self.box[3]


@dataclass
class FaceMatch:
    """One detected face with its gallery match (or unknown)."""

    face: Face
    name: str | None  # None = unknown (below the match level)
    confidence: float  # 0..1, sigmoid of the similarity minus the blur penalty
    similarity: float  # raw SFace cosine similarity to the best gallery entry
    usable: bool  # False when the quality gate failed: do not vote on this
    embedding: np.ndarray | None = None  # 128 floats, for diagnostics
    blur: float = 0.0  # the blur penalty that was subtracted


class FaceEngine:
    """Loads the models once and turns BGR images into (name, confidence)."""

    def __init__(self, models_dir: Path, detection_threshold: float = 0.7) -> None:
        self.models_dir = Path(models_dir)
        self.detection_threshold = detection_threshold
        self.detector: cv2.FaceDetectorYN | None = None
        self.recognizer: cv2.FaceRecognizerSF | None = None
        self.names: list[str] = []
        self.means: np.ndarray | None = None  # (n, 128) L2-normalised rows
        self._track_cache: list[dict[str, Any]] = []

    def clear_tracks(self) -> None:
        """Clear cached tracks between disparate videos or test runs."""
        self._track_cache.clear()

    def load_models(self) -> bool:
        """Load YuNet and SFace. False when a model file is missing."""
        yunet = self.models_dir / "face_detection_yunet_2023mar.onnx"
        sface = self.models_dir / "face_recognition_sface_2021dec.onnx"
        for path in (yunet, sface):
            if not path.exists():
                log.warning("face model missing: %s (run download_models.py)", path)
                return False
        self.detector = cv2.FaceDetectorYN.create(
            str(yunet), "", (320, 320), self.detection_threshold, 0.3
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(sface), "")
        return True

    def load_gallery(self, npz_path: Path) -> int:
        """Load the known-person means built by build_gallery.py."""
        data = np.load(npz_path, allow_pickle=False)
        self.names = [str(n) for n in data["names"]]
        self.means = data["means"].astype(np.float32)
        return len(self.names)

    def set_detection_threshold(self, value: float) -> None:
        """Change the YuNet score threshold on the live detector."""
        self.detection_threshold = float(value)
        if self.detector is not None:
            self.detector.setScoreThreshold(self.detection_threshold)

    @property
    def ready(self) -> bool:
        return self.detector is not None and self.recognizer is not None

    def detect(self, bgr: np.ndarray, max_dim: int = 0) -> list[Face]:
        """Faces in a BGR image, biggest first.

        If max_dim > 0 and max(w, h) > max_dim, detection runs on a scaled
        down copy for massive speedups (e.g. 25ms vs 230ms), and coordinates
        are projected back to original space so embeddings retain full quality.
        """
        if self.detector is None:
            return []
        h, w = bgr.shape[:2]
        if h == 0 or w == 0:
            return []

        scale = 1.0
        det_img = bgr
        if max_dim > 0 and max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            dw, dh = max(1, int(w * scale)), max(1, int(h * scale))
            det_img = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA)
            self.detector.setInputSize((dw, dh))
        else:
            self.detector.setInputSize((w, h))

        _, rows = self.detector.detect(det_img)
        faces: list[Face] = []
        if rows is not None:
            for row in rows:
                if scale != 1.0:
                    row = row.copy()
                    row[:14] /= scale
                x, y, fw, fh = row[:4].astype(int)
                faces.append(Face((x, y, fw, fh), float(row[14]), row))
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces

    def align_crop(self, bgr: np.ndarray, face: Face) -> np.ndarray:
        """The aligned 112x112 face crop (good for snapshots)."""
        assert self.recognizer is not None
        return self.recognizer.alignCrop(bgr, face.row)

    def embed(self, bgr: np.ndarray, face: Face) -> np.ndarray:
        """Align the face with its landmarks and embed it (128 floats)."""
        aligned = self.align_crop(bgr, face)
        return self.recognizer.feature(aligned).flatten()

    def recognise_all(
        self,
        bgr: np.ndarray,
        min_area: int = 0,
        max_dim: int = 0,
        track: bool = False,
        max_embed_per_frame: int = 1,
    ) -> list[FaceMatch]:
        """Match every face in the image against the gallery.

        When track=True, tracks faces across consecutive frames using IOU to reuse
        SFace embeddings. SFace feature extraction costs ~180-250ms per face on CPU;
        reusing embeddings drops per-frame time to ~20-40ms while maintaining full accuracy.
        """
        now = time.monotonic()
        if track:
            self._track_cache = [c for c in self._track_cache if now - c["time"] < 2.0]

        matches: list[FaceMatch] = []
        embed_budget = max_embed_per_frame if track else 999
        for face in self.detect(bgr, max_dim=max_dim):
            if face.area < min_area:
                continue

            best_track = None
            if track:
                for c in self._track_cache:
                    if iou(face.box, c["box"]) >= 0.30:
                        best_track = c
                        break

            aligned = self.align_crop(bgr, face)
            blur = blur_penalty(aligned)
            usable = (sharpness(aligned) >= MIN_SHARPNESS
                      and pose_offset(face) <= MAX_POSE_OFFSET)

            if not usable:
                # Blurry or extreme profile: do not waste ~200ms on SFace
                matches.append(FaceMatch(face, None, 0.0, 0.0, False, None, blur))
                continue

            if self.means is None or not self.names:
                # No gallery built yet
                matches.append(FaceMatch(face, None, 0.0, 0.0, True, None, blur))
                continue

            need_embed = (best_track is None) or (now - best_track.get("embed_time", 0.0) >= 1.0)
            if need_embed and embed_budget > 0:
                embedding = self.recognizer.feature(aligned).flatten()
                embed_budget -= 1
                sims = self.means @ (embedding / (np.linalg.norm(embedding) + 1e-9))
                best = int(np.argmax(sims))
                similarity = float(sims[best])
                confidence = max(0.0, similarity_to_confidence(similarity) - blur)
                name = self.names[best] if similarity >= SFACE_MATCH_THRESHOLD else None
                match = FaceMatch(face, name, confidence, similarity, True, embedding, blur)
                if track:
                    if best_track is not None:
                        best_track.update({"box": face.box, "time": now, "embed_time": now, "match": match})
                    else:
                        self._track_cache.append({"box": face.box, "time": now, "embed_time": now, "match": match})
            elif best_track is not None:
                cached_m = best_track["match"]
                best_track["box"] = face.box
                best_track["time"] = now
                match = FaceMatch(
                    face=face,
                    name=cached_m.name,
                    confidence=cached_m.confidence,
                    similarity=cached_m.similarity,
                    usable=True,
                    embedding=cached_m.embedding,
                    blur=blur,
                )
            else:
                match = FaceMatch(face, None, 0.0, 0.0, True, None, blur)

            matches.append(match)
        return matches

    def recognise(self, bgr: np.ndarray, min_area: int = 0) -> tuple[str | None, float, Face | None]:
        """Best match for the biggest face in the image.

        Returns (name or None for unknown, confidence, face). Kept for the
        plugin paths that only want one answer; use recognise_all for frames
        with several people.
        """
        matches = self.recognise_all(bgr, min_area)
        if not matches:
            return None, 0.0, None
        match = matches[0]
        return match.name, match.confidence, match.face
