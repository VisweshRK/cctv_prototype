"""Vehicle and plate recognition using vehicle-candidate crops from the shared detector."""

from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from rapidfuzz.distance import Levenshtein

from cctv_ai.plugin import Event, Plugin as Base
from cctv_ai.objects import TrackedObject
from cctv_ai.tracker import box_iou

from . import plates
from .clustering import PlateRead, best_plate
from .ocr import PlateOCR, enhance_contrast, prepare_crop

log = logging.getLogger("cctv_ai.vehicle_lpr")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class Plugin(Base):
    use_case = "vehicle_lpr"
    enabled = True  # stays silent until the OCR models are present
    needs = {"objects"}
    interval = 2.0
    heavy = True  # the pipeline should run this plugin on its own thread

    # tuning (see docs/plans/p2-krithick.md)
    min_plate_area = 1000  # px of a text box to count as a plate candidate
    min_read_confidence = 0.80  # RapidOCR avg char confidence; 0.85 dropped clean renders too
    min_plate_length = plates.MIN_PLATE_LENGTH
    cluster_threshold = 0.85  # Jaro-Winkler similarity to group reads
    confirm_reads = 1  # one high-confidence, format-valid crop read is enough
    max_reads = 25  # per track: last fps x 5 reads (5 s at 5 fps)
    min_read_interval = 1.0  # demo mode: seconds between full-frame OCR runs
    duplicate_window = 60.0  # demo mode: no repeated event for the same plate
    match_distance = 1  # Levenshtein distance for known plates

    def setup(self, cameras: list[str]) -> None:
        self.ocr = PlateOCR()
        self.ready = self.ocr.load()
        self.known: dict[str, list[str]] = self._load_known_plates()
        self.reads: dict[str, deque] = {c: deque(maxlen=self.max_reads) for c in cameras}
        self.announced: dict[str, dict[str, float]] = {c: {} for c in cameras}
        self.already_read: set[str] = set()  # object mode: stationary, read once
        self.last_ocr: dict[str, float] = {c: 0.0 for c in cameras}
        self.vehicle_tracker = VehicleObjectTracker()

    def track_candidates(self, camera: str, detections: list[dict], ts: datetime):
        return self.vehicle_tracker.update(camera, detections, ts)

    def _load_known_plates(self) -> dict[str, list[str]]:
        path = repo_root() / "data" / "known_plates.yaml"
        if not path.exists():
            return {}
        try:
            data = yaml.safe_load(path.read_text()) or {}
            return {str(k): [plates.clean_text(p) for p in v] for k, v in data.items()}
        except Exception as err:
            log.warning("cannot read %s: %s", path, err)
            return {}

    def known_as(self, plate: str) -> str | None:
        """Name from data/known_plates.yaml with Levenshtein distance <= 1."""
        for name, plates_list in self.known.items():
            for known in plates_list:
                if Levenshtein.distance(plate, known) <= self.match_distance:
                    return name
        return None

    def _reads_from_lines(self, lines: list[tuple[tuple[float, float, float, float], str, float]]) -> list[PlateRead]:
        """Plate-shaped OCR lines (or stacked pairs) turned into valid reads."""
        if not lines:
            return []
        merged = plates.merge_two_line([box for box, _, _ in lines])
        reads: list[PlateRead] = []
        for box in merged:
            assembled = plates.text_in_box(lines, box)
            if assembled is None:
                continue
            text, conf, _count, two_line = assembled
            x, y, w, h = box
            if not plates.is_plate_shaped(w, h, self.min_plate_area, two_line=two_line):
                continue
            plate = plates.normalize(text, self.min_plate_length)
            if plate is None or conf < self.min_read_confidence:
                continue
            reads.append(PlateRead(plate, conf, int(w * h)))
        return reads

    def _event(self, camera: str, plate: str, conf: float, reads: int,
               vehicle: str | None = None) -> Event:
        event = self.event(camera, "vehicle", conf, data={
            "plate": plate,
            "vehicle": vehicle,
            "reads": reads,
            "known_as": self.known_as(plate),
        })
        log.info("%s: plate %s conf %.2f from %d reads", camera, plate, conf, reads)
        return event

    def _announce(self, camera: str, result, vehicle: str | None = None) -> Event | None:
        if result is None or result[2] < self.confirm_reads:
            return None
        plate, conf, agreeing = result
        now = time.monotonic()
        if now - self.announced.setdefault(camera, {}).get(plate, 0.0) < self.duplicate_window:
            return None
        self.announced[camera][plate] = now
        return self._event(camera, plate, conf, agreeing, vehicle=vehicle)

    # ---- demo path (whole frame, per camera, no tracks) ------------------

    def on_frame(self, camera: str, frame: np.ndarray, ts: datetime) -> list[Event]:
        if not self.ready:
            return []
        now = time.monotonic()
        last = self.last_ocr.get(camera, 0.0)
        if now - last < self.min_read_interval:
            return []
        self.last_ocr[camera] = now

        if camera not in self.reads:
            self.reads[camera] = deque(maxlen=self.max_reads)
            self.announced[camera] = {}
        self.reads[camera].extend(self._reads_from_lines(self.ocr.detect_text(frame)))

        result = best_plate(list(self.reads[camera]), self.cluster_threshold)
        if result is None or result[2] < self.confirm_reads:
            return []
        plate, conf, agreeing = result
        last = self.announced[camera].get(plate)
        if last is not None and now - last < self.duplicate_window:
            return []
        self.announced[camera][plate] = now
        return [self._event(camera, plate, conf, agreeing)]

    # ---- real path (tracked objects, once the shared pipeline lands) ----

    def on_objects(self, camera: str, frame: np.ndarray, ts: datetime,
                   objects: list, ended: list) -> list[Event]:
        """objects/ended are TrackedObject instances from cctv_ai.objects."""
        if not self.ready:
            return []
        events: list[Event] = []

        for obj in ended:  # vehicle left: report what we have
            key = f"{camera}:{obj.id}"
            self.already_read.discard(key)  # keep the set from growing forever
            if key not in self.reads:
                continue
            result = best_plate(list(self.reads[key]), self.cluster_threshold)
            del self.reads[key]
            if event := self._announce(camera, result, vehicle=obj.label):
                events.append(event)

        for obj in objects:
            if obj.label not in plates.VEHICLE_LABELS or not obj.true_positive:
                continue
            key = f"{camera}:{obj.id}"
            if key not in self.reads:
                self.reads[key] = deque(maxlen=self.max_reads)
            if obj.stationary and key in self.already_read:
                continue  # parked and already read: stop OCRing it
            result = best_plate(list(self.reads[key]), self.cluster_threshold)
            if result and result[2] >= self.confirm_reads:
                self.already_read.add(key)  # confirmed: no more reads needed
                continue

            x1, y1, x2, y2 = obj.box
            crop = frame[y1:y2, x1:x2]
            crop = prepare_crop(crop)
            reads = self._reads_from_lines(self.ocr.detect_text(crop))
            self.reads[key].extend(reads)

            result = best_plate(list(self.reads[key]), self.cluster_threshold)
            if result and result[2] >= self.confirm_reads:
                self.already_read.add(key)
                if event := self._announce(camera, result, vehicle=obj.label):
                    events.append(event)
        return events


class VehicleObjectTracker:
    """Short-lived IoU tracks that let OCR run only on detected vehicle crops."""

    def __init__(self, max_missed: int = 10, threshold: float = 0.18):
        self.max_missed = max_missed
        self.threshold = threshold
        self.next_id = 1
        self.tracks: dict[str, dict[str, dict]] = {}

    def update(self, camera: str, detections: list[dict], ts: datetime):
        tracks = self.tracks.setdefault(camera, {})
        candidates = [d for d in detections if d.get("label") in plates.VEHICLE_LABELS]
        assigned: set[str] = set()
        active: list[TrackedObject] = []

        for detection in candidates:
            box = tuple(map(int, detection["box"]))
            score = float(detection["score"])
            label = str(detection["label"])
            best_id, best_overlap = None, 0.0
            for track_id, track in tracks.items():
                if track_id in assigned:
                    continue
                overlap = box_iou(track["box"], box)
                if overlap > best_overlap:
                    best_id, best_overlap = track_id, overlap
            if best_id is None or best_overlap < self.threshold:
                best_id = f"{camera}-vehicle{self.next_id}"
                self.next_id += 1
                tracks[best_id] = {"box": box, "label": label, "score": score, "missed": 0,
                                   "motionless": 0, "history": [box], "first_seen": ts}
            else:
                track = tracks[best_id]
                track["motionless"] = track["motionless"] + 1 if box_iou(track["box"], box) >= 0.8 else 0
                track.update(box=box, label=label, score=score, missed=0)
                track["history"].append(box)
                track["history"] = track["history"][-25:]
            assigned.add(best_id)
            track = tracks[best_id]
            active.append(TrackedObject(
                id=best_id, label=label, box=box, score=score, median_score=score,
                true_positive=score >= 0.4, stationary=track["motionless"] >= 10,
                motionless_count=track["motionless"], first_seen=track["first_seen"],
                last_seen=ts, history=list(track["history"]),
            ))

        ended: list[TrackedObject] = []
        for track_id, track in list(tracks.items()):
            if track_id in assigned:
                continue
            track["missed"] += 1
            if track["missed"] >= self.max_missed:
                ended.append(TrackedObject(
                    id=track_id, label=track["label"], box=track["box"], score=track["score"],
                    median_score=track["score"], true_positive=True,
                    stationary=track["motionless"] >= 10,
                    motionless_count=track["motionless"], first_seen=track["first_seen"],
                    last_seen=ts, history=list(track["history"]),
                ))
                del tracks[track_id]
        return active, ended
