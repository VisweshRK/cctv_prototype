"""Vehicle + licence plate recognition: read plates at the gate
(plan: docs/plans/p2-krithick.md).

OCR: RapidOCR (PaddleOCR det/cls/rec ONNX, Apache 2.0, models bundled in the
pip package). Matching: rapidfuzz (MIT). One event per vehicle track, with
the plate confirmed by several agreeing reads.

Today's worker calls on_frame() with the whole frame, so this plugin runs in
a demo mode: OCR on the full frame (throttled), plate-shaped text boxes only,
reads clustered per camera over the last max_reads frames. When Viswesh's
pipeline lands (TrackedObject + on_objects, see docs/plans/README.md), set
needs = {"objects"} and the real path below takes over: reads per vehicle
track, event when the plate is confirmed by >= confirm_reads reads or the
track ends - exactly like Frigate.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rapidfuzz.distance import Levenshtein

from cctv_ai.plugin import Event, Plugin as Base

from . import plates
from .clustering import PlateRead, best_plate
from .ocr import PlateOCR, enhance_contrast, prepare_crop

log = logging.getLogger("cctv_ai.vehicle_lpr")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class Plugin(Base):
    use_case = "vehicle_lpr"
    enabled = True  # stays silent until the OCR models are present
    needs = {"video"}  # switch to {"objects"} when the shared tracker lands
    interval = 1.0
    heavy = True  # the pipeline should run this plugin on its own thread

    # tuning (see docs/plans/p2-krithick.md)
    min_plate_area = 1000  # px of a text box to count as a plate candidate
    min_read_confidence = 0.80  # RapidOCR avg char confidence; 0.85 dropped clean renders too
    min_plate_length = plates.MIN_PLATE_LENGTH
    cluster_threshold = 0.85  # Jaro-Winkler similarity to group reads
    confirm_reads = 3  # agreeing reads before the plate is announced
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

    def _reads_from_lines(self, lines: list[tuple[tuple[float, float, float, float], str, float]],
                          frame_shape: tuple[int, int] | None = None) -> list[PlateRead]:
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
            fractions = None
            if frame_shape is not None:
                fh, fw = frame_shape
                fractions = (
                    max(0.0, x / fw), max(0.0, y / fh),
                    min(1.0, (x + w) / fw), min(1.0, (y + h) / fh),
                )
            reads.append(PlateRead(plate, conf, int(w * h), fractions))
        return reads

    def _event(self, camera: str, plate: str, conf: float, reads: int,
               vehicle: str | None = None,
               box: tuple[float, float, float, float] | None = None) -> Event:
        data: dict[str, Any] = {
            "plate": plate,
            "vehicle": vehicle,
            "reads": reads,
            "known_as": self.known_as(plate),
        }
        if box:
            data["box"] = list(box)
        event = self.event(camera, "vehicle", conf, data=data)
        log.info("%s: plate %s conf %.2f from %d reads", camera, plate, conf, reads)
        return event

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
        self.reads[camera].extend(
            self._reads_from_lines(self.ocr.detect_text(frame), frame.shape[:2])
        )

        result = best_plate(list(self.reads[camera]), self.cluster_threshold)
        if result is None or result[2] < self.confirm_reads:
            return []
        plate, conf, agreeing = result
        last = self.announced[camera].get(plate)
        if last is not None and now - last < self.duplicate_window:
            return []
        self.announced[camera][plate] = now
        box = next((r.box for r in self.reads[camera] if r.plate == plate and r.box), None)
        return [self._event(camera, plate, conf, agreeing, box=box)]

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
            if result and result[2] >= 1:
                events.append(self._event(camera, result[0], result[1], result[2],
                                          vehicle=obj.label))

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
                events.append(self._event(camera, result[0], result[1], result[2],
                                          vehicle=obj.label))
        return events
