"""Intrusion detection plugin.

Raises an event when a **person** appears on a camera during armed hours
(e.g. night-time, or when everyone is away).

Architecture
------------
1.  **Cheap motion gate** – inspired by Frigate's ``improved_motion.py``.
    A background-subtraction check runs on every frame; the heavy person
    detector is *only* invoked when motion is detected.
2.  **Shared YOLOX detector** – uses the project-wide detector provided
    by ``cctv_ai`` (no extra model files).
3.  **Simple IoU tracker** – tracks persons across frames so that one
    intruder produces one event, not one per frame.
4.  **Armed schedule** – detections outside the armed window are silently
    ignored.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, time as dt_time
from typing import Any

import cv2
import numpy as np

from cctv_ai.plugin import Event, Plugin as Base

from .motion import MotionDetector
from .tracker import SimpleTracker, TrackedPerson

logger = logging.getLogger(__name__)


class Plugin(Base):
    """Intrusion-detection plugin for cctv_ai."""

    use_case = "intrusion"
    enabled = True
    needs = {"video"}
    interval = 1.0  # seconds between processed frames

    # ── Armed schedule (class-level defaults; move to config later) ──
    # During these hours the plugin is "armed" and will fire events.
    # Use 24-hour time.  If start > end the window wraps past midnight.
    armed_start: dt_time = dt_time(22, 0)   # 10 PM
    armed_end:   dt_time = dt_time(6, 0)    # 6 AM

    # ── Detection thresholds ─────────────────────────────────────────
    person_confidence: float = 0.45         # min confidence from YOLOX
    cooldown_seconds: float = 60.0          # per-camera cooldown

    # ── Motion detector settings ─────────────────────────────────────
    motion_threshold: int = 25              # pixel-diff threshold
    motion_min_area: int = 500              # ignore tiny contours
    motion_history: int = 30                # bg-subtractor history

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------

    def setup(self, cameras: list[str]) -> None:
        """Initialise motion detectors, tracker, and reference the
        shared YOLOX detector (loaded once by the cctv_ai runtime).
        """
        # The shared detector is injected by the cctv_ai runtime at
        # ``self.detector``.  It exposes:
        #     self.detector.detect(frame) -> list[Detection]
        # where Detection has .label, .confidence, .box (x1,y1,x2,y2).

        # Per-camera motion detectors
        self._motion: dict[str, MotionDetector] = {}
        for cam in cameras:
            self._motion[cam] = MotionDetector(
                threshold=self.motion_threshold,
                min_area=self.motion_min_area,
                history=self.motion_history,
            )

        # Per-camera IoU trackers
        self._tracker: dict[str, SimpleTracker] = {}
        for cam in cameras:
            self._tracker[cam] = SimpleTracker(
                max_disappeared=5,  # frames before dropping a track
                iou_threshold=0.3,
            )

        # Per-camera cooldown tracking: {camera: last_event_monotonic}
        self._last_event_time: dict[str, float] = {}

        logger.info(
            "Intrusion plugin ready – armed %s→%s, %d camera(s).",
            self.armed_start,
            self.armed_end,
            len(cameras),
        )

    # ----------------------------------------------------------------
    # Per-frame processing
    # ----------------------------------------------------------------

    def on_frame(
        self,
        camera: str,
        frame: np.ndarray,
        ts: datetime,
    ) -> list[Event]:
        """Process one frame.  Returns intrusion events (if any)."""

        # 1) Check armed schedule ──────────────────────────────────────
        if not self._is_armed(ts):
            return []

        # 2) Cheap motion check ────────────────────────────────────────
        motion_det = self._motion.get(camera)
        if motion_det is None:
            # Camera appeared after setup(); create on-the-fly
            motion_det = MotionDetector(
                threshold=self.motion_threshold,
                min_area=self.motion_min_area,
                history=self.motion_history,
            )
            self._motion[camera] = motion_det

        motion_boxes = motion_det.detect(frame)
        if not motion_boxes:
            # No motion → no need to run the heavy detector.
            # Still update tracker so disappeared counts tick up.
            tracker = self._get_tracker(camera)
            tracker.update([])
            return []

        # 3) Run person detector ONLY on motion regions ────────────────
        person_boxes = self._detect_persons(frame, motion_boxes)

        # 4) Update tracker ────────────────────────────────────────────
        tracker = self._get_tracker(camera)
        new_persons = tracker.update(person_boxes)

        # 5) Fire events for NEW tracks only ───────────────────────────
        if not new_persons:
            return []

        # Per-camera cooldown
        now = time.monotonic()
        last = self._last_event_time.get(camera, 0.0)
        if (now - last) < self.cooldown_seconds:
            return []
        self._last_event_time[camera] = now

        events: list[Event] = []
        for person in new_persons:
            ev = self.event(
                camera,
                "person",
                person.confidence,
                data={"person_armed_hours": True},
            )
            events.append(ev)
            logger.info(
                "[%s] Intrusion – new person (track=%s, conf=%.2f)",
                camera,
                person.track_id,
                person.confidence,
            )

        return events

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _is_armed(self, ts: datetime) -> bool:
        """Return True if the current time falls within the armed window."""
        now = ts.time()
        if self.armed_start <= self.armed_end:
            # Simple range, e.g. 08:00 → 17:00
            return self.armed_start <= now <= self.armed_end
        else:
            # Wraps midnight, e.g. 22:00 → 06:00
            return now >= self.armed_start or now <= self.armed_end

    def _get_tracker(self, camera: str) -> SimpleTracker:
        """Get (or lazily create) the tracker for *camera*."""
        if camera not in self._tracker:
            self._tracker[camera] = SimpleTracker(
                max_disappeared=5,
                iou_threshold=0.3,
            )
        return self._tracker[camera]

    def _detect_persons(
        self,
        frame: np.ndarray,
        motion_boxes: list[tuple[int, int, int, int]],
    ) -> list[dict[str, Any]]:
        """Run the shared YOLOX detector and return person boxes.

        Following Frigate's pattern, we *only* run detection on regions
        where motion was found.  Each motion box is expanded slightly,
        cropped, detected, and the results mapped back to full-frame
        coordinates.
        """
        h, w = frame.shape[:2]
        person_boxes: list[dict[str, Any]] = []
        seen_centres: set[tuple[int, int]] = set()  # crude dedup

        for mx1, my1, mx2, my2 in motion_boxes:
            # Expand the motion region by 20 % for context
            rw, rh = mx2 - mx1, my2 - my1
            pad_x, pad_y = int(rw * 0.2), int(rh * 0.2)
            rx1 = max(0, mx1 - pad_x)
            ry1 = max(0, my1 - pad_y)
            rx2 = min(w, mx2 + pad_x)
            ry2 = min(h, my2 + pad_y)

            region = frame[ry1:ry2, rx1:rx2]
            if region.size == 0:
                continue

            # Use the shared detector
            detections = self.detector.detect(region)

            for det in detections:
                if det.label != "person":
                    continue
                if det.confidence < self.person_confidence:
                    continue

                # Map box back to full-frame coordinates
                bx1, by1, bx2, by2 = det.box
                abs_box = (
                    int(bx1 + rx1),
                    int(by1 + ry1),
                    int(bx2 + rx1),
                    int(by2 + ry1),
                )

                # Crude dedup: skip if centre is very close to one
                # we already added (overlapping motion regions).
                cx = (abs_box[0] + abs_box[2]) // 2
                cy = (abs_box[1] + abs_box[3]) // 2
                key = (cx // 20, cy // 20)  # quantise to 20 px grid
                if key in seen_centres:
                    continue
                seen_centres.add(key)

                person_boxes.append({
                    "box": abs_box,
                    "confidence": det.confidence,
                })

        return person_boxes
