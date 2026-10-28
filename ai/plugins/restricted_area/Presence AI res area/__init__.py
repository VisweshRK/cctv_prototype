"""Restricted area monitoring CCTV plugin.

Raises an event when a person enters, or stays too long (loitering) in,
a marked polygon zone of an image.

Key Features
------------
1. **Events**:
   - use_case: "restricted_area"
   - Labels: "zone_entered" and "loitering"
   - Data payload includes zone name (e.g. data={"zone": zone_name, ...}).
2. **Zones & Coordinates**:
   - Defined as relative normalized coordinates (0.0 to 1.0) per camera.
   - Evaluated using bottom-centre point of bounding box via cv2.pointPolygonTest
     (following Frigate's pattern).
3. **Inertia & Loitering**:
   - Inertia: requires a person to be inside for multiple consecutive frames / seconds
     before triggering "zone_entered" to avoid false positives.
   - Loitering: triggers a "loitering" event if a track stays inside the restricted
     zone longer than a configured threshold.
4. **Shared Detector & Motion Gate**:
   - Reuses project-wide shared YOLOX detector injected at self.detector.
   - Optional cheap background subtraction motion gate to skip static frames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

import cv2
import numpy as np

from cctv_ai.plugin import Event, Plugin as Base

from .motion import MotionDetector
from .tracker import SimpleTracker, TrackedPerson
from .zones import ZoneManager, get_bottom_centre

logger = logging.getLogger(__name__)


@dataclass
class ZoneStay:
    """State of a tracked person inside a specific restricted zone."""

    zone: str
    track_id: str
    first_seen: datetime
    last_seen: datetime
    frames_in_zone: int = 1
    entered_fired: bool = False
    loitering_fired: bool = False


class Plugin(Base):
    """Restricted area and loitering detection plugin."""

    use_case = "restricted_area"
    enabled = True
    needs = {"video"}
    interval = 1.0  # seconds between processed frames

    # ── Detection & Thresholds ───────────────────────────────────────────
    person_confidence: float = 0.45

    # ── Inertia & Loitering Settings ─────────────────────────────────────
    # Inertia: frames or seconds required in zone before "zone_entered"
    inertia: int = 3                # consecutive frames before zone_entered
    inertia_seconds: float = 0.0    # alternative/additional seconds threshold

    # Loitering: time required in zone before "loitering" event fires
    loitering_time: float = 30.0    # seconds in zone to trigger loitering
    loitering_frames: int = 0       # optional frame count alternative (0 = disabled)

    # ── Motion Detector Gate ─────────────────────────────────────────────
    use_motion_gate: bool = True
    motion_threshold: int = 25
    motion_min_area: int = 500
    motion_history: int = 30

    # ── Zones Configuration ──────────────────────────────────────────────
    # Format: {camera_name: {zone_name: [(x1, y1), (x2, y2), ...]}}
    zones: dict[str, Any] = {}

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------

    def setup(
        self,
        cameras: list[str],
        zones: dict[str, Any] | None = None,
    ) -> None:
        """Initialise motion detectors, trackers, and zone manager."""
        if zones is not None:
            self.zones = zones

        self._zone_manager = ZoneManager(self.zones)

        # Per-camera motion detectors
        self._motion: dict[str, MotionDetector] = {}
        for cam in cameras:
            self._motion[cam] = MotionDetector(
                threshold=self.motion_threshold,
                min_area=self.motion_min_area,
                history=self.motion_history,
            )

        # Per-camera trackers
        self._tracker: dict[str, SimpleTracker] = {}
        for cam in cameras:
            self._tracker[cam] = SimpleTracker(
                max_disappeared=5,
                iou_threshold=0.3,
            )

        # Per-camera stays: {camera: {(track_id, zone_name): ZoneStay}}
        self._stays: dict[str, dict[tuple[str, str], ZoneStay]] = {}
        for cam in cameras:
            self._stays[cam] = {}

        logger.info(
            "RestrictedArea plugin ready: %d camera(s), inertia=%d frames, loiter=%.1fs",
            len(cameras),
            self.inertia,
            self.loitering_time,
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
        """Process one frame and return zone_entered / loitering events."""
        cam_zones = self._zone_manager.get_zones(camera)
        if not cam_zones:
            # No restricted zones configured for this camera
            return []

        h, w = frame.shape[:2]
        tracker = self._get_tracker(camera)

        # 1) Cheap motion check ────────────────────────────────────────
        motion_boxes: list[tuple[int, int, int, int]] = []
        if self.use_motion_gate:
            motion_det = self._get_motion_detector(camera)
            motion_boxes = motion_det.detect(frame)

            # If no motion and no existing tracks to monitor, skip
            if not motion_boxes and not tracker.tracks:
                tracker.update([])
                self._prune_stays(camera, tracker)
                return []

        # 2) Run person detector ───────────────────────────────────────
        if self.use_motion_gate and motion_boxes:
            # Search in motion regions plus regions around active tracks
            search_regions = list(motion_boxes)
            for track in tracker.tracks.values():
                search_regions.append(track.box)
            person_boxes = self._detect_persons(frame, search_regions)
        elif self.use_motion_gate and tracker.tracks:
            # No motion but active tracks exist -> monitor tracked persons
            search_regions = [track.box for track in tracker.tracks.values()]
            person_boxes = self._detect_persons(frame, search_regions)
        else:
            # Motion gate disabled: run on full frame
            person_boxes = self._detect_persons(frame, None)

        # 3) Update tracker ────────────────────────────────────────────
        active_tracks = tracker.update(person_boxes)

        # 4) Evaluate Zone Presence, Inertia & Loitering ────────────────
        events = self._evaluate_zones(camera, active_tracks, w, h, ts)

        # 5) Prune stays for tracks that disappeared ───────────────────
        self._prune_stays(camera, tracker)

        return events

    # ----------------------------------------------------------------
    # Internal Logic
    # ----------------------------------------------------------------

    def _evaluate_zones(
        self,
        camera: str,
        active_tracks: list[TrackedPerson],
        width: int,
        height: int,
        ts: datetime,
    ) -> list[Event]:
        """Check all active tracks against camera zones, firing events as thresholds are reached."""
        if camera not in self._stays:
            self._stays[camera] = {}
        cam_stays = self._stays[camera]

        events: list[Event] = []
        current_in_zones: set[tuple[str, str]] = set()

        for track in active_tracks:
            # Bottom-centre of person represents foot placement
            bc = get_bottom_centre(track.box)
            matched_zones = self._zone_manager.find_zones_for_point(
                camera, bc, width, height, include_edge=True
            )

            for zone_name in matched_zones:
                stay_key = (track.track_id, zone_name)
                current_in_zones.add(stay_key)

                if stay_key not in cam_stays:
                    # New zone entry
                    stay = ZoneStay(
                        zone=zone_name,
                        track_id=track.track_id,
                        first_seen=ts,
                        last_seen=ts,
                        frames_in_zone=1,
                    )
                    cam_stays[stay_key] = stay
                else:
                    # Ongoing zone stay
                    stay = cam_stays[stay_key]
                    stay.frames_in_zone += 1
                    stay.last_seen = ts

                duration = (ts - stay.first_seen).total_seconds()

                # Check Inertia for zone_entered event
                if not stay.entered_fired:
                    frames_ok = stay.frames_in_zone >= self.inertia
                    time_ok = (
                        self.inertia_seconds <= 0.0
                        or duration >= self.inertia_seconds
                    )
                    if frames_ok and time_ok:
                        stay.entered_fired = True
                        ev = self.event(
                            camera=camera,
                            label="zone_entered",
                            confidence=track.confidence,
                            data={
                                "zone": zone_name,
                                "track_id": track.track_id,
                                "box": list(track.box),
                                "bottom_centre": [round(bc[0], 1), round(bc[1], 1)],
                                "frames_in_zone": stay.frames_in_zone,
                            },
                        )
                        events.append(ev)
                        logger.info(
                            "[%s] Zone entered: %s by track %s (conf=%.2f, frames=%d)",
                            camera,
                            zone_name,
                            track.track_id,
                            track.confidence,
                            stay.frames_in_zone,
                        )

                # Check Loitering
                if not stay.loitering_fired:
                    time_ok = duration >= self.loitering_time
                    frames_ok = (
                        self.loitering_frames > 0
                        and stay.frames_in_zone >= self.loitering_frames
                    )
                    if time_ok or frames_ok:
                        stay.loitering_fired = True
                        ev = self.event(
                            camera=camera,
                            label="loitering",
                            confidence=track.confidence,
                            data={
                                "zone": zone_name,
                                "track_id": track.track_id,
                                "box": list(track.box),
                                "bottom_centre": [round(bc[0], 1), round(bc[1], 1)],
                                "duration": round(duration, 2),
                                "frames_in_zone": stay.frames_in_zone,
                            },
                        )
                        events.append(ev)
                        logger.info(
                            "[%s] Loitering in zone: %s by track %s (duration=%.1fs, frames=%d)",
                            camera,
                            zone_name,
                            track.track_id,
                            duration,
                            stay.frames_in_zone,
                        )

        # Handle exits: if a track is active but left a zone, remove its stay
        for track in active_tracks:
            for (tid, zname) in list(cam_stays.keys()):
                if tid == track.track_id and (tid, zname) not in current_in_zones:
                    del cam_stays[(tid, zname)]

        return events

    def _prune_stays(self, camera: str, tracker: SimpleTracker) -> None:
        """Remove stays for tracks that no longer exist in the tracker."""
        if camera not in self._stays:
            return
        active_ids = set(tracker.tracks.keys())
        cam_stays = self._stays[camera]
        for (tid, zname) in list(cam_stays.keys()):
            if tid not in active_ids:
                del cam_stays[(tid, zname)]

    def _detect_persons(
        self,
        frame: np.ndarray,
        regions: list[tuple[int, int, int, int]] | None,
    ) -> list[dict[str, Any]]:
        """Run shared YOLOX detector on frame or cropped regions."""
        if self.detector is None:
            return []

        h, w = frame.shape[:2]
        person_boxes: list[dict[str, Any]] = []

        if regions is None:
            # Full frame detection
            detections = self.detector.detect(frame)
            for det in detections:
                if det.label != "person" or det.confidence < self.person_confidence:
                    continue
                person_boxes.append({
                    "box": tuple(int(v) for v in det.box[:4]),
                    "confidence": float(det.confidence),
                })
            return person_boxes

        # Regional detection with deduplication
        seen_centres: set[tuple[int, int]] = set()

        for rx1, ry1, rx2, ry2 in regions:
            # Pad region by 20%
            rw, rh = rx2 - rx1, ry2 - ry1
            pad_x, pad_y = int(rw * 0.2), int(rh * 0.2)
            cx1 = max(0, rx1 - pad_x)
            cy1 = max(0, ry1 - pad_y)
            cx2 = min(w, rx2 + pad_x)
            cy2 = min(h, ry2 + pad_y)

            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            detections = self.detector.detect(crop)
            for det in detections:
                if det.label != "person" or det.confidence < self.person_confidence:
                    continue

                bx1, by1, bx2, by2 = det.box
                abs_box = (
                    int(bx1 + cx1),
                    int(by1 + cy1),
                    int(bx2 + cx1),
                    int(by2 + cy1),
                )

                # Dedup overlapping regions via 20px grid
                center_key = (
                    ((abs_box[0] + abs_box[2]) // 2) // 20,
                    ((abs_box[1] + abs_box[3]) // 2) // 20,
                )
                if center_key in seen_centres:
                    continue
                seen_centres.add(center_key)

                person_boxes.append({
                    "box": abs_box,
                    "confidence": float(det.confidence),
                })

        return person_boxes

    def _get_tracker(self, camera: str) -> SimpleTracker:
        """Get or lazily create SimpleTracker for camera."""
        if camera not in self._tracker:
            self._tracker[camera] = SimpleTracker(
                max_disappeared=5,
                iou_threshold=0.3,
            )
        return self._tracker[camera]

    def _get_motion_detector(self, camera: str) -> MotionDetector:
        """Get or lazily create MotionDetector for camera."""
        if camera not in self._motion:
            self._motion[camera] = MotionDetector(
                threshold=self.motion_threshold,
                min_area=self.motion_min_area,
                history=self.motion_history,
            )
        return self._motion[camera]
