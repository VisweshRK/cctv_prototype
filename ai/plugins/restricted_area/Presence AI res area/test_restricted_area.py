"""Tests for the restricted area CCTV plugin.

Run with:
    cd ai
    python -m pytest tests/test_restricted_area.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cctv_ai.plugins.restricted_area.zones import (
    Zone,
    ZoneManager,
    get_bottom_centre,
    point_in_polygon,
    scale_polygon,
)
from cctv_ai.plugins.restricted_area.tracker import SimpleTracker, TrackedPerson, _iou
from cctv_ai.plugins.restricted_area import Plugin, ZoneStay


# =========================================================================
# 1. Zone and Geometry Tests
# =========================================================================

class TestZoneGeometry:
    """Unit tests for coordinate scaling and point-in-polygon tests."""

    def test_get_bottom_centre(self):
        """Bottom-centre should be ((x1+x2)/2, y2)."""
        box = (100, 50, 200, 300)
        bc = get_bottom_centre(box)
        assert bc == (150.0, 300.0)

        # Zero-width or asymmetric box
        box2 = (0, 0, 80, 100)
        assert get_bottom_centre(box2) == (40.0, 100.0)

    def test_scale_polygon_standard(self):
        """Scaling relative normalized coordinates [0, 1] to pixel dimensions."""
        norm_poly = [(0.1, 0.2), (0.5, 0.2), (0.5, 0.8), (0.1, 0.8)]
        scaled = scale_polygon(norm_poly, width=1000, height=500)

        assert scaled.shape == (4, 2)
        np.testing.assert_array_equal(
            scaled,
            np.array([
                [100, 100],
                [500, 100],
                [500, 400],
                [100, 400],
            ], dtype=np.int32),
        )

    def test_scale_polygon_clamping(self):
        """Points outside [0, 1] range should be clamped."""
        out_poly = [(-0.1, 1.5), (0.5, 0.5)]
        scaled = scale_polygon(out_poly, width=100, height=100)
        np.testing.assert_array_equal(
            scaled,
            np.array([[0, 100], [50, 50]], dtype=np.int32),
        )

    def test_scale_empty_polygon(self):
        scaled = scale_polygon([], width=640, height=480)
        assert len(scaled) == 0

    def test_point_in_polygon(self):
        poly = np.array([[10, 10], [100, 10], [100, 100], [10, 100]], dtype=np.int32)

        # Inside
        assert point_in_polygon((50.0, 50.0), poly) is True
        # Outside
        assert point_in_polygon((5.0, 5.0), poly) is False
        assert point_in_polygon((150.0, 50.0), poly) is False
        # On edge
        assert point_in_polygon((10.0, 50.0), poly, include_edge=True) is True
        assert point_in_polygon((10.0, 50.0), poly, include_edge=False) is False

    def test_zone_caching_and_contains(self):
        zone = Zone(
            name="doorstep",
            polygon=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)],
        )
        # Width=1000, Height=1000 -> rect [200, 200, 800, 800]
        assert zone.contains_point((500.0, 500.0), 1000, 1000) is True
        assert zone.contains_point((100.0, 500.0), 1000, 1000) is False
        # Check caching
        assert (1000, 1000) in zone._scaled_cache

    def test_frigate_bottom_centre_pattern(self):
        """Frigate pattern: a person's head/torso can be inside the zone,
        but if their feet (bottom-centre) are outside, they are NOT in the zone.
        Conversely, if feet are inside, they ARE in the zone.
        """
        # Zone occupies bottom half of image: y from 0.5 to 1.0 (pixels 500 to 1000)
        zone = Zone(
            name="loading_bay",
            polygon=[(0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)],
        )
        w, h = 1000, 1000

        # Person standing in top half: y1=300, y2=490 (bottom-centre at 490 -> outside)
        box_outside = (400, 300, 600, 490)
        assert zone.contains_box_bottom_centre(box_outside, w, h) is False

        # Person standing just inside: y1=450, y2=510 (bottom-centre at 510 -> inside)
        box_inside = (400, 450, 600, 510)
        assert zone.contains_box_bottom_centre(box_inside, w, h) is True


class TestZoneManager:
    """Unit tests for ZoneManager multi-camera and multi-zone routing."""

    def test_load_and_query_zones(self):
        config = {
            "cam_front": {
                "porch": [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)],
                "driveway": [(0.5, 0.5), (0.9, 0.5), (0.9, 0.9), (0.5, 0.9)],
            },
            "cam_back": {
                "pool": [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)],
            },
        }
        zm = ZoneManager(config)

        # Check cam_front porch
        # Box bottom-centre at (300, 300) in 1000x1000 frame -> inside porch [100-500, 100-500]
        box_porch = (250, 150, 350, 300)
        zones = zm.find_zones_for_box("cam_front", box_porch, 1000, 1000)
        assert zones == ["porch"]

        # Box bottom-centre at (700, 700) -> inside driveway [500-900, 500-900]
        box_driveway = (650, 600, 750, 700)
        zones = zm.find_zones_for_box("cam_front", box_driveway, 1000, 1000)
        assert zones == ["driveway"]

        # Box in unzoned region -> empty
        box_nowhere = (50, 50, 90, 90)
        assert zm.find_zones_for_box("cam_front", box_nowhere, 1000, 1000) == []

        # Camera with no zones
        assert zm.find_zones_for_box("cam_unknown", box_porch, 1000, 1000) == []


# =========================================================================
# 2. Tracker Tests
# =========================================================================

class TestTracker:
    """Unit tests for IoU-based tracking with bottom-centre."""

    def test_new_detections_and_active_tracks(self):
        tracker = SimpleTracker(max_disappeared=3, iou_threshold=0.3)
        dets = [
            {"box": (10, 10, 50, 100), "confidence": 0.9},
            {"box": (200, 200, 300, 400), "confidence": 0.8},
        ]
        active = tracker.update(dets)
        assert len(active) == 2
        assert all(t.is_new for t in active)

        # Check bottom_centre
        t0 = active[0]
        assert t0.bottom_centre == (30.0, 100.0)

    def test_tracking_across_frames(self):
        tracker = SimpleTracker(max_disappeared=3, iou_threshold=0.3)
        # Frame 1
        active1 = tracker.update([{"box": (10, 10, 50, 100), "confidence": 0.9}])
        assert len(active1) == 1
        tid = active1[0].track_id

        # Frame 2: slight movement
        active2 = tracker.update([{"box": (12, 14, 52, 104), "confidence": 0.92}])
        assert len(active2) == 1
        assert active2[0].track_id == tid
        assert not active2[0].is_new
        assert active2[0].disappeared == 0

    def test_track_pruning_after_disappeared(self):
        tracker = SimpleTracker(max_disappeared=2, iou_threshold=0.3)
        tracker.update([{"box": (10, 10, 50, 100), "confidence": 0.9}])
        assert len(tracker.tracks) == 1

        tracker.update([])  # disappeared = 1
        assert len(tracker.tracks) == 1
        tracker.update([])  # disappeared = 2
        assert len(tracker.tracks) == 1
        tracker.update([])  # disappeared = 3 > 2 -> pruned
        assert len(tracker.tracks) == 0


# =========================================================================
# 3. Inertia & Loitering Logic Tests
# =========================================================================

class TestInertiaAndLoitering:
    """Unit tests for inertia and loitering event generation."""

    def _make_plugin(self, inertia=3, loitering_time=10.0) -> Plugin:
        p = Plugin.__new__(Plugin)
        p.use_case = "restricted_area"
        p.enabled = True
        p.person_confidence = 0.45
        p.inertia = inertia
        p.inertia_seconds = 0.0
        p.loitering_time = loitering_time
        p.loitering_frames = 0
        p.use_motion_gate = False  # test logic directly
        p._motion = {}
        p._tracker = {}
        p._stays = {}

        # Set up a zone on cam1: normalized rect [0.1, 0.1, 0.9, 0.9]
        p._zone_manager = ZoneManager({
            "cam1": {
                "danger_zone": [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)],
            }
        })

        # We can use the real self.event or mock it cleanly
        return p

    def test_inertia_requires_consecutive_frames(self):
        """A person must stay in zone for `inertia` consecutive frames before zone_entered."""
        p = self._make_plugin(inertia=3)
        cam = "cam1"
        w, h = 1000, 1000
        # Box inside danger_zone: bottom-centre at (500, 500)
        track = TrackedPerson(
            track_id="trk1",
            box=(400, 300, 600, 500),
            confidence=0.88,
            disappeared=0,
            is_new=True,
        )

        base_time = datetime(2026, 10, 1, 12, 0, 0)

        # Frame 1: frames_in_zone = 1 < 3 -> no event
        ev1 = p._evaluate_zones(cam, [track], w, h, base_time)
        assert ev1 == []

        # Frame 2: frames_in_zone = 2 < 3 -> no event
        ev2 = p._evaluate_zones(cam, [track], w, h, base_time + timedelta(seconds=1))
        assert ev2 == []

        # Frame 3: frames_in_zone = 3 == 3 -> zone_entered event fired!
        ev3 = p._evaluate_zones(cam, [track], w, h, base_time + timedelta(seconds=2))
        assert len(ev3) == 1
        assert ev3[0].label == "zone_entered"
        assert ev3[0].data["zone"] == "danger_zone"
        assert ev3[0].data["track_id"] == "trk1"
        assert ev3[0].data["frames_in_zone"] == 3

        # Frame 4: frames_in_zone = 4 -> entered already fired, no duplicate!
        ev4 = p._evaluate_zones(cam, [track], w, h, base_time + timedelta(seconds=3))
        assert ev4 == []

    def test_inertia_reset_if_exits_before_threshold(self):
        """If a person leaves the zone before reaching inertia frames, no event fires."""
        p = self._make_plugin(inertia=3)
        cam = "cam1"
        w, h = 1000, 1000

        track_in = TrackedPerson(
            track_id="trk1", box=(400, 300, 600, 500), confidence=0.85
        )
        track_out = TrackedPerson(
            track_id="trk1", box=(10, 10, 50, 50), confidence=0.85
        )

        t0 = datetime(2026, 10, 1, 12, 0, 0)

        # Frame 1: inside (count=1)
        p._evaluate_zones(cam, [track_in], w, h, t0)
        # Frame 2: inside (count=2)
        p._evaluate_zones(cam, [track_in], w, h, t0 + timedelta(seconds=1))
        # Frame 3: moved outside zone!
        ev3 = p._evaluate_zones(cam, [track_out], w, h, t0 + timedelta(seconds=2))
        assert ev3 == []
        # Stays for trk1 should be cleared
        assert ("trk1", "danger_zone") not in p._stays[cam]

        # Frame 4: re-enters zone -> starts from count=1
        ev4 = p._evaluate_zones(cam, [track_in], w, h, t0 + timedelta(seconds=3))
        assert ev4 == []
        assert p._stays[cam][("trk1", "danger_zone")].frames_in_zone == 1

    def test_loitering_triggers_after_threshold(self):
        """When duration in zone reaches loitering_time, fires loitering event."""
        p = self._make_plugin(inertia=2, loitering_time=5.0)
        cam = "cam1"
        w, h = 1000, 1000
        track = TrackedPerson(
            track_id="trk1", box=(400, 300, 600, 500), confidence=0.9
        )

        t0 = datetime(2026, 10, 1, 12, 0, 0)

        # Frame 1: entry
        p._evaluate_zones(cam, [track], w, h, t0)
        # Frame 2: inertia met -> zone_entered
        ev2 = p._evaluate_zones(cam, [track], w, h, t0 + timedelta(seconds=1))
        assert len(ev2) == 1
        assert ev2[0].label == "zone_entered"

        # Frame 3: t = 4s < 5s -> no loitering yet
        ev3 = p._evaluate_zones(cam, [track], w, h, t0 + timedelta(seconds=4))
        assert ev3 == []

        # Frame 4: t = 5.5s >= 5s -> loitering event fired!
        ev4 = p._evaluate_zones(cam, [track], w, h, t0 + timedelta(seconds=5, milliseconds=500))
        assert len(ev4) == 1
        assert ev4[0].label == "loitering"
        assert ev4[0].data["zone"] == "danger_zone"
        assert ev4[0].data["duration"] >= 5.0

        # Frame 5: t = 10s -> loitering already fired, no duplicate spam!
        ev5 = p._evaluate_zones(cam, [track], w, h, t0 + timedelta(seconds=10))
        assert ev5 == []

    def test_re_entry_after_stay_triggers_new_events(self):
        """If person leaves after firing zone_entered/loitering and re-enters,
        new events can be triggered.
        """
        p = self._make_plugin(inertia=2, loitering_time=5.0)
        cam = "cam1"
        w, h = 1000, 1000

        track_in = TrackedPerson(
            track_id="trk1", box=(400, 300, 600, 500), confidence=0.9
        )
        track_out = TrackedPerson(
            track_id="trk1", box=(10, 10, 50, 50), confidence=0.9
        )

        t0 = datetime(2026, 10, 1, 12, 0, 0)
        # 2 frames inside
        p._evaluate_zones(cam, [track_in], w, h, t0)
        p._evaluate_zones(cam, [track_in], w, h, t0 + timedelta(seconds=1))

        # Person steps outside
        p._evaluate_zones(cam, [track_out], w, h, t0 + timedelta(seconds=2))

        # Person re-enters
        t_reentry = t0 + timedelta(seconds=10)
        p._evaluate_zones(cam, [track_in], w, h, t_reentry)
        ev_new = p._evaluate_zones(cam, [track_in], w, h, t_reentry + timedelta(seconds=1))

        assert len(ev_new) == 1
        assert ev_new[0].label == "zone_entered"


# =========================================================================
# 4. Plugin End-to-End on_frame Integration Tests
# =========================================================================

class TestPluginOnFrame:
    """Integration tests for on_frame with mocked detector and motion gate."""

    def _setup_plugin(self, use_motion_gate=True) -> Plugin:
        p = Plugin()
        p.inertia = 2
        p.loitering_time = 4.0
        p.use_motion_gate = use_motion_gate
        p.setup(
            cameras=["cam_vault"],
            zones={
                "cam_vault": {
                    "safe_room": [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)],
                }
            },
        )
        # We can use the real self.event or mock it cleanly
        return p

    def test_no_events_on_static_empty_frame(self):
        """When scene is static and no persons exist, no events fire."""
        p = self._setup_plugin(use_motion_gate=True)
        frame = np.full((500, 500, 3), 120, dtype=np.uint8)
        ts = datetime(2026, 10, 1, 12, 0, 0)

        # Calibrate motion detector on identical frames
        for _ in range(15):
            events = p.on_frame("cam_vault", frame, ts)
            assert events == []

    def test_person_detected_outside_zone_fires_nothing(self):
        """Person is detected, but outside the safe_room zone -> no events."""
        p = self._setup_plugin(use_motion_gate=False)

        # Person at top-left corner: bottom-centre at (40, 80) in 500x500 -> outside [100-400]
        mock_det = SimpleNamespace(
            label="person",
            confidence=0.92,
            box=(20, 20, 60, 80),
        )
        p.detector = SimpleNamespace(detect=MagicMock(return_value=[mock_det]))

        frame = np.full((500, 500, 3), 120, dtype=np.uint8)
        ts = datetime(2026, 10, 1, 12, 0, 0)

        # Run multiple frames
        for i in range(5):
            events = p.on_frame("cam_vault", frame, ts + timedelta(seconds=i))
            assert events == []

    def test_full_lifecycle_entry_and_loiter(self):
        """Full lifecycle:
        1. Person appears inside zone
        2. Reaches inertia frames -> zone_entered event
        3. Stays beyond loitering_time -> loitering event
        4. Exits zone -> stay cleared
        """
        p = self._setup_plugin(use_motion_gate=False)

        # Person inside safe_room: bottom-centre at (250, 300) in 500x500
        person_inside = SimpleNamespace(
            label="person",
            confidence=0.95,
            box=(200, 150, 300, 300),
        )
        p.detector = SimpleNamespace(detect=MagicMock(return_value=[person_inside]))

        frame = np.full((500, 500, 3), 120, dtype=np.uint8)
        t0 = datetime(2026, 10, 1, 12, 0, 0)

        # Frame 1: count=1 < inertia(2) -> no events
        ev1 = p.on_frame("cam_vault", frame, t0)
        assert ev1 == []

        # Frame 2: count=2 == inertia(2) -> zone_entered!
        ev2 = p.on_frame("cam_vault", frame, t0 + timedelta(seconds=1))
        assert len(ev2) == 1
        assert ev2[0].label == "zone_entered"
        assert ev2[0].data["zone"] == "safe_room"

        # Frame 3: t = 2s < 4s -> no new events
        ev3 = p.on_frame("cam_vault", frame, t0 + timedelta(seconds=2))
        assert ev3 == []

        # Frame 4: t = 4.2s >= 4s -> loitering!
        ev4 = p.on_frame("cam_vault", frame, t0 + timedelta(seconds=4, milliseconds=200))
        assert len(ev4) == 1
        assert ev4[0].label == "loitering"
        assert ev4[0].data["zone"] == "safe_room"
        assert ev4[0].data["duration"] >= 4.0

        # Frame 5: person steps across the boundary (feet move outside safe_room: y2=410 > 400)
        # Previous box was (200, 150, 300, 300); moving to (200, 220, 300, 370) then (200, 270, 300, 420)
        person_stepping_out = SimpleNamespace(
            label="person",
            confidence=0.95,
            box=(200, 220, 300, 370),
        )
        p.detector.detect.return_value = [person_stepping_out]
        p.on_frame("cam_vault", frame, t0 + timedelta(seconds=5))

        # Person's feet now cross the boundary: y2 = 420 > 400
        person_outside = SimpleNamespace(
            label="person",
            confidence=0.95,
            box=(200, 270, 300, 420),
        )
        p.detector.detect.return_value = [person_outside]
        ev6 = p.on_frame("cam_vault", frame, t0 + timedelta(seconds=6))
        assert ev6 == []

        # Stays for vault should be cleared because track left the zone
        assert len(p._stays["cam_vault"]) == 0
