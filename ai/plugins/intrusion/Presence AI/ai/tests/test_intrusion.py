"""Tests for the intrusion-detection plugin.

Run with:
    cd ai
    python -m pytest tests/test_intrusion.py -v
"""

from __future__ import annotations

from datetime import datetime, time as dt_time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Import the modules under test ────────────────────────────────────────
from cctv_ai.plugins.intrusion.motion import MotionDetector
from cctv_ai.plugins.intrusion.tracker import SimpleTracker, TrackedPerson, _iou
from cctv_ai.plugins.intrusion import Plugin


# =========================================================================
# Motion detector
# =========================================================================

class TestMotionDetector:
    """Unit tests for the cheap motion gate."""

    def _make_frame(self, w: int = 640, h: int = 480, val: int = 128) -> np.ndarray:
        """Create a solid-colour BGR frame."""
        return np.full((h, w, 3), val, dtype=np.uint8)

    def test_no_motion_on_identical_frames(self):
        """Identical frames should produce zero motion boxes after calibration."""
        md = MotionDetector(history=5)
        frame = self._make_frame()
        # Let the background model calibrate
        for _ in range(10):
            boxes = md.detect(frame)
        # After calibration, static scene → no motion
        boxes = md.detect(frame)
        assert boxes == []

    def test_motion_detected_on_changed_frame(self):
        """A large change in the frame should produce at least one box."""
        md = MotionDetector(history=5, min_area=50)
        base = self._make_frame(val=100)
        # Calibrate on base frame
        for _ in range(10):
            md.detect(base)

        # Introduce a big bright block
        changed = base.copy()
        changed[100:300, 100:400] = 255
        boxes = md.detect(changed)
        assert len(boxes) >= 1, "Expected at least one motion box"

        # Verify the box roughly covers the changed region
        x1, y1, x2, y2 = boxes[0]
        assert x1 < 200 and y1 < 200, "Box should start near the change"
        assert x2 > 200 and y2 > 200, "Box should extend over the change"


# =========================================================================
# IoU tracker
# =========================================================================

class TestIoU:
    """Tests for the IoU utility function."""

    def test_identical_boxes(self):
        assert _iou((0, 0, 100, 100), (0, 0, 100, 100)) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_partial_overlap(self):
        iou = _iou((0, 0, 10, 10), (5, 5, 15, 15))
        # intersection = 5*5=25, union = 100+100-25=175
        assert iou == pytest.approx(25 / 175, abs=1e-4)

    def test_zero_area_box(self):
        assert _iou((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0


class TestSimpleTracker:
    """Tests for the IoU-based person tracker."""

    def test_new_detections_are_returned(self):
        tracker = SimpleTracker(max_disappeared=3, iou_threshold=0.3)
        dets = [
            {"box": (10, 10, 50, 50), "confidence": 0.9},
            {"box": (200, 200, 300, 300), "confidence": 0.8},
        ]
        new = tracker.update(dets)
        assert len(new) == 2
        assert all(t.is_new for t in new)

    def test_matched_track_not_returned_as_new(self):
        tracker = SimpleTracker(max_disappeared=3, iou_threshold=0.3)
        det1 = [{"box": (10, 10, 50, 50), "confidence": 0.9}]
        tracker.update(det1)

        # Same-ish position → should match, NOT return as new
        det2 = [{"box": (12, 12, 52, 52), "confidence": 0.88}]
        new = tracker.update(det2)
        assert len(new) == 0

    def test_track_disappears_after_max_frames(self):
        tracker = SimpleTracker(max_disappeared=2, iou_threshold=0.3)
        tracker.update([{"box": (10, 10, 50, 50), "confidence": 0.9}])
        assert len(tracker.tracks) == 1

        # No detections for 3 frames → track should be pruned
        for _ in range(3):
            tracker.update([])
        assert len(tracker.tracks) == 0

    def test_second_person_is_new(self):
        tracker = SimpleTracker(max_disappeared=3, iou_threshold=0.3)
        tracker.update([{"box": (10, 10, 50, 50), "confidence": 0.9}])

        # A completely different location → new person
        new = tracker.update([
            {"box": (10, 10, 50, 50), "confidence": 0.9},   # existing
            {"box": (400, 400, 500, 500), "confidence": 0.7},  # new
        ])
        assert len(new) == 1
        assert new[0].is_new


# =========================================================================
# Plugin (armed schedule + integration)
# =========================================================================

class TestArmedSchedule:
    """Tests for the armed-hours logic."""

    def _make_plugin(self, start: dt_time, end: dt_time) -> Plugin:
        """Create a Plugin instance with a custom armed window."""
        p = Plugin.__new__(Plugin)
        p.armed_start = start
        p.armed_end = end
        return p

    def test_armed_during_night(self):
        """22:00–06:00 window should be armed at midnight."""
        p = self._make_plugin(dt_time(22, 0), dt_time(6, 0))
        midnight = datetime(2026, 1, 1, 0, 30)
        assert p._is_armed(midnight) is True

    def test_not_armed_during_day(self):
        """22:00–06:00 window should NOT be armed at noon."""
        p = self._make_plugin(dt_time(22, 0), dt_time(6, 0))
        noon = datetime(2026, 1, 1, 12, 0)
        assert p._is_armed(noon) is False

    def test_armed_at_start_boundary(self):
        p = self._make_plugin(dt_time(22, 0), dt_time(6, 0))
        edge = datetime(2026, 1, 1, 22, 0)
        assert p._is_armed(edge) is True

    def test_simple_daytime_range(self):
        """08:00–17:00 range (no midnight wrap)."""
        p = self._make_plugin(dt_time(8, 0), dt_time(17, 0))
        assert p._is_armed(datetime(2026, 1, 1, 10, 0)) is True
        assert p._is_armed(datetime(2026, 1, 1, 20, 0)) is False


class TestPluginOnFrame:
    """Integration-style test for on_frame (with mocked detector)."""

    def _make_plugin(self) -> Plugin:
        p = Plugin.__new__(Plugin)
        p.use_case = "intrusion"
        p.armed_start = dt_time(0, 0)   # always armed
        p.armed_end = dt_time(23, 59)
        p.person_confidence = 0.4
        p.cooldown_seconds = 0  # no cooldown in tests
        p.motion_threshold = 25
        p.motion_min_area = 50
        p.motion_history = 5
        p._motion = {}
        p._tracker = {}
        p._last_event_time = {}

        # Mock the self.event() helper from the base class
        p.event = MagicMock(
            side_effect=lambda cam, label, conf, **kw: SimpleNamespace(
                camera=cam, label=label, confidence=conf, data=kw.get("data")
            )
        )
        return p

    def test_no_event_when_not_armed(self):
        p = self._make_plugin()
        p.armed_start = dt_time(22, 0)
        p.armed_end = dt_time(6, 0)

        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        ts = datetime(2026, 6, 15, 12, 0)  # noon → not armed
        events = p.on_frame("cam1", frame, ts)
        assert events == []

    def test_no_event_when_no_motion(self):
        p = self._make_plugin()
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        ts = datetime(2026, 6, 15, 1, 0)  # 1 AM → armed

        # Feed identical frames (calibrate + test)
        for _ in range(10):
            events = p.on_frame("cam1", frame, ts)
        assert events == []

    def test_event_on_person_detection(self):
        """When motion + person detection → should fire an event."""
        p = self._make_plugin()

        # Mock the shared detector to return a person
        mock_det = SimpleNamespace(
            label="person",
            confidence=0.85,
            box=(10, 10, 60, 120),
        )
        p.detector = SimpleNamespace(detect=MagicMock(return_value=[mock_det]))

        # Mock the motion detector to always report motion
        mock_motion = MagicMock()
        mock_motion.detect.return_value = [(0, 0, 640, 480)]
        p._motion["cam1"] = mock_motion

        ts = datetime(2026, 6, 15, 2, 0)  # armed
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        events = p.on_frame("cam1", frame, ts)

        assert len(events) == 1
        p.event.assert_called_once()
        call_args = p.event.call_args
        assert call_args[0][1] == "person"  # label
        assert call_args[1]["data"] == {"person_armed_hours": True}

    def test_no_duplicate_event_for_same_person(self):
        """Same person on consecutive frames → only 1 event (tracking)."""
        p = self._make_plugin()
        mock_det = SimpleNamespace(
            label="person", confidence=0.85, box=(10, 10, 60, 120)
        )
        p.detector = SimpleNamespace(detect=MagicMock(return_value=[mock_det]))
        mock_motion = MagicMock()
        mock_motion.detect.return_value = [(0, 0, 640, 480)]
        p._motion["cam1"] = mock_motion

        ts = datetime(2026, 6, 15, 2, 0)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)

        events1 = p.on_frame("cam1", frame, ts)
        assert len(events1) == 1  # first sighting → event

        events2 = p.on_frame("cam1", frame, ts)
        assert len(events2) == 0  # same person → no new event
