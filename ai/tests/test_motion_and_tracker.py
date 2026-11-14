"""Tests for Motion Detection, Region Extraction, and Kalman Tracker.
Validates the architecture:
  frame reader -> motion -> regions -> shared detector (YOLOX)
  filters + per-label NMS -> tracker (Norfair, Kalman) -> TrackedObject
"""

from datetime import datetime, timezone, timedelta
import cv2
import numpy as np
import pytest

from ai.cctv_ai.motion import MotionDetector
from ai.cctv_ai.tracker import SharedTracker, KalmanBoxTracker, calculate_iou
from ai.cctv_ai.detector import SharedDetector


def test_motion_detector_static_vs_moving():
    detector = MotionDetector(scale=0.5, motion_threshold=20, min_motion_area=100)

    # Frame 1: empty room
    f1 = np.full((480, 640, 3), 50, dtype=np.uint8)
    res1 = detector.detect(f1)
    # First frame has no baseline, so has_motion is False
    assert res1.has_motion is False
    assert len(res1.motion_boxes) == 0

    # Frame 2: identical static frame -> NO motion
    f2 = np.full((480, 640, 3), 50, dtype=np.uint8)
    res2 = detector.detect(f2)
    assert res2.has_motion is False
    assert len(res2.regions) == 0

    # Frame 3: moving object enters (bright white rectangle at 200, 150)
    f3 = np.full((480, 640, 3), 50, dtype=np.uint8)
    cv2.rectangle(f3, (200, 150), (280, 320), (220, 220, 220), -1)
    res3 = detector.detect(f3)

    assert res3.has_motion is True
    assert len(res3.motion_boxes) >= 1
    assert len(res3.regions) >= 1

    # Candidate region should envelop the moving object
    rx1, ry1, rx2, ry2 = res3.regions[0]
    assert rx1 <= 200 and rx2 >= 280
    assert ry1 <= 150 and ry2 >= 320


def test_motion_regions_make_square():
    detector = MotionDetector(make_square=True, padding_ratio=0.2)
    f1 = np.full((480, 640, 3), 40, dtype=np.uint8)
    detector.detect(f1)

    # Moving tall vertical object
    f2 = np.full((480, 640, 3), 40, dtype=np.uint8)
    cv2.rectangle(f2, (300, 100), (340, 350), (240, 240, 240), -1)
    res = detector.detect(f2)

    assert res.has_motion is True
    assert len(res.regions) > 0
    rx1, ry1, rx2, ry2 = res.regions[0]
    rw = rx2 - rx1
    rh = ry2 - ry1
    # Square aspect ratio (or constrained by frame borders)
    assert abs(rw - rh) <= 2 or rx1 == 0 or rx2 == 640 or ry1 == 0 or ry2 == 480


def test_kalman_box_tracker_velocity_and_prediction():
    # Initial box at (100, 100, 150, 200) - moving rightwards by +10px each frame
    kalman = KalmanBoxTracker((100, 100, 150, 200))

    # Feed updates along horizontal trajectory
    for i in range(1, 6):
        x = 100 + i * 10
        kalman.predict()
        kalman.update((x, 100, x + 50, 200))

    # Measured velocity in x should be approximately +10
    vx, vy = kalman.velocity
    assert vx > 5.0  # Positive horizontal speed
    assert abs(vy) < 5.0  # Vertical speed close to zero

    # Predict next frame without measurement (occlusion prediction)
    predicted_box = kalman.predict()
    px1, py1, px2, py2 = predicted_box
    # Predicted x1 should advance past 150
    assert px1 >= 150


def test_shared_tracker_kalman_occlusion_smoothing():
    tracker = SharedTracker(camera_id="cam_test", min_hits_to_confirm=2, max_missed_frames=5, use_kalman=True)
    t0 = datetime(2026, 10, 30, 12, 0, 0, tzinfo=timezone.utc)

    # Frame 1: Detection 1
    d1 = [{"box": (100, 100, 160, 250), "label": "person", "score": 0.90}]
    active, ended = tracker.update(d1, t0)
    assert len(active) == 0  # Requires 2 hits to confirm

    # Frame 2: Detection 2 (moving right by 10px)
    t1 = t0 + timedelta(milliseconds=200)
    d2 = [{"box": (110, 100, 170, 250), "label": "person", "score": 0.92}]
    active, ended = tracker.update(d2, t1)
    assert len(active) == 1
    track_id = active[0].id
    assert active[0].true_positive is True

    # Frame 3: Person momentarily occluded (0 detections)
    # Tracker should keep the track alive using Kalman prediction
    t2 = t1 + timedelta(milliseconds=200)
    active3, ended3 = tracker.update([], t2)
    # Track is still active in internal state
    assert track_id in tracker._tracks
    # Track position predicted forward
    pred_box = tracker._tracks[track_id]["box"]
    assert pred_box[0] >= 110


def test_shared_detector_region_detection():
    detector = SharedDetector()
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    regions = [(50, 50, 350, 350)]

    # Detect in region should run without error and return list
    dets = detector.detect(frame, regions=regions)
    assert isinstance(dets, list)
