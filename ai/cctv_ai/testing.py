"""Testing utilities and mock generators for Presence B2C CCTV AI Platform.
Supports Section 6 and Plan P1 testing without needing physical camera feeds.
"""

from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple
import cv2
import numpy as np

from ai.cctv_ai.objects import TrackedObject


def create_test_person(
    track_id: str = "person_1",
    box: Tuple[int, int, int, int] = (150, 100, 250, 350),
    score: float = 0.92,
    median_score: float = 0.92,
    true_positive: bool = True,
    stationary: bool = False,
    motionless_count: int = 0,
    timestamp: Optional[datetime] = None,
    history: Optional[List[Tuple[int, int, int, int]]] = None,
) -> TrackedObject:
    """Creates a TrackedObject representing a detected person."""
    ts = timestamp or datetime.now(timezone.utc)
    return TrackedObject(
        id=track_id,
        label="person",
        box=box,
        score=score,
        median_score=median_score,
        true_positive=true_positive,
        stationary=stationary,
        motionless_count=motionless_count,
        first_seen=ts,
        last_seen=ts,
        history=history if history is not None else [box],
    )


def create_stationary_person(
    track_id: str = "person_couch",
    box: Tuple[int, int, int, int] = (180, 150, 260, 320),
    timestamp: Optional[datetime] = None,
    motionless_count: int = 45,
) -> TrackedObject:
    """Creates a stationary person sitting still (e.g. reading or sleeping on couch)."""
    return create_test_person(
        track_id=track_id,
        box=box,
        score=0.94,
        median_score=0.94,
        true_positive=True,
        stationary=True,
        motionless_count=motionless_count,
        timestamp=timestamp,
    )


def generate_test_frame(
    width: int = 640,
    height: int = 480,
    light_on: bool = False,
    light_roi: Optional[Tuple[int, int, int, int]] = (50, 50, 150, 150),
    screen_on: bool = False,
    screen_roi: Optional[Tuple[int, int, int, int]] = (400, 80, 550, 200),
    base_brightness: int = 40,
) -> np.ndarray:
    """Generates a synthetic camera room frame with optional light and screen illumination."""
    frame = np.full((height, width, 3), base_brightness, dtype=np.uint8)

    if light_roi:
        lx1, ly1, lx2, ly2 = light_roi
        lum = 220 if light_on else base_brightness
        # Color with subtle warm incandescent tint
        frame[ly1:ly2, lx1:lx2] = (lum - 10, lum, lum + 15)

    if screen_roi:
        sx1, sy1, sx2, sy2 = screen_roi
        if screen_on:
            # Active screen display with color variation
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (210, 180, 100), -1)
        else:
            # Dark screen
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (25, 25, 25), -1)

    return frame


def generate_ir_test_frame(
    width: int = 640,
    height: int = 480,
    light_roi: Optional[Tuple[int, int, int, int]] = (50, 50, 150, 150),
    ir_reflection_brightness: int = 180,
) -> np.ndarray:
    """Generates an IR night-vision frame (grayscale / saturation ~ 0).
    Even if an object or light fixture reflects IR light and looks bright,
    saturation ≈ 0 denotes that the camera switched to IR and room lights are OFF.
    """
    # Create pure gray frame (saturation = 0)
    frame = np.full((height, width, 3), 50, dtype=np.uint8)

    if light_roi:
        lx1, ly1, lx2, ly2 = light_roi
        # IR illuminator reflection (bright, but completely monochrome / grey)
        frame[ly1:ly2, lx1:lx2] = (ir_reflection_brightness, ir_reflection_brightness, ir_reflection_brightness)

    return frame


def generate_fan_frame_pair(
    width: int = 640,
    height: int = 480,
    fan_roi: Tuple[int, int, int, int] = (250, 20, 350, 100),
    fan_moving: bool = True,
    angle_step: float = 45.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a pair of consecutive frames to test fan blade motion via frame differencing."""
    f1 = np.full((height, width, 3), 60, dtype=np.uint8)
    f2 = np.full((height, width, 3), 60, dtype=np.uint8)

    fx1, fy1, fx2, fy2 = fan_roi
    cx, cy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
    r = min(fx2 - fx1, fy2 - fy1) // 3

    # Frame 1: fan blade horizontal
    cv2.line(f1, (cx - r, cy), (cx + r, cy), (200, 200, 200), 4)

    if fan_moving:
        # Frame 2: fan blade rotated (e.g. diagonal)
        cv2.line(f2, (cx - r // 2, cy - r // 2), (cx + r // 2, cy + r // 2), (200, 200, 200), 4)
    else:
        # Fan is stationary
        cv2.line(f2, (cx - r, cy), (cx + r, cy), (200, 200, 200), 4)

    return f1, f2
