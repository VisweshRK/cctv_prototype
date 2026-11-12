"""TrackedObject dataclass representing the shared contract between tracker and plugins.
Defined per docs/plans/README.md (Viswesh).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class TrackedObject:
    """An object tracked across frames with Kalman prediction and identity retention."""

    id: str  # "cam1-17": same id while the tracker follows this object
    label: str  # "person", "car", "motorcycle", "bus", "truck", etc.
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels of the frame
    score: float  # this frame's detection score (0.0 if predicted, not detected)
    median_score: float  # median of last 10 scores (missed frames count as 0.0)
    true_positive: bool  # median_score has reached >= 0.7 at least once
    stationary: bool = False  # has not moved for a while (still tracked, not re-detected)
    motionless_count: int = 0  # frames in a row without moving
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    history: list[tuple[int, int, int, int]] = field(default_factory=list)  # last ~25 boxes, oldest first
