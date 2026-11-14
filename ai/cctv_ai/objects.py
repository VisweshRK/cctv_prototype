"""Shared Tracked Object contracts for Presence B2C CCTV AI Platform.
Matches Section 4 contract specifications.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Tuple


@dataclass
class TrackedObject:
    """Shared tracker representation across all AI modules."""
    id: str
    label: str  # "person", "car", "motorcycle", "bus", "truck", "pet"
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    score: float
    median_score: float
    true_positive: bool
    stationary: bool
    motionless_count: int
    first_seen: datetime
    last_seen: datetime
    history: List[Tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    @property
    def bottom_center(self) -> Tuple[int, int]:
        """Ground contact point for zone footprint testing."""
        x1, y1, x2, y2 = self.box
        return (int((x1 + x2) / 2), int(y2))

    @property
    def width(self) -> int:
        return max(0, self.box[2] - self.box[0])

    @property
    def height(self) -> int:
        return max(0, self.box[3] - self.box[1])

    @property
    def aspect_ratio(self) -> float:
        """Width / Height. Sudden increase can indicate a fall."""
        h = self.height
        return (self.width / h) if h > 0 else 1.0

    @property
    def velocity(self) -> float:
        """Pixels per frame over recent history."""
        if len(self.history) < 2:
            return 0.0
        p1 = self.history[-1]
        p0 = self.history[-2]
        c1 = ((p1[0] + p1[2]) / 2, (p1[1] + p1[3]) / 2)
        c0 = ((p0[0] + p0[2]) / 2, (p0[1] + p0[3]) / 2)
        return ((c1[0] - c0[0]) ** 2 + (c1[1] - c0[1]) ** 2) ** 0.5
