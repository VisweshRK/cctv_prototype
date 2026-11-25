"""testing.py: Fake pipeline fixtures and helpers for tests without requiring full neural nets.
Per docs/plans/viswesh.md Step 6.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from .objects import TrackedObject


def make_object(
    obj_id: str = "cam1-1",
    label: str = "person",
    box: tuple[int, int, int, int] = (100, 100, 200, 300),
    score: float = 0.85,
    median_score: float = 0.85,
    true_positive: bool = True,
    stationary: bool = False,
    motionless_count: int = 0,
    history: Optional[list[tuple[int, int, int, int]]] = None,
    first_seen: Optional[datetime] = None,
    last_seen: Optional[datetime] = None,
) -> TrackedObject:
    now = datetime.now(timezone.utc)
    return TrackedObject(
        id=obj_id,
        label=label,
        box=box,
        score=score,
        median_score=median_score,
        true_positive=true_positive,
        stationary=stationary,
        motionless_count=motionless_count,
        first_seen=first_seen or now,
        last_seen=last_seen or now,
        history=history if history is not None else [box],
    )
