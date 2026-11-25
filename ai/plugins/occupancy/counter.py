"""counter.py: Per-camera and per-zone occupancy debounce tracker, capacity monitor,
and tagged occupant registry.
Per ZoneFallPlan.md Section 4 and docs/plans/p3-p4-sanjay-kishore.md Section 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class OccupantTag:
    """Detailed metadata for an individual person in the camera view."""

    object_id: str
    display_tag: str  # e.g. "Person #1"
    box: tuple[int, int, int, int]
    score: float
    stationary: bool
    status: str  # "stationary" or "active"
    current_zone: str  # zone name or "whole_view"
    first_seen: datetime
    seconds_present: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.object_id,
            "tag": self.display_tag,
            "box": list(self.box),
            "score": round(self.score, 2),
            "status": self.status,
            "zone": self.current_zone,
            "seconds_present": round(self.seconds_present, 1),
        }


@dataclass
class DebouncedZoneCount:
    """State of debounced occupancy for a single zone or camera scope."""

    stable_count: int = 0
    candidate_count: Optional[int] = None
    candidate_since: Optional[datetime] = None
    is_over_capacity: bool = False
    last_summary_ts: Optional[datetime] = None
    initialized: bool = False


class OccupancyDebouncer:
    """Debounces occupancy count fluctuations and monitors capacity thresholds.

    Rules:
    - 3-second stability debounce before accepting a count change.
    - over_capacity event emitted once on rising edge, re-armed on falling edge.
    - Periodic summary emitted at configured interval (default 300 s / 5 min).
    """

    def __init__(
        self,
        debounce_seconds: float = 3.0,
        capacity_limit: Optional[int] = None,
        summary_interval: float = 300.0,
    ):
        self.debounce_seconds = debounce_seconds
        self.capacity_limit = capacity_limit
        self.summary_interval = summary_interval
        # Key: (camera, zone_name) -> DebouncedZoneCount
        self.scopes: dict[tuple[str, str], DebouncedZoneCount] = {}

    def get_scope(self, camera: str, zone: str) -> DebouncedZoneCount:
        key = (camera, zone)
        if key not in self.scopes:
            self.scopes[key] = DebouncedZoneCount()
        return self.scopes[key]

    def get_stable_count(self, camera: str, zone: str) -> int:
        return self.get_scope(camera, zone).stable_count

    def update_raw_count(
        self,
        camera: str,
        zone: str,
        raw_count: int,
        ts: datetime,
        occupants: Optional[list[dict]] = None,
        all_zone_counts: Optional[dict[str, int]] = None,
        emit_events: bool = True,
    ) -> list[tuple[str, dict]]:
        """Feed a raw instantaneous count observed at timestamp `ts`.
        Returns a list of event tuples: [(event_label, event_data), ...]
        """
        scope = self.get_scope(camera, zone)
        events = []

        # 1. Debounce logic
        count_changed = False
        old_stable = scope.stable_count

        if scope.initialized and raw_count == scope.stable_count:
            scope.candidate_count = None
            scope.candidate_since = None
        else:
            if scope.candidate_count != raw_count:
                scope.candidate_count = raw_count
                scope.candidate_since = ts
            else:
                if scope.candidate_since is not None:
                    elapsed = (ts - scope.candidate_since).total_seconds()
                    if elapsed >= self.debounce_seconds:
                        count_changed = True
                        scope.initialized = True
                        scope.stable_count = raw_count
                        scope.candidate_count = None
                        scope.candidate_since = None

        if count_changed:
            if emit_events:
                unit = "person" if scope.stable_count == 1 else "people"
                loc = f"in {zone}" if zone else "in view"
                area_str = f"Zone: {zone}" if zone else "Whole view"
                evt_payload = {
                    "zone": zone,
                    "count": scope.stable_count,
                    "previous_count": old_stable,
                    "change": "count_changed",
                    "area": area_str,
                    "detail": f"{scope.stable_count} {unit} {loc}",
                }
                if occupants is not None:
                    evt_payload["occupants"] = occupants
                if all_zone_counts is not None:
                    evt_payload["zone_counts"] = all_zone_counts

                events.append(("occupancy", evt_payload))

                # Check capacity threshold
                if self.capacity_limit is not None:
                    if scope.stable_count > self.capacity_limit and not scope.is_over_capacity:
                        scope.is_over_capacity = True
                        events.append(
                            (
                                "over_capacity",
                                {
                                    "zone": zone,
                                    "count": scope.stable_count,
                                    "capacity_limit": self.capacity_limit,
                                    "area": area_str,
                                    "detail": f"Over capacity: {scope.stable_count} > {self.capacity_limit}",
                                },
                            )
                        )
                    elif scope.stable_count <= self.capacity_limit and scope.is_over_capacity:
                        scope.is_over_capacity = False

        # 2. Periodic summary heartbeat
        if scope.last_summary_ts is None:
            scope.last_summary_ts = ts
        else:
            elapsed_summary = (ts - scope.last_summary_ts).total_seconds()
            if elapsed_summary >= self.summary_interval:
                scope.last_summary_ts = ts
                if emit_events:
                    unit = "person" if scope.stable_count == 1 else "people"
                    loc = f"in {zone}" if zone else "in view"
                    events.append(
                        (
                            "summary",
                            {
                                "zone": zone,
                                "count": scope.stable_count,
                                "summary_interval": self.summary_interval,
                                "area": f"Zone: {zone}" if zone else "Whole view",
                                "detail": f"Occupancy heartbeat: {scope.stable_count} {unit} {loc}",
                            },
                        )
                    )

        return events
