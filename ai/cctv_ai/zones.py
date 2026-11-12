"""zones.py: Shared Zone Helper for CCTV AI.
Handles zone polygon loading, coordinate normalization, feet calculation,
and the ZoneTracker inertia state machine (inertia=3 default, loitering, count).

Per docs/plans/p3-p4-sanjay-kishore.md and Downloads/ZoneFallPlan.md.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator, Optional

import cv2
import numpy as np
import yaml

from .objects import TrackedObject

log = logging.getLogger("cctv_ai.zones")


def feet(box: tuple[int, int, int, int]) -> tuple[float, float]:
    """Calculate the bottom-centre (ground contact) point for a bounding box (x1, y1, x2, y2).
    Feet, not centre: a person in front of a door is not inside the room behind it.
    """
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, float(y2))


def point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    """Return True if point is inside or on the boundary of polygon."""
    return cv2.pointPolygonTest(polygon, point, measureDist=False) >= 0


@dataclass
class ZoneConfig:
    """Configuration for a single zone."""

    name: str
    points: list[list[float]]  # Fractions of width and height: [[x, y], ...]
    objects: list[str] = field(default_factory=lambda: ["person"])
    inertia: int = 3  # Consecutive frames to enter / leave
    loitering_time: float = 0.0  # Seconds before loitering event; 0 = disabled
    exclude: bool = False  # If True, used to exclude regions (e.g. bed/sofa)


@dataclass
class ZoneTransition:
    """Represents a state change of an object with respect to a zone."""

    camera: str
    object: TrackedObject
    zone: str
    change: str  # "entered", "left", "loitering"
    seconds_inside: float = 0.0

    def __iter__(self) -> Iterator[Any]:
        """Support unpacking as (object, zone, change) for backward compatibility."""
        yield self.object
        yield self.zone
        yield self.change


@dataclass
class _ObjectZoneState:
    is_inside: bool = False
    inside_count: int = 0
    outside_count: int = 0
    inside_since: Optional[datetime] = None
    loitered_emitted: bool = False
    last_seen: Optional[datetime] = None


def load_zones_config(yaml_path: str) -> dict[str, dict[str, ZoneConfig]]:
    """Load zones from a YAML file.
    Returns: dict mapping camera_id -> {zone_name: ZoneConfig}
    """
    if not os.path.exists(yaml_path):
        log.warning("Zones file %s not found, returning empty configuration", yaml_path)
        return {}

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as err:
        log.error("Failed to parse zones YAML from %s: %s", yaml_path, err)
        return {}

    result: dict[str, dict[str, ZoneConfig]] = {}
    for cam_id, zones_dict in raw.items():
        if not isinstance(zones_dict, dict):
            continue
        cam_zones = {}
        for z_name, z_data in zones_dict.items():
            if not isinstance(z_data, dict):
                continue
            points = z_data.get("points", [])
            objects = z_data.get("objects", ["person"])
            inertia = int(z_data.get("inertia", 3))
            loitering_time = float(z_data.get("loitering_time", 0.0))
            exclude = bool(z_data.get("exclude", False))
            cam_zones[z_name] = ZoneConfig(
                name=z_name,
                points=points,
                objects=objects,
                inertia=inertia,
                loitering_time=loitering_time,
                exclude=exclude,
            )
        result[cam_id] = cam_zones
    return result


class ZoneTracker:
    """Maintains per-camera zone membership and inertia state for tracked objects."""

    def __init__(self, camera: str, zones: dict[str, ZoneConfig]):
        self.camera = camera
        self.zones = zones
        self._resolution: Optional[tuple[int, int]] = None  # (width, height)
        self._polygons: dict[str, np.ndarray] = {}  # Pixel coordinates
        self._states: dict[tuple[str, str], _ObjectZoneState] = {}  # (obj_id, zone_name) -> state

    def set_resolution(self, width: int, height: int) -> None:
        """Update or recalculate pixel polygons when camera frame resolution is known or changes."""
        if self._resolution == (width, height) and self._polygons:
            return
        self._resolution = (width, height)
        self._polygons = {}
        for name, config in self.zones.items():
            pts = []
            for pt in config.points:
                px = float(pt[0] * width)
                py = float(pt[1] * height)
                pts.append([px, py])
            if pts:
                self._polygons[name] = np.array(pts, dtype=np.float32)

    @property
    def polygons(self) -> dict[str, np.ndarray]:
        """Returns map of zone_name -> pixel polygon array."""
        return self._polygons

    def update(
        self,
        objects: list[TrackedObject],
        ts: datetime,
        frame_shape: Optional[tuple[int, int]] = None,
        ended: Optional[list[TrackedObject]] = None,
    ) -> list[ZoneTransition]:
        """Update zone tracking with the current list of tracked objects.

        Args:
            objects: List of TrackedObject currently active in this frame.
            ts: Frame timestamp.
            frame_shape: Optional (height, width) or (width, height) to update resolution.
            ended: Optional list of TrackedObject lost by the tracker.

        Returns:
            List of ZoneTransition events ('entered', 'left', 'loitering').
        """
        if frame_shape is not None:
            if len(frame_shape) >= 2:
                # If passed shape is (h, w), width is index 1, height is index 0
                h, w = frame_shape[0], frame_shape[1]
                self.set_resolution(w, h)

        transitions: list[ZoneTransition] = []
        active_ids = {obj.id for obj in objects}

        # 1. Update each active object against each zone
        for obj in objects:
            ft = feet(obj.box)
            for z_name, z_config in self.zones.items():
                # Check allowed object labels
                if z_config.objects and obj.label not in z_config.objects:
                    continue

                poly = self._polygons.get(z_name)
                if poly is None:
                    continue

                state_key = (obj.id, z_name)
                if state_key not in self._states:
                    self._states[state_key] = _ObjectZoneState()
                state = self._states[state_key]
                state.last_seen = ts

                raw_inside = point_in_polygon(ft, poly)

                if raw_inside:
                    state.inside_count += 1
                    state.outside_count = 0  # reset on contradiction

                    if not state.is_inside:
                        if state.inside_count >= z_config.inertia:
                            state.is_inside = True
                            state.inside_since = ts
                            state.loitered_emitted = False
                            transitions.append(
                                ZoneTransition(
                                    camera=self.camera,
                                    object=obj,
                                    zone=z_name,
                                    change="entered",
                                    seconds_inside=0.0,
                                )
                            )
                    else:
                        # Already inside: check loitering
                        if z_config.loitering_time > 0 and not state.loitered_emitted:
                            if state.inside_since:
                                duration = (ts - state.inside_since).total_seconds()
                                if duration >= z_config.loitering_time:
                                    state.loitered_emitted = True
                                    transitions.append(
                                        ZoneTransition(
                                            camera=self.camera,
                                            object=obj,
                                            zone=z_name,
                                            change="loitering",
                                            seconds_inside=duration,
                                        )
                                    )
                else:
                    # Raw outside
                    state.outside_count += 1
                    state.inside_count = 0  # reset on contradiction

                    if state.is_inside:
                        if state.outside_count >= z_config.inertia:
                            state.is_inside = False
                            duration = (
                                (ts - state.inside_since).total_seconds()
                                if state.inside_since
                                else 0.0
                            )
                            state.inside_since = None
                            state.loitered_emitted = False
                            transitions.append(
                                ZoneTransition(
                                    camera=self.camera,
                                    object=obj,
                                    zone=z_name,
                                    change="left",
                                    seconds_inside=duration,
                                )
                            )

        # 2. Handle missing objects (objects that were tracked previously but not in this frame)
        # We increment their outside_count so they eventually leave after inertia misses
        for (obj_id, z_name), state in list(self._states.items()):
            if obj_id not in active_ids and state.is_inside:
                state.outside_count += 1
                state.inside_count = 0
                if state.outside_count >= self.zones[z_name].inertia:
                    state.is_inside = False
                    duration = (
                        (ts - state.inside_since).total_seconds()
                        if state.inside_since
                        else 0.0
                    )
                    state.inside_since = None
                    state.loitered_emitted = False
                    # Create placeholder object representation for the left transition
                    dummy_obj = TrackedObject(
                        id=obj_id,
                        label="unknown",
                        box=(0, 0, 0, 0),
                        score=0.0,
                        median_score=0.0,
                        true_positive=True,
                    )
                    transitions.append(
                        ZoneTransition(
                            camera=self.camera,
                            object=dummy_obj,
                            zone=z_name,
                            change="left",
                            seconds_inside=duration,
                        )
                    )

        # 3. Clean up ended objects explicitly reported by the tracker
        if ended:
            ended_ids = {obj.id for obj in ended}
            for (obj_id, z_name) in list(self._states.keys()):
                if obj_id in ended_ids:
                    del self._states[(obj_id, z_name)]

        return transitions

    def count(self, zone_name: str) -> int:
        """Return the immediate confirmed object count in the specified zone."""
        return sum(
            1
            for (obj_id, z_name), state in self._states.items()
            if z_name == zone_name and state.is_inside
        )

    def is_in_zone(self, obj_id: str, zone_name: str) -> bool:
        """Check if an object currently has confirmed membership in a zone."""
        state = self._states.get((obj_id, zone_name))
        return bool(state and state.is_inside)

    def is_in_any_exclude_zone(self, box: tuple[int, int, int, int]) -> bool:
        """Check if a box's feet fall within any zone marked as exclude=True (e.g. bed/sofa)."""
        ft = feet(box)
        for z_name, z_config in self.zones.items():
            if z_config.exclude:
                poly = self._polygons.get(z_name)
                if poly is not None and point_in_polygon(ft, poly):
                    return True
        return False
