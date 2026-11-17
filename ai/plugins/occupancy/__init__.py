"""Occupancy counting and individual person tagging plugin.
Slot: Occupancy counting (ai/plugins/occupancy)

Features:
- Tags and identifies each individual person in the room (e.g. "Person #1 [Active]", "Person #2 [Stationary]").
- Distinguishes active moving persons from stationary seated individuals.
- Tracks exact live headcount per zone and across the entire room.
- Debounces count changes with a 3-second stability window (anti-flicker).
- Emits events matching frontend contract: 'occupancy', 'over_capacity', and 'summary'.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import numpy as np

from cctv_ai.objects import TrackedObject
from cctv_ai.plugin import Event, Plugin as Base
from cctv_ai.zones import ZoneTracker, feet, load_zones_config, point_in_polygon
from .counter import OccupancyDebouncer, OccupantTag


class Plugin(Base):
    use_case = "occupancy"
    enabled = True
    needs = {"objects", "video"}
    interval = 0.2  # 5 FPS nominal rate

    def __init__(self):
        super().__init__()
        self.trackers: dict[str, ZoneTracker] = {}
        self.debouncer = OccupancyDebouncer(debounce_seconds=3.0, capacity_limit=None, summary_interval=300.0)
        self.zones_file = "zones.yaml"
        # Map object_id -> stable tag number (1, 2, 3...)
        self.id_to_num: dict[str, int] = {}
        self.next_num = 1
        # Active occupants cache per camera
        self.active_occupants: dict[str, list[OccupantTag]] = {}

    def setup(self, cameras: list[str]) -> None:
        """Called once before first frame. Loads zones configuration."""
        possible_paths = [
            self.zones_file,
            os.path.join(os.path.dirname(__file__), "..", "..", "zones.yaml"),
            os.path.join(os.path.dirname(__file__), "..", "..", "ai", "zones.yaml"),
        ]
        resolved_path = self.zones_file
        for p in possible_paths:
            if os.path.exists(p):
                resolved_path = os.path.abspath(p)
                break

        zones_by_camera = load_zones_config(resolved_path)

        for cam in cameras:
            cam_zones = zones_by_camera.get(cam, {})
            self.trackers[cam] = ZoneTracker(camera=cam, zones=cam_zones)
            self.active_occupants[cam] = []

    def on_frame(self, camera: str, frame: np.ndarray, ts: datetime) -> list[Event]:
        """Backward compatibility for legacy runner calling on_frame."""
        return []

    def on_objects(
        self,
        camera: str,
        frame: np.ndarray,
        ts: datetime,
        objects: list[TrackedObject],
        ended: list[TrackedObject],
    ) -> list[Event]:
        """Evaluates confirmed people, tags each individual person, updates zone occupancy,
        and generates debounced events.
        """
        if camera not in self.trackers:
            self.trackers[camera] = ZoneTracker(camera=camera, zones={})
        if camera not in self.active_occupants:
            self.active_occupants[camera] = []

        tracker = self.trackers[camera]
        if frame is not None and len(frame.shape) >= 2:
            tracker.set_resolution(width=frame.shape[1], height=frame.shape[0])

        # Purge ended tracks from tag mapping
        if ended:
            for obj in ended:
                self.id_to_num.pop(obj.id, None)

        # Filter strictly for confirmed true-positive persons (excluding unconfirmed furniture/appliances)
        person_objects = [
            obj for obj in objects if obj.label == "person" and obj.true_positive
        ]

        # Update zone tracker with active persons
        tracker.update(person_objects, ts, ended=ended)

        # Build tagged occupant list for this frame
        current_tags: list[OccupantTag] = []
        zone_counts: dict[str, int] = {z_name: 0 for z_name in tracker.zones.keys()}

        # Maintain active occupant numbers (1, 2, ...) so a single person in the room is always Person #1
        active_ids = {obj.id for obj in person_objects}
        self.id_to_num = {tid: num for tid, num in self.id_to_num.items() if tid in active_ids}
        used_nums = set(self.id_to_num.values())

        for obj in person_objects:
            if obj.id not in self.id_to_num:
                num = 1
                while num in used_nums:
                    num += 1
                self.id_to_num[obj.id] = num
                used_nums.add(num)

            person_num = self.id_to_num[obj.id]
            status_str = "stationary" if obj.stationary else "active"
            tag_label = f"Person #{person_num} [{status_str.capitalize()}]"

            # Determine primary zone
            assigned_zone = "whole_room"
            for z_name, z_poly in tracker.polygons.items():
                if z_poly is not None and point_in_polygon(feet(obj.box), z_poly):
                    assigned_zone = z_name
                    zone_counts[z_name] = zone_counts.get(z_name, 0) + 1
                    break

            elapsed_presence = (ts - obj.first_seen).total_seconds() if obj.first_seen else 0.0

            current_tags.append(
                OccupantTag(
                    object_id=obj.id,
                    display_tag=tag_label,
                    box=obj.box,
                    score=obj.score,
                    stationary=obj.stationary,
                    status=status_str,
                    current_zone=assigned_zone,
                    first_seen=obj.first_seen,
                    seconds_present=elapsed_presence,
                )
            )

        self.active_occupants[camera] = current_tags

        # Determine which zones to emit events for to prevent UI double-counting
        configured_zones = list(tracker.zones.keys())

        emit_counts: dict[str, int] = {}
        if not configured_zones:
            # No zones configured: emit whole view under ""
            emit_counts[""] = len(person_objects)
        elif len(configured_zones) == 1 and configured_zones[0] == "room":
            # Default room zone: emit under "room"
            emit_counts["room"] = len(person_objects)
        else:
            # Subzones defined: emit each subzone
            for z_name in configured_zones:
                emit_counts[z_name] = zone_counts.get(z_name, 0)
            # Count any persons not inside any defined subzone
            inside_subzones = sum(zone_counts.get(z_name, 0) for z_name in configured_zones)
            unzoned = len(person_objects) - inside_subzones
            if unzoned > 0 or self.debouncer.get_scope(camera, "").stable_count > 0:
                emit_counts[""] = max(0, unzoned)

        occupants_data = [tag.to_dict() for tag in current_tags]
        events: list[Event] = []

        # 1. Update debouncer for total / "all" internally without emitting duplicate events to UI
        self.debouncer.update_raw_count(
            camera=camera,
            zone="all",
            raw_count=len(person_objects),
            ts=ts,
            occupants=occupants_data,
            all_zone_counts=emit_counts,
            emit_events=False,
        )

        # Also update "room" scope if it wasn't in emit_counts so get_count(cam, "room") works
        if "room" not in emit_counts:
            self.debouncer.update_raw_count(
                camera=camera,
                zone="room",
                raw_count=len(person_objects),
                ts=ts,
                occupants=occupants_data,
                all_zone_counts=emit_counts,
                emit_events=False,
            )

        # 2. Update debouncer and emit events for the monitored zones
        for zone_name, raw_cnt in emit_counts.items():
            triggered = self.debouncer.update_raw_count(
                camera=camera,
                zone=zone_name,
                raw_count=raw_cnt,
                ts=ts,
                occupants=occupants_data,
                all_zone_counts=emit_counts,
                emit_events=True,
            )
            for evt_label, evt_data in triggered:
                events.append(
                    self.event(
                        camera=camera,
                        label=evt_label,
                        confidence=1.0,
                        data=evt_data,
                    )
                )

        return events

    def get_tagged_occupants(self, camera: str) -> list[OccupantTag]:
        """Returns the list of currently tagged occupants for the camera."""
        return self.active_occupants.get(camera, [])

    def get_count(self, camera: str, zone: str = "room") -> int:
        """Direct query for power_when_absent integration.
        Returns accepted stable count for the zone or 'all'.
        """
        if zone in ("all", "total"):
            return self.debouncer.get_stable_count(camera, "all")
        if zone == "room" and (camera, "room") not in self.debouncer.scopes:
            if (camera, "") in self.debouncer.scopes:
                return self.debouncer.get_stable_count(camera, "")
            return self.debouncer.get_stable_count(camera, "all")
        return self.debouncer.get_stable_count(camera, zone)
