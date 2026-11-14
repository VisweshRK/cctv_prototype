"""Polygon Zone Helper for Presence B2C CCTV AI Platform.
Provides polygon containment, loitering analysis, and zones.yaml management.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import yaml

from ai.cctv_ai.objects import TrackedObject


@dataclass
class ZoneOccupant:
    object_id: str
    first_entered: datetime
    last_seen: datetime
    loiter_alerted: bool = False

    @property
    def dwell_seconds(self) -> float:
        return (self.last_seen - self.first_entered).total_seconds()


@dataclass
class MonitoringZone:
    zone_id: str
    name: str
    camera_id: str
    polygon: List[Tuple[int, int]]  # List of (x, y) coordinates
    zone_type: str = "restricted"  # "intrusion", "restricted", "occupancy", "appliance", "cooking"
    loiter_threshold_seconds: int = 10
    color: str = "#EF4444"
    active_occupants: Dict[str, ZoneOccupant] = field(default_factory=dict)

    def contains_point(self, point: Tuple[int, int]) -> bool:
        """Test if (x, y) is inside the polygon."""
        if len(self.polygon) < 3:
            return False
        pts = np.array(self.polygon, dtype=np.int32)
        # cv2.pointPolygonTest returns positive if inside, zero if on edge, negative if outside
        dist = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False)
        return dist >= 0

    def contains_object(self, obj: TrackedObject, use_bottom: bool = True) -> bool:
        """Test if tracked object is inside zone.
        use_bottom=True checks object's ground contact point (more accurate for floor zones).
        """
        pt = obj.bottom_center if use_bottom else obj.center
        return self.contains_point(pt)

    def update_occupancy(
        self,
        current_objects: List[TrackedObject],
        now: Optional[datetime] = None
    ) -> Tuple[List[TrackedObject], List[ZoneOccupant], List[ZoneOccupant]]:
        """Update occupants.
        Returns:
            (entered_objects, loitering_occupants, exited_occupants)
        """
        if not now:
            now = datetime.now(timezone.utc)

        entered: List[TrackedObject] = []
        loitering: List[ZoneOccupant] = []
        exited: List[ZoneOccupant] = []

        current_inside_ids = set()

        for obj in current_objects:
            if self.contains_object(obj):
                current_inside_ids.add(obj.id)
                if obj.id not in self.active_occupants:
                    # Fresh entry
                    occ = ZoneOccupant(
                        object_id=obj.id,
                        first_entered=now,
                        last_seen=now,
                        loiter_alerted=False
                    )
                    self.active_occupants[obj.id] = occ
                    entered.append(obj)
                else:
                    # Update occupant
                    occ = self.active_occupants[obj.id]
                    occ.last_seen = now
                    if (not occ.loiter_alerted and 
                        occ.dwell_seconds >= self.loiter_threshold_seconds):
                        occ.loiter_alerted = True
                        loitering.append(occ)

        # Check who left
        missing_ids = [oid for oid in self.active_occupants if oid not in current_inside_ids]
        for mid in missing_ids:
            occ = self.active_occupants.pop(mid)
            exited.append(occ)

        return entered, loitering, exited

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "camera_id": self.camera_id,
            "polygon": self.polygon,
            "zone_type": self.zone_type,
            "loiter_threshold_seconds": self.loiter_threshold_seconds,
            "color": self.color,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MonitoringZone":
        raw_poly = data.get("polygon", [])
        polygon = [tuple(p) for p in raw_poly]
        return cls(
            zone_id=data["zone_id"],
            name=data["name"],
            camera_id=data.get("camera_id", "camera_main"),
            polygon=polygon,
            zone_type=data.get("zone_type", "restricted"),
            loiter_threshold_seconds=data.get("loiter_threshold_seconds", 10),
            color=data.get("color", "#EF4444"),
        )


class ZoneManager:
    """Manages all polygon zones across all cameras."""

    def __init__(self, zones_file: Optional[str] = None):
        self.zones: Dict[str, MonitoringZone] = {}
        self.zones_file = zones_file
        if zones_file:
            self.load_from_file(zones_file)

    def add_zone(self, zone: MonitoringZone) -> None:
        self.zones[zone.zone_id] = zone

    def remove_zone(self, zone_id: str) -> Optional[MonitoringZone]:
        return self.zones.pop(zone_id, None)

    def get_camera_zones(self, camera_id: str) -> List[MonitoringZone]:
        return [z for z in self.zones.values() if z.camera_id == camera_id]

    def load_from_file(self, filepath: str) -> None:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                for z_data in data.get("zones", []):
                    zone = MonitoringZone.from_dict(z_data)
                    self.zones[zone.zone_id] = zone
        except FileNotFoundError:
            pass

    def save_to_file(self, filepath: Optional[str] = None) -> None:
        target = filepath or self.zones_file
        if not target:
            return
        data = {
            "zones": [z.to_dict() for z in self.zones.values()]
        }
        with open(target, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
