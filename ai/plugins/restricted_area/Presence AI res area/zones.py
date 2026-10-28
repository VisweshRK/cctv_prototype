"""Zone management and geometric point-in-polygon tests.

Follows Frigate's pattern:
- Zones are defined as normalized polygons (0.0 to 1.0 relative coordinates).
- Polygons are scaled to frame dimensions (width, height).
- Person presence is evaluated using the bottom-centre of their bounding box,
  representing where the person is standing on the ground plane.
- Point-in-polygon tests use OpenCV's cv2.pointPolygonTest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def get_bottom_centre(
    box: Sequence[float | int] | tuple[int, int, int, int],
) -> tuple[float, float]:
    """Calculate the bottom-centre point (cx, bottom_y) of a bounding box.

    Parameters
    ----------
    box : Sequence of (x1, y1, x2, y2)
        Bounding box coordinates.

    Returns
    -------
    tuple[float, float]
        (cx, bottom_y) where cx is horizontal midpoint and bottom_y is the bottom edge.
    """
    x1, y1, x2, y2 = box[:4]
    cx = (float(x1) + float(x2)) / 2.0
    bottom_y = float(y2)
    return (cx, bottom_y)


def scale_polygon(
    polygon: Sequence[Sequence[float]],
    width: int,
    height: int,
    dtype: type = np.int32,
) -> np.ndarray:
    """Scale normalized polygon vertices [0.0, 1.0] to frame pixel dimensions.

    Parameters
    ----------
    polygon : Sequence of (x, y) pairs in range [0.0, 1.0]
        Normalized polygon vertices.
    width : int
        Frame width in pixels.
    height : int
        Frame height in pixels.
    dtype : numpy dtype, default np.int32
        Target data type for the scaled coordinates.

    Returns
    -------
    np.ndarray
        Array of scaled vertex points with shape (N, 1, 2) or (N, 2).
    """
    if not polygon:
        return np.empty((0, 2), dtype=dtype)

    pts = []
    for pt in polygon:
        x, y = float(pt[0]), float(pt[1])
        # Clamp to [0.0, 1.0] range
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        scaled_x = x * width
        scaled_y = y * height
        pts.append([scaled_x, scaled_y])

    arr = np.array(pts, dtype=dtype)
    return arr


def point_in_polygon(
    point: tuple[float, float],
    polygon: np.ndarray,
    include_edge: bool = True,
) -> bool:
    """Check if a point is inside a polygon using OpenCV cv2.pointPolygonTest.

    Parameters
    ----------
    point : tuple[float, float]
        (x, y) coordinate to test.
    polygon : np.ndarray
        Polygon vertex array (shape (N, 2) or (N, 1, 2), int32 or float32).
    include_edge : bool, default True
        If True, points directly on the boundary/vertices are considered inside.

    Returns
    -------
    bool
        True if the point is inside the polygon, False otherwise.
    """
    if polygon is None or len(polygon) < 3:
        return False

    score = cv2.pointPolygonTest(
        polygon,
        (float(point[0]), float(point[1])),
        measureDist=False,
    )
    return score >= 0 if include_edge else score > 0


@dataclass
class Zone:
    """A named restricted polygon zone defined in normalized coordinates."""

    name: str
    polygon: list[tuple[float, float]]
    _scaled_cache: dict[tuple[int, int], np.ndarray] = field(
        default_factory=dict, repr=False, compare=False
    )

    def get_scaled_polygon(self, width: int, height: int) -> np.ndarray:
        """Get (or compute and cache) scaled polygon array for frame dimensions."""
        dim_key = (width, height)
        if dim_key not in self._scaled_cache:
            self._scaled_cache[dim_key] = scale_polygon(self.polygon, width, height)
        return self._scaled_cache[dim_key]

    def contains_point(
        self,
        point: tuple[float, float],
        width: int,
        height: int,
        include_edge: bool = True,
    ) -> bool:
        """Check if an absolute (x, y) point is inside this zone."""
        poly = self.get_scaled_polygon(width, height)
        return point_in_polygon(point, poly, include_edge=include_edge)

    def contains_box_bottom_centre(
        self,
        box: Sequence[float | int] | tuple[int, int, int, int],
        width: int,
        height: int,
        include_edge: bool = True,
    ) -> bool:
        """Check if the bottom-centre of a bounding box is inside this zone."""
        bc = get_bottom_centre(box)
        return self.contains_point(bc, width, height, include_edge=include_edge)


class ZoneManager:
    """Manages restricted zones across multiple cameras."""

    def __init__(
        self,
        zones_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialise ZoneManager.

        Parameters
        ----------
        zones_config : dict, optional
            Either:
            1. {camera_name: {zone_name: [(x1, y1), (x2, y2), ...]}}
            2. {camera_name: [{"name": zone_name, "polygon": [(x1, y1), ...]}]}
        """
        self._zones: dict[str, dict[str, Zone]] = {}
        if zones_config:
            self.load_config(zones_config)

    def load_config(self, zones_config: dict[str, Any]) -> None:
        """Load or update zone configurations."""
        for cam_name, cam_zones in zones_config.items():
            self._zones.setdefault(cam_name, {})
            if isinstance(cam_zones, dict):
                for zone_name, poly in cam_zones.items():
                    if isinstance(poly, dict) and "polygon" in poly:
                        self.add_zone(cam_name, zone_name, poly["polygon"])
                    else:
                        self.add_zone(cam_name, zone_name, poly)
            elif isinstance(cam_zones, list):
                for item in cam_zones:
                    if isinstance(item, dict) and "name" in item and "polygon" in item:
                        self.add_zone(cam_name, item["name"], item["polygon"])

    def add_zone(
        self,
        camera: str,
        name: str,
        polygon: Sequence[Sequence[float]],
    ) -> Zone:
        """Register a new zone for a camera."""
        poly_tuples = [(float(p[0]), float(p[1])) for p in polygon]
        zone = Zone(name=name, polygon=poly_tuples)
        self._zones.setdefault(camera, {})[name] = zone
        return zone

    def get_zones(self, camera: str) -> dict[str, Zone]:
        """Return dict of zone_name -> Zone for a camera."""
        return self._zones.get(camera, {})

    def find_zones_for_point(
        self,
        camera: str,
        point: tuple[float, float],
        width: int,
        height: int,
        include_edge: bool = True,
    ) -> list[str]:
        """Return list of zone names that contain the given point."""
        cam_zones = self.get_zones(camera)
        matched = []
        for name, zone in cam_zones.items():
            if zone.contains_point(point, width, height, include_edge=include_edge):
                matched.append(name)
        return matched

    def find_zones_for_box(
        self,
        camera: str,
        box: Sequence[float | int] | tuple[int, int, int, int],
        width: int,
        height: int,
        include_edge: bool = True,
    ) -> list[str]:
        """Return list of zone names that contain the bottom-centre of box."""
        bc = get_bottom_centre(box)
        return self.find_zones_for_point(
            camera, bc, width, height, include_edge=include_edge
        )
