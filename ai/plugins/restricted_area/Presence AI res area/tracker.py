"""Simple IoU-based person tracker for restricted area detection.

Tracks persons across consecutive frames to support:
- Inertia (presence for multiple consecutive frames before triggering zone_entered)
- Loitering (staying in a restricted zone for longer than a threshold duration)
- Bottom-centre foot tracking
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TrackedPerson:
    """State for one tracked person across frames."""

    track_id: str
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    disappeared: int = 0
    is_new: bool = True  # True only on the frame the track was created

    @property
    def centre(self) -> tuple[float, float]:
        """Horizontal and vertical midpoint of the box."""
        x1, y1, x2, y2 = self.box
        return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)

    @property
    def bottom_centre(self) -> tuple[float, float]:
        """Bottom-centre of the box (representing feet on the ground plane)."""
        x1, y1, x2, y2 = self.box
        return ((float(x1) + float(x2)) / 2.0, float(y2))


class SimpleTracker:
    """Track persons across frames using IoU matching.

    Parameters
    ----------
    max_disappeared : int
        Drop a track after this many consecutive frames without a match.
    iou_threshold : float
        Minimum IoU to consider a detection a match for an existing track.
    """

    def __init__(
        self,
        max_disappeared: int = 5,
        iou_threshold: float = 0.3,
    ) -> None:
        self._max_disappeared = max_disappeared
        self._iou_threshold = iou_threshold
        self._tracks: dict[str, TrackedPerson] = {}

    @property
    def tracks(self) -> dict[str, TrackedPerson]:
        """All tracks (including temporarily disappeared)."""
        return dict(self._tracks)

    @property
    def active_tracks(self) -> list[TrackedPerson]:
        """Only tracks that were detected on the most recent frame (disappeared == 0)."""
        return [t for t in self._tracks.values() if t.disappeared == 0]

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def update(
        self, detections: list[dict[str, Any]]
    ) -> list[TrackedPerson]:
        """Feed new detections and return all currently active tracks on this frame.

        Parameters
        ----------
        detections : list[dict]
            Each dict has ``"box": (x1, y1, x2, y2)`` and ``"confidence": float``.

        Returns
        -------
        list[TrackedPerson]
            All active tracks visible on this frame.
        """
        # Reset is_new flag from previous frame
        for t in self._tracks.values():
            t.is_new = False

        if not detections:
            self._tick_disappeared()
            return []

        if not self._tracks:
            # No existing tracks → all detections become new tracks
            return self._register_all(detections)

        # ── Match existing tracks to detections via IoU ───────────────
        track_ids = list(self._tracks.keys())
        det_indices = list(range(len(detections)))

        # Build IoU matrix: rows = tracks, cols = detections
        iou_matrix = np.zeros(
            (len(track_ids), len(detections)), dtype=np.float64
        )
        for i, tid in enumerate(track_ids):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = _iou(self._tracks[tid].box, det["box"])

        matched_tracks: set[str] = set()
        matched_dets: set[int] = set()

        # Greedy matching (highest IoU first)
        while True:
            if iou_matrix.size == 0:
                break
            idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            best_iou = iou_matrix[idx]
            if best_iou < self._iou_threshold:
                break

            ti, di = int(idx[0]), int(idx[1])
            tid = track_ids[ti]
            det = detections[di]

            # Update matched track
            self._tracks[tid].box = det["box"]
            self._tracks[tid].confidence = det["confidence"]
            self._tracks[tid].disappeared = 0

            matched_tracks.add(tid)
            matched_dets.add(di)

            # Zero out the matched row/col so they aren't reused
            iou_matrix[ti, :] = 0
            iou_matrix[:, di] = 0

        # ── Handle unmatched tracks ───────────────────────────────────
        for tid in track_ids:
            if tid not in matched_tracks:
                self._tracks[tid].disappeared += 1

        # Prune long-gone tracks
        self._prune()

        # ── Register unmatched detections as new tracks ───────────────
        for j in det_indices:
            if j not in matched_dets:
                self._register(detections[j])

        # Return all active tracks detected this frame
        return self.active_tracks

    # ----------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------

    def _register(self, det: dict[str, Any]) -> TrackedPerson:
        track = TrackedPerson(
            track_id=uuid.uuid4().hex[:8],
            box=tuple(int(v) for v in det["box"][:4]),
            confidence=float(det["confidence"]),
            is_new=True,
        )
        self._tracks[track.track_id] = track
        return track

    def _register_all(self, detections: list[dict[str, Any]]) -> list[TrackedPerson]:
        return [self._register(d) for d in detections]

    def _tick_disappeared(self) -> list[TrackedPerson]:
        """No detections this frame → increment all disappeared counts."""
        for t in self._tracks.values():
            t.disappeared += 1
        self._prune()
        return []

    def _prune(self) -> None:
        """Remove tracks that have disappeared for too long."""
        gone = [
            tid
            for tid, t in self._tracks.items()
            if t.disappeared > self._max_disappeared
        ]
        for tid in gone:
            del self._tracks[tid]


# ── Utility ─────────────────────────────────────────────────────────────

def _iou(
    box_a: tuple[int, int, int, int] | Sequence[int],
    box_b: tuple[int, int, int, int] | Sequence[int],
) -> float:
    """Compute Intersection-over-Union for two ``(x1, y1, x2, y2)`` boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
