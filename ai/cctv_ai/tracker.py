"""tracker.py: Production Multi-Object Tracker for CCTV AI (Apache 2.0 / BSD compatible).
Provides stable identity tracking, Kalman-style linear motion prediction,
stationary person classification, and median-score true-positive filtering.
Follows the contract defined in docs/plans/README.md and docs/how-frigate-works.md.
"""
from __future__ import annotations

from datetime import datetime, timezone
import statistics
from typing import Optional

import numpy as np

from .objects import TrackedObject


def box_iou(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes (x1, y1, x2, y2)."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    a1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = a1 + a2 - inter
    return float(inter) / float(union) if union > 0 else 0.0


def box_center(b: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


class SingleTrack:
    """Internal state for a single tracked person."""

    def __init__(self, track_id: str, label: str, box: tuple[int, int, int, int], score: float, ts: datetime):
        self.id = track_id
        self.label = label
        self.box = box
        self.scores: list[float] = [score]
        self.true_positive: bool = score >= 0.70
        self.first_seen = ts
        self.last_seen = ts
        self.history: list[tuple[int, int, int, int]] = [box]
        self.lost_count = 0
        self.motionless_count = 0
        self.stationary = False

        # Velocity tracking (dx, dy per frame)
        self.velocity = (0.0, 0.0)

    def predict(self) -> tuple[int, int, int, int]:
        """Predict box position in next frame using velocity estimate."""
        if self.stationary:
            return self.box

        vx, vy = self.velocity
        x1, y1, x2, y2 = self.box
        pred_x1 = int(x1 + vx)
        pred_y1 = int(y1 + vy)
        pred_x2 = int(x2 + vx)
        pred_y2 = int(y2 + vy)
        return (pred_x1, pred_y1, pred_x2, pred_y2)

    def update(self, box: tuple[int, int, int, int], score: float, ts: datetime) -> None:
        """Update track with a newly matched detection."""
        # Calculate displacement for velocity & stationary status
        prev_center = box_center(self.box)
        new_center = box_center(box)
        dx = new_center[0] - prev_center[0]
        dy = new_center[1] - prev_center[1]

        # Smooth velocity with exponential moving average
        alpha = 0.5
        self.velocity = (alpha * dx + (1 - alpha) * self.velocity[0], alpha * dy + (1 - alpha) * self.velocity[1])

        # Check for stationary status (IoU > 0.85 with previous box and minimal center movement)
        iou_prev = box_iou(self.box, box)
        h = max(1, box[3] - box[1])
        disp = (dx * dx + dy * dy) ** 0.5
        if iou_prev >= 0.80 or disp < (0.05 * h):
            self.motionless_count += 1
        else:
            self.motionless_count = max(0, self.motionless_count - 2)

        # Flagged stationary after 10 consecutive motionless frames (~2 s @ 5 fps)
        self.stationary = self.motionless_count >= 10

        self.box = box
        self.scores.append(score)
        if len(self.scores) > 10:
            self.scores.pop(0)

        # Calculate median score of last 10 frames
        med = statistics.median(self.scores)
        bw = max(1, box[2] - box[0])
        bh = max(1, box[3] - box[1])
        aspect_h_over_w = bh / bw

        # Prevent static background furniture (shelves, coffee machines, horizontal appliances) from becoming true_positive:
        # 1. Motionless objects must have human vertical proportions (H/W >= 0.85) and high confidence (>= 0.55).
        # 2. Moving tracks can confirm with >= 0.45 after 4 consistent frames.
        is_moving_person = (len(self.history) >= 4 and med >= 0.45 and self.motionless_count < len(self.history))
        is_stationary_person = (med >= 0.55 and aspect_h_over_w >= 0.85 and len(self.history) >= 3)
        if is_moving_person or is_stationary_person:
            self.true_positive = True

        self.last_seen = ts
        self.history.append(box)
        if len(self.history) > 30:
            self.history.pop(0)

        self.lost_count = 0

    def mark_missed(self) -> None:
        """Handle frame where object was not detected."""
        self.lost_count += 1
        # For genuine confirmed seated/stationary persons, maintain confidence so count doesn't oscillate to 0.
        # But if the track has zero real human motion history and weak confidence, let it decay so phantom shelves disappear.
        is_ghost_stationary = self.stationary and (self.motionless_count >= len(self.history)) and (statistics.median(self.scores) < 0.52)
        if not self.stationary or is_ghost_stationary:
            self.scores.append(0.0)
            if len(self.scores) > 10:
                self.scores.pop(0)
        # Apply predicted box for continuity
        self.box = self.predict()
        self.history.append(self.box)
        if len(self.history) > 30:
            self.history.pop(0)

    def to_tracked_object(self) -> TrackedObject:
        """Convert to the shared contract TrackedObject."""
        current_score = self.scores[-1] if self.scores else 0.0
        med = statistics.median(self.scores) if self.scores else 0.0
        return TrackedObject(
            id=self.id,
            label=self.label,
            box=self.box,
            score=current_score,
            median_score=med,
            true_positive=self.true_positive,
            stationary=self.stationary,
            motionless_count=self.motionless_count,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            history=list(self.history),
        )


class PersonTracker:
    """Multi-object tracker with persistence, Kalman velocity prediction, and stationary recognition."""

    def __init__(self, camera: str = "cam1", iou_threshold: float = 0.28, max_lost_frames: int = 40):
        self.camera = camera
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.next_id = 1
        self.tracks: dict[str, SingleTrack] = {}

    def update(
        self, detections: list[dict], ts: datetime
    ) -> tuple[list[TrackedObject], list[TrackedObject]]:
        """Updates tracks with a new frame's detections.
        detections: list of {"label": str, "box": (x1, y1, x2, y2), "score": float}
        Returns: (active_objects, ended_objects)
        """
        # Filter detections for person
        person_dets = [d for d in detections if d.get("label") == "person"]

        track_ids = list(self.tracks.keys())
        assigned_det_indices = set()
        assigned_track_ids = set()

        # Step 1: Match existing tracks with detections using IoU matrix
        if track_ids and person_dets:
            # Build cost matrix
            cost_matrix = []
            for tid in track_ids:
                track = self.tracks[tid]
                pred_box = track.predict()
                row = []
                for det in person_dets:
                    iou = box_iou(pred_box, det["box"])
                    row.append(iou)
                cost_matrix.append(row)

            # Greedy matching in order of highest IoU
            matched_pairs = []
            while True:
                best_iou = 0.0
                best_t_idx = -1
                best_d_idx = -1
                for t_idx in range(len(track_ids)):
                    if track_ids[t_idx] in assigned_track_ids:
                        continue
                    for d_idx in range(len(person_dets)):
                        if d_idx in assigned_det_indices:
                            continue
                        iou = cost_matrix[t_idx][d_idx]
                        if iou > best_iou:
                            best_iou = iou
                            best_t_idx = t_idx
                            best_d_idx = d_idx

                if best_iou >= self.iou_threshold and best_t_idx >= 0 and best_d_idx >= 0:
                    tid = track_ids[best_t_idx]
                    matched_pairs.append((tid, best_d_idx))
                    assigned_track_ids.add(tid)
                    assigned_det_indices.add(best_d_idx)
                else:
                    break

            # Update matched tracks
            for tid, d_idx in matched_pairs:
                det = person_dets[d_idx]
                self.tracks[tid].update(det["box"], det["score"], ts)

        # Step 2: Handle unmatched existing tracks
        for tid in track_ids:
            if tid not in assigned_track_ids:
                track = self.tracks[tid]
                track.mark_missed()

        # Step 3: Create new tracks for unmatched detections
        for d_idx, det in enumerate(person_dets):
            if d_idx not in assigned_det_indices:
                tid = f"{self.camera}-{self.next_id}"
                self.next_id += 1
                new_track = SingleTrack(
                    track_id=tid,
                    label="person",
                    box=det["box"],
                    score=det["score"],
                    ts=ts,
                )
                self.tracks[tid] = new_track

        # Step 4: Purge expired tracks
        active_objects: list[TrackedObject] = []
        ended_objects: list[TrackedObject] = []

        expired_ids = []
        for tid, track in self.tracks.items():
            if track.lost_count > self.max_lost_frames:
                expired_ids.append(tid)
                ended_objects.append(track.to_tracked_object())
            else:
                active_objects.append(track.to_tracked_object())

        for tid in expired_ids:
            del self.tracks[tid]

        # Step 5: Deduplicate overlapping tracks tracking the exact same person
        # Note: In social environments (e.g. people seated at a table or walking together),
        # individual bounding boxes overlap with IoU ~ 0.20-0.45. Only merge when IoU is high (>= 0.55)
        # or one is virtually completely contained inside another with aligned centers.
        deduped_active: list[TrackedObject] = []
        for obj in sorted(active_objects, key=lambda o: (o.true_positive, o.stationary, o.median_score, len(o.history)), reverse=True):
            suppressed = False
            b1 = obj.box
            w1 = b1[2] - b1[0]
            h1 = b1[3] - b1[1]
            a1 = max(1, w1 * h1)
            cx1 = (b1[0] + b1[2]) / 2.0

            for kept in deduped_active:
                b2 = kept.box
                w2 = b2[2] - b2[0]
                h2 = b2[3] - b2[1]
                a2 = max(1, w2 * h2)
                cx2 = (b2[0] + b2[2]) / 2.0

                ix1 = max(b1[0], b2[0])
                iy1 = max(b1[1], b2[1])
                ix2 = min(b1[2], b2[2])
                iy2 = min(b1[3], b2[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if inter > 0:
                    union = a1 + a2 - inter
                    iou = inter / union if union > 0 else 0.0
                    containment = inter / a1
                    is_inner_sub_box = (containment >= 0.60) and (b1[1] >= b2[1] + 0.15 * h2)
                    if iou >= 0.55 or is_inner_sub_box or (containment >= 0.85 and abs(cx1 - cx2) < 0.15 * min(w1, w2)):
                        suppressed = True
                        break
            if not suppressed:
                deduped_active.append(obj)

        return deduped_active, ended_objects
