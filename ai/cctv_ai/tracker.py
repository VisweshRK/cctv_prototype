"""Shared Object Tracker for Presence B2C CCTV AI Platform.
Implements Plan architecture:
  filters + per-label NMS -> tracker (Norfair, Kalman)
Provides multi-object tracking, Kalman velocity state estimation,
rolling median score smoothing, and motionless analysis.
Emits TrackedObject instances matching Section 4 contract.
"""

from collections import deque
from datetime import datetime, timezone
import statistics
from typing import Dict, List, Optional, Tuple
import uuid
import numpy as np

from ai.cctv_ai.objects import TrackedObject


def calculate_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union = areaA + areaB - inter_area
    return (inter_area / union) if union > 0 else 0.0


class KalmanBoxTracker:
    """Kalman Filter state estimator for bounding box trajectory and velocity.
    State vector: [cx, cy, w, h, vx, vy]^T
    """

    def __init__(self, initial_box: Tuple[int, int, int, int]):
        x1, y1, x2, y2 = initial_box
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        cx = float(x1 + x2) / 2.0
        cy = float(y1 + y2) / 2.0

        # State: [cx, cy, w, h, vx, vy]
        self.x = np.array([cx, cy, w, h, 0.0, 0.0], dtype=np.float32)

        # State transition matrix F
        dt = 1.0
        self.F = np.eye(6, dtype=np.float32)
        self.F[0, 4] = dt
        self.F[1, 5] = dt

        # Measurement matrix H (we measure [cx, cy, w, h])
        self.H = np.zeros((4, 6), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # Covariance matrix P
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 100.0, 100.0]).astype(np.float32)

        # Process noise Q
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 5.0, 5.0]).astype(np.float32)

        # Measurement noise R
        self.R = np.diag([4.0, 4.0, 8.0, 8.0]).astype(np.float32)

    def predict(self) -> Tuple[int, int, int, int]:
        """Predicts the next state using motion model."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.get_box()

    def update(self, measurement_box: Tuple[int, int, int, int]) -> None:
        """Updates the state with a new measurement bounding box."""
        x1, y1, x2, y2 = measurement_box
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        cx = float(x1 + x2) / 2.0
        cy = float(y1 + y2) / 2.0
        z = np.array([cx, cy, w, h], dtype=np.float32)

        # Innovation
        y = z - np.dot(self.H, self.x)
        # Innovation covariance
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        # Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Updated state and covariance
        self.x = self.x + np.dot(K, y)
        I = np.eye(6, dtype=np.float32)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    def get_box(self) -> Tuple[int, int, int, int]:
        cx, cy, w, h = self.x[0], self.x[1], self.x[2], self.x[3]
        x1 = int(round(cx - w / 2.0))
        y1 = int(round(cy - h / 2.0))
        x2 = int(round(cx + w / 2.0))
        y2 = int(round(cy + h / 2.0))
        return (x1, y1, x2, y2)

    @property
    def velocity(self) -> Tuple[float, float]:
        return (float(self.x[4]), float(self.x[5]))

    @property
    def speed(self) -> float:
        return float((self.x[4]**2 + self.x[5]**2) ** 0.5)


class SharedTracker:
    """Shared tracker maintaining track history, Kalman prediction, motionless counters, and track lifecycle."""

    def __init__(
        self,
        camera_id: str,
        iou_threshold: float = 0.3,
        max_missed_frames: int = 15,
        min_hits_to_confirm: int = 3,
        motionless_displacement_px: float = 15.0,
        use_kalman: bool = True
    ):
        self.camera_id = camera_id
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.min_hits_to_confirm = min_hits_to_confirm
        self.motionless_displacement_px = motionless_displacement_px
        self.use_kalman = use_kalman

        # Internal active tracks: track_id -> internal dict
        self._tracks: Dict[str, dict] = {}
        self._frame_count = 0

    def update(
        self,
        detections: List[dict],  # list of {"box": (x1, y1, x2, y2), "label": str, "score": float}
        timestamp: Optional[datetime] = None
    ) -> Tuple[List[TrackedObject], List[TrackedObject]]:
        """Update tracker with frame detections.
        Returns:
            (active_objects, ended_objects)
        """
        if not timestamp:
            timestamp = datetime.now(timezone.utc)
        self._frame_count += 1

        # 1. Predict next box with Kalman filter for each track
        predicted_boxes = {}
        for tid, t_data in self._tracks.items():
            if self.use_kalman and "kalman" in t_data:
                pred = t_data["kalman"].predict()
                predicted_boxes[tid] = pred
            else:
                predicted_boxes[tid] = t_data["box"]

        det_matched = [False] * len(detections)
        track_matched = {tid: False for tid in self._tracks.keys()}

        # 2. Match existing tracks with detections via IoU and label
        for tid, t_data in self._tracks.items():
            best_iou = self.iou_threshold
            best_det_idx = -1
            ref_box = predicted_boxes[tid]

            for d_idx, det in enumerate(detections):
                if det_matched[d_idx]:
                    continue
                if det["label"] != t_data["label"]:
                    continue

                iou = calculate_iou(ref_box, det["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = d_idx

            if best_det_idx != -1:
                det = detections[best_det_idx]
                det_matched[best_det_idx] = True
                track_matched[tid] = True

                # Update Kalman filter
                if self.use_kalman and "kalman" in t_data:
                    t_data["kalman"].update(det["box"])
                    smoothed_box = t_data["kalman"].get_box()
                else:
                    smoothed_box = det["box"]

                old_box = t_data["box"]
                new_box = smoothed_box
                t_data["box"] = new_box
                t_data["scores"].append(det["score"])
                t_data["history"].append(new_box)
                t_data["last_seen"] = timestamp
                t_data["hits"] += 1
                t_data["misses"] = 0

                # Motionless calculation
                c_old = ((old_box[0] + old_box[2]) / 2, (old_box[1] + old_box[3]) / 2)
                c_new = ((new_box[0] + new_box[2]) / 2, (new_box[1] + new_box[3]) / 2)
                disp = ((c_new[0] - c_old[0]) ** 2 + (c_new[1] - c_old[1]) ** 2) ** 0.5

                if disp < self.motionless_displacement_px:
                    t_data["motionless_count"] += 1
                else:
                    t_data["motionless_count"] = 0

                t_data["stationary"] = (t_data["motionless_count"] > 10)

        # 3. Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if not det_matched[d_idx]:
                new_tid = f"track_{uuid.uuid4().hex[:8]}"
                kalman = KalmanBoxTracker(det["box"]) if self.use_kalman else None
                self._tracks[new_tid] = {
                    "id": new_tid,
                    "label": det["label"],
                    "box": det["box"],
                    "scores": deque([det["score"]], maxlen=30),
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "history": deque([det["box"]], maxlen=60),
                    "hits": 1,
                    "misses": 0,
                    "motionless_count": 0,
                    "stationary": False,
                    "kalman": kalman,
                }

        # 4. Age unmatched tracks (Kalman prediction keeps box current during temporary occlusion)
        ended_objects: List[TrackedObject] = []
        to_delete = []

        for tid, t_data in self._tracks.items():
            if not track_matched.get(tid, False):
                t_data["misses"] += 1
                # Use Kalman predicted box if missed
                if self.use_kalman and tid in predicted_boxes:
                    t_data["box"] = predicted_boxes[tid]

                if t_data["misses"] > self.max_missed_frames:
                    to_delete.append(tid)
                    if t_data["hits"] >= self.min_hits_to_confirm:
                        ended_objects.append(self._to_tracked_object(t_data))

        for tid in to_delete:
            del self._tracks[tid]

        # 5. Active confirmed tracks
        active_objects: List[TrackedObject] = []
        for tid, t_data in self._tracks.items():
            if t_data["hits"] >= self.min_hits_to_confirm and t_data["misses"] == 0:
                active_objects.append(self._to_tracked_object(t_data))

        return active_objects, ended_objects

    def _to_tracked_object(self, t_data: dict) -> TrackedObject:
        scores = list(t_data["scores"])
        med_score = statistics.median(scores) if scores else 0.5
        curr_score = scores[-1] if scores else 0.5
        return TrackedObject(
            id=t_data["id"],
            label=t_data["label"],
            box=t_data["box"],
            score=float(curr_score),
            median_score=float(med_score),
            true_positive=(med_score >= 0.4 and t_data["hits"] >= self.min_hits_to_confirm),
            stationary=t_data["stationary"],
            motionless_count=t_data["motionless_count"],
            first_seen=t_data["first_seen"],
            last_seen=t_data["last_seen"],
            history=list(t_data["history"]),
        )
