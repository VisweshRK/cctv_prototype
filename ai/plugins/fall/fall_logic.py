"""fall_logic.py: Tier 1 cheap screening, pose validation, and fall state machine.
Per ZoneFallPlan.md Section 5 and docs/plans/p3-p4-sanjay-kishore.md Section 4.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from cctv_ai.objects import TrackedObject
from .pose import PoseEstimator, PoseResult


@dataclass
class TrackFallState:
    """State tracking for a single person track."""

    first_seen: datetime
    standing_height: float = 0.0  # Learned baseline upright height in pixels
    is_suspicious: bool = False
    suspicious_since: Optional[datetime] = None
    down_since: Optional[datetime] = None
    event_emitted: bool = False
    last_pose: Optional[PoseResult] = None
    last_aspect_ratio: float = 0.5
    upright_since: Optional[datetime] = None
    fast_drop_observed: bool = False


class FallDetector:
    """Evaluates person tracks for falls using Tier 1 cheap screening,
    Tier 2 pose estimation, and a 3-second persistence state machine.
    """

    def __init__(
        self,
        pose_estimator: Optional[PoseEstimator] = None,
        persistence_seconds: float = 2.0,
        aspect_ratio_lying: float = 1.35,
        aspect_ratio_upright: float = 0.85,
        drop_ratio_threshold: float = 0.35,
    ):
        self.pose = pose_estimator or PoseEstimator()
        self.persistence_seconds = persistence_seconds
        self.aspect_ratio_lying = aspect_ratio_lying
        self.aspect_ratio_upright = aspect_ratio_upright
        self.drop_ratio_threshold = drop_ratio_threshold
        # Key: obj_id -> TrackFallState
        self.states: dict[str, TrackFallState] = {}

    def clean_ended_tracks(self, ended_ids: set[str]) -> None:
        """Purge state for tracks that have ended."""
        for obj_id in ended_ids:
            self.states.pop(obj_id, None)

    @staticmethod
    def get_aspect_ratio(box: tuple[int, int, int, int]) -> float:
        """Calculate width / max(height, 1)."""
        x1, y1, x2, y2 = box
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        return float(w) / float(h)

    def check_tier1_drop(self, obj: TrackedObject, state: TrackFallState) -> bool:
        """Examine box history for rapid downward displacement of top edge (y1)."""
        if not obj.history or len(obj.history) < 2:
            return False

        current_box = obj.box
        curr_y1 = current_box[1]
        curr_h = max(1, current_box[3] - current_box[1])

        # Update learned standing height if person appears upright
        curr_ar = self.get_aspect_ratio(current_box)
        if curr_ar < self.aspect_ratio_upright:
            state.standing_height = max(state.standing_height, float(curr_h))

        ref_height = state.standing_height if state.standing_height > 0 else float(curr_h)

        # Check top edge y1 against recent history (~1 second / 5 frames back)
        window = obj.history[-6:]
        oldest_y1 = window[0][1]
        delta_y1 = curr_y1 - oldest_y1  # Positive value means moved downward

        is_drop = delta_y1 > (self.drop_ratio_threshold * ref_height)
        if is_drop:
            state.fast_drop_observed = True
        return is_drop

    def evaluate_track(
        self,
        obj: TrackedObject,
        frame,
        ts: datetime,
        is_in_exclude_zone: bool = False,
    ) -> Optional[dict]:
        """Process a single tracked person. Returns fall event payload dict if a fall is confirmed."""
        if obj.id not in self.states:
            self.states[obj.id] = TrackFallState(first_seen=ts)
        state = self.states[obj.id]

        current_ar = self.get_aspect_ratio(obj.box)
        state.last_aspect_ratio = current_ar

        # Update standing height baseline
        if current_ar < self.aspect_ratio_upright:
            h = obj.box[3] - obj.box[1]
            state.standing_height = max(state.standing_height, float(h))

        # --- Tier 1 Screening ---
        has_fast_drop = self.check_tier1_drop(obj, state)
        is_horizontal_box = current_ar >= self.aspect_ratio_lying

        # Run pose estimation on person crop for EVERY frame (even after fall)
        pose_res = None
        if self.pose.is_available() and frame is not None and frame.size > 0:
            x1, y1, x2, y2 = obj.box
            h_f, w_f = frame.shape[:2]
            pad = 12
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(w_f, x2 + pad)
            cy2 = min(h_f, y2 + pad)
            crop = frame[cy1:cy2, cx1:cx2]

            pose_res = self.pose.estimate(crop)
            state.last_pose = pose_res

        # Determine if posture is currently upright vs lying down
        is_upright = False
        is_lying_posture = False
        torso_angle = 0.0
        pose_conf = obj.score

        if pose_res and pose_res.keypoints_valid:
            torso_angle = pose_res.torso_angle_deg
            pose_conf = pose_res.confidence
            if pose_res.is_upright_sitting or torso_angle < 45.0:
                is_upright = True
            elif pose_res.is_lying_pose or pose_res.head_below_hips:
                is_lying_posture = True
        else:
            if current_ar < 1.05:
                is_upright = True
            elif (state.fast_drop_observed or has_fast_drop) and is_horizontal_box:
                is_lying_posture = True
                torso_angle = 80.0
            elif current_ar >= 1.65:
                is_lying_posture = True
                torso_angle = 85.0

        if pose_res and pose_res.is_upright_sitting:
            is_upright = True
            is_lying_posture = False

        # --- Immediate Recovery Check ---
        # When person gets back up (sitting or standing), immediately clear the fall alert!
        if state.event_emitted:
            if is_upright:
                state.event_emitted = False
                state.down_since = None
                state.is_suspicious = False
                state.fast_drop_observed = False
                state.upright_since = None
            return None

        # If posture is upright, reset suspicious down timer
        if is_upright:
            state.down_since = None
            state.is_suspicious = False
            state.suspicious_since = None
            state.fast_drop_observed = False
            return None

        # --- State Machine & Faster Persistence ---
        if is_lying_posture or has_fast_drop:
            state.is_suspicious = True
            if state.down_since is None:
                state.down_since = ts

            elapsed_down = (ts - state.down_since).total_seconds()

            # Furniture / Bed exclusion check: suppress alert if resting in bed/sofa
            if is_in_exclude_zone:
                return None

            req_persistence = self.persistence_seconds

            if elapsed_down >= req_persistence:
                state.event_emitted = True
                return {
                    "object_id": obj.id,
                    "seconds_down": round(elapsed_down, 1),
                    "torso_angle": round(torso_angle, 1),
                    "confidence": round(float(pose_conf), 2),
                    "aspect_ratio": round(current_ar, 2),
                }

        return None
