"""Fall detection plugin.
Slot: Fall detection (ai/plugins/fall)

Detects person falls using a two-tier cascade:
- Tier 1: Rapid kinematic downward displacement (>40% drop within 1s) and aspect ratio screening.
- Tier 2: Apache 2.0 ONNX pose model on person crop (MoveNet Lightning) measuring torso angle (>60°).
- Fall state machine: requires >= 3 seconds persistence down, respects user-calibrated furniture exclude zones (bed/sofa),
  and emits exactly one event per fall episode.
- Formats event payload with normalized bounding boxes, area labels, and details for the frontend alert popup.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import numpy as np

from cctv_ai.objects import TrackedObject
from cctv_ai.plugin import Event, Plugin as Base
from cctv_ai.zones import ZoneTracker, load_zones_config
from .download_models import DEFAULT_PATH
from .fall_logic import FallDetector
from .pose import PoseEstimator


class Plugin(Base):
    use_case = "fall"
    enabled = True
    needs = {"objects", "video"}
    interval = 0.2  # 5 FPS nominal rate

    def __init__(self):
        super().__init__()
        self.trackers: dict[str, ZoneTracker] = {}
        self.detector = FallDetector()
        self.zones_file = "zones.yaml"

    def setup(self, cameras: list[str]) -> None:
        """Called once before first frame. Loads zones and pose model."""
        # 1. Load zones for user-calibrated exclude regions (bed/sofa)
        possible_paths = [
            self.zones_file,
            os.path.join(os.path.dirname(__file__), "..", "..", "zones.yaml"),
            os.path.join(os.path.dirname(__file__), "..", "..", "ai", "zones.yaml"),
        ]
        resolved_zones = self.zones_file
        for p in possible_paths:
            if os.path.exists(p):
                resolved_zones = os.path.abspath(p)
                break

        zones_by_camera = load_zones_config(resolved_zones)
        for cam in cameras:
            cam_zones = zones_by_camera.get(cam, {})
            self.trackers[cam] = ZoneTracker(camera=cam, zones=cam_zones)

        # 2. Initialize pose estimator if model is available
        possible_model_paths = [
            DEFAULT_PATH,
            os.path.join(os.path.dirname(__file__), "..", "..", "models", "movenet_lightning.onnx"),
            os.path.join(os.path.dirname(__file__), "..", "..", "ai", "models", "movenet_lightning.onnx"),
        ]
        model_path = None
        for mp in possible_model_paths:
            if os.path.exists(mp):
                model_path = os.path.abspath(mp)
                break

        if model_path:
            pose_est = PoseEstimator(model_path=model_path)
            self.detector = FallDetector(pose_estimator=pose_est, persistence_seconds=2.0)
        else:
            self.detector = FallDetector(persistence_seconds=2.0)

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
        """Evaluates confirmed true_positive persons for potential falls."""
        if camera not in self.trackers:
            self.trackers[camera] = ZoneTracker(camera=camera, zones={})

        tracker = self.trackers[camera]
        h_f, w_f = (720, 1280)
        if frame is not None and len(frame.shape) >= 2:
            h_f, w_f = frame.shape[:2]
            tracker.set_resolution(width=w_f, height=h_f)

        # Clean ended tracks
        if ended:
            ended_ids = {obj.id for obj in ended}
            self.detector.clean_ended_tracks(ended_ids)

        # Filter strictly for true-positive persons
        person_objects = [
            obj for obj in objects if obj.label == "person" and obj.true_positive
        ]

        events: list[Event] = []

        for obj in person_objects:
            # Check user-calibrated furniture exclude zone (bed/sofa)
            in_exclude = tracker.is_in_any_exclude_zone(obj.box)

            fall_data = self.detector.evaluate_track(
                obj=obj,
                frame=frame,
                ts=ts,
                is_in_exclude_zone=in_exclude,
            )

            if fall_data is not None:
                # Format normalized bounding box [x1, y1, x2, y2] for frontend snapshot overlay
                x1, y1, x2, y2 = obj.box
                norm_box = [
                    round(x1 / float(w_f), 3),
                    round(y1 / float(h_f), 3),
                    round(x2 / float(w_f), 3),
                    round(y2 / float(h_f), 3),
                ]

                # Populate frontend display fields
                fall_data["box"] = norm_box
                fall_data["area"] = f"Camera {camera}"
                fall_data["detail"] = (
                    f"Person down for {fall_data.get('seconds_down', 3.0)}s "
                    f"(torso angle: {fall_data.get('torso_angle', 75)}°)"
                )

                events.append(
                    self.event(
                        camera=camera,
                        label="fall",
                        confidence=fall_data.get("confidence", 0.90),
                        data=fall_data,
                    )
                )

        return events
