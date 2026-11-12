"""pose.py: Lightweight Apache 2.0 ONNX Pose Estimator (MoveNet Lightning).
Extracts 17 COCO body keypoints from a person crop and computes biomechanical features
(torso angle from vertical, head-above-shoulders upright verification, head below hips).
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("cctv_ai.fall.pose")


@dataclass
class PoseResult:
    """Biomechanical pose features extracted from a person crop."""

    torso_angle_deg: float  # 0° = vertical, 90° = horizontal
    is_lying_pose: bool  # True if torso_angle > 60°
    head_below_hips: bool
    is_upright_sitting: bool  # True if head is upright high above shoulders
    confidence: float
    keypoints_valid: bool


class PoseEstimator:
    """Runs ONNX pose estimation on person crops. Uses MoveNet Lightning (Apache 2.0)."""

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            for p in [
                os.path.join(os.path.dirname(__file__), "..", "..", "models", "movenet_lightning.onnx"),
                os.path.join(os.path.dirname(__file__), "..", "..", "ai", "models", "movenet_lightning.onnx"),
            ]:
                if os.path.exists(p):
                    model_path = os.path.abspath(p)
                    break
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.input_shape = (192, 192)  # MoveNet Lightning default
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path or not os.path.exists(self.model_path):
            log.warning("MoveNet model path not found: %s", self.model_path)
            return

        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                self.model_path, sess_options=opts, providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            inp_shape = self.session.get_inputs()[0].shape
            if len(inp_shape) == 4 and inp_shape[1] is not None and inp_shape[2] is not None:
                self.input_shape = (int(inp_shape[1]), int(inp_shape[2]))
            log.info("Loaded pose model from %s", self.model_path)
        except Exception as err:
            log.warning("Could not initialize ONNX pose session: %s", err)
            self.session = None

    def is_available(self) -> bool:
        return self.session is not None

    def estimate(self, crop: np.ndarray) -> PoseResult:
        """Run pose inference on a cropped person bounding box."""
        if not self.is_available() or crop is None or crop.size == 0:
            return PoseResult(
                torso_angle_deg=0.0,
                is_lying_pose=False,
                head_below_hips=False,
                is_upright_sitting=False,
                confidence=0.0,
                keypoints_valid=False,
            )

        target_h, target_w = self.input_shape
        resized = cv2.resize(crop, (target_w, target_h))

        inp_meta = self.session.get_inputs()[0]
        channels = int(inp_meta.shape[3]) if len(inp_meta.shape) == 4 and inp_meta.shape[3] is not None else 3
        if channels == 4:
            processed = cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)
        else:
            processed = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        inp_type = inp_meta.type.lower()
        if "uint8" in inp_type:
            inp_data = np.expand_dims(processed.astype(np.uint8), axis=0)
        elif "int" in inp_type and "uint" not in inp_type:
            inp_data = np.expand_dims(processed.astype(np.int32), axis=0)
        else:
            inp_data = np.expand_dims(processed.astype(np.float32), axis=0)

        try:
            outputs = self.session.run(None, {self.input_name: inp_data})
            # MoveNet output shape is [1, 1, 17, 3]: (y, x, score) per keypoint normalized 0..1 in crop
            kpts = outputs[0]
            if len(kpts.shape) == 4:
                kpts = kpts[0, 0]
            elif len(kpts.shape) == 3:
                kpts = kpts[0]

            # Keypoints:
            # 0: nose, 1: left_eye, 2: right_eye, 5: left_shoulder, 6: right_shoulder, 11: left_hip, 12: right_hip
            nose = kpts[0]
            ls = kpts[5]
            rs = kpts[6]
            lh = kpts[11]
            rh = kpts[12]

            sh_conf = max(ls[2], rs[2])  # At least one shoulder visible
            both_sh_conf = min(ls[2], rs[2])
            nose_conf = nose[2]

            # Upright Sitting Detection:
            # When a person is sitting in a chair or behind a desk, head is clearly above shoulders
            # Even if hips are hidden below the desk/table, this is definitively SITTING UPRIGHT!
            is_upright_sitting = False
            if nose_conf >= 0.20 and sh_conf >= 0.20:
                sh_top_y = min(ls[0], rs[0]) if both_sh_conf >= 0.20 else (ls[0] if ls[2] > rs[2] else rs[0])
                if nose[0] < sh_top_y:  # y=0 is top of image, so nose is higher than shoulders
                    is_upright_sitting = True

            hip_conf = min(lh[2], rh[2])

            if both_sh_conf < 0.20 or hip_conf < 0.20:
                # Full body not visible (e.g. sitting at desk, upper body only)
                return PoseResult(
                    torso_angle_deg=0.0,
                    is_lying_pose=False,
                    head_below_hips=False,
                    is_upright_sitting=is_upright_sitting,
                    confidence=float(max(sh_conf, nose_conf)),
                    keypoints_valid=False,
                )

            # Shoulder midpoint (x, y)
            sh_x = (ls[1] + rs[1]) / 2.0
            sh_y = (ls[0] + rs[0]) / 2.0
            # Hip midpoint (x, y)
            hip_x = (lh[1] + rh[1]) / 2.0
            hip_y = (lh[0] + rh[0]) / 2.0

            dx = abs(hip_x - sh_x)
            dy = abs(hip_y - sh_y)

            # Torso angle from vertical (0° = perfectly upright, 90° = horizontal)
            angle_rad = math.atan2(dx, max(dy, 1e-5))
            angle_deg = math.degrees(angle_rad)

            head_below_hips = bool(nose[0] > hip_y and nose[2] >= 0.20)
            is_lying = (angle_deg > 60.0 or head_below_hips) and not is_upright_sitting
            conf = float((both_sh_conf + hip_conf) / 2.0)

            return PoseResult(
                torso_angle_deg=round(angle_deg, 1),
                is_lying_pose=is_lying,
                head_below_hips=head_below_hips,
                is_upright_sitting=is_upright_sitting,
                confidence=round(conf, 2),
                keypoints_valid=True,
            )
        except Exception as err:
            log.warning("Pose inference failed: %s", err)
            return PoseResult(
                torso_angle_deg=0.0,
                is_lying_pose=False,
                head_below_hips=False,
                is_upright_sitting=False,
                confidence=0.0,
                keypoints_valid=False,
            )
