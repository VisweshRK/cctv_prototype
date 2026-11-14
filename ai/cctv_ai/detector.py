"""Shared Object Detector for Presence B2C CCTV AI Platform.
Accurate real-time person and vehicle detection using YOLOX-Nano ONNX.
Extracts standard COCO classes (person, car, motorcycle, bus, truck)
with precomputed grid anchor decoding and OpenCV NMS.
"""

from dataclasses import dataclass
import logging
import os
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    label: str  # "person", "car", "motorcycle", "bus", "truck"
    score: float


class SharedDetector:
    """Single shared real-time object detector per stream.
    Reused across Intrusion, Restricted Area, Occupancy, Fall, Face, Vehicle, Electricity.
    Powered by YOLOX-Nano ONNX.
    """

    # COCO Class IDs for relevant Presence B2C safety classes
    TARGET_CLASSES = {
        0: "person",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        target_size: Tuple[int, int] = (416, 416),
        use_int8: bool = False
    ):
        self.use_int8 = use_int8 or os.getenv("USE_INT8_DETECTOR", "0") == "1"
        
        # Default model search locations (supports use-case names electricity.onnx & yolox_nano.onnx)
        if self.use_int8:
            default_paths = [
                model_path,
                "e:/cctv/ai/models/electricity_int8.onnx",
                "ai/models/electricity_int8.onnx",
                "e:/cctv/backend/models/electricity_int8.onnx",
                "backend/models/electricity_int8.onnx",
                "e:/cctv/backend/models/yolox_nano_int8.onnx",
                "backend/models/yolox_nano_int8.onnx",
                "e:/cctv/ai/models/yolox_nano_int8.onnx",
                "ai/models/yolox_nano_int8.onnx",
                "e:/cctv/ai/models/electricity.onnx",
                "ai/models/electricity.onnx",
                "e:/cctv/backend/models/yolox_nano.onnx",
                "backend/models/yolox_nano.onnx",
                "e:/cctv/ai/models/yolox_nano.onnx",
                "ai/models/yolox_nano.onnx",
            ]
        else:
            default_paths = [
                model_path,
                "e:/cctv/ai/models/electricity.onnx",
                "ai/models/electricity.onnx",
                "e:/cctv/backend/models/electricity.onnx",
                "backend/models/electricity.onnx",
                "e:/cctv/backend/models/yolox_nano.onnx",
                "backend/models/yolox_nano.onnx",
                "e:/cctv/ai/models/yolox_nano.onnx",
                "ai/models/yolox_nano.onnx",
                "e:/cctv/ai/models/electricity_int8.onnx",
                "ai/models/electricity_int8.onnx",
                "e:/cctv/backend/models/yolox_nano_int8.onnx",
                "backend/models/yolox_nano_int8.onnx",
                "e:/cctv/ai/models/yolox_nano_int8.onnx",
                "ai/models/yolox_nano_int8.onnx",
                r"C:\Users\HP\Downloads\yolox_nano.onnx"
            ]
        resolved_path = None
        for p in default_paths:
            if p and os.path.exists(p):
                resolved_path = os.path.abspath(p)
                break

        self.model_path = resolved_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.target_size = target_size
        self.session = None
        self.backend = "uninitialized"
        if resolved_path and "int8" in resolved_path.lower():
            self.model_version = "yolox_nano_int8_1.0.0"
        else:
            self.model_version = "yolox_nano_onnx_1.0.0"
        self.latency_ms = 0.0

        # Precompute grid anchors and strides for fast real-time decoding
        self._init_grids()

        # Fallback HOG descriptor if model file missing
        self._hog = None

        self._init_backend()

    def _init_grids(self) -> None:
        """Precomputes grid coordinates for 8, 16, 32 stride anchors."""
        strides = [8, 16, 32]
        grids, expanded_strides = [], []
        tw, th = self.target_size
        for s in strides:
            h, w = th // s, tw // s
            y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            grid = np.stack((x, y), 2).reshape(1, -1, 2)
            grids.append(grid)
            expanded_strides.append(np.full((1, grid.shape[1], 1), s))

        self._grids = np.concatenate(grids, 1).astype(np.float32)
        self._strides = np.concatenate(expanded_strides, 1).astype(np.float32)

    def _init_backend(self) -> None:
        if self.model_path and os.path.exists(self.model_path):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.session = ort.InferenceSession(
                    self.model_path,
                    opts,
                    providers=["CPUExecutionProvider"]
                )
                self.input_name = self.session.get_inputs()[0].name
                self.backend = "onnx_yolox_nano"
                if self.model_path and "int8" in self.model_path.lower():
                    self.model_version = "yolox_nano_int8_1.0.0"
                else:
                    self.model_version = "yolox_nano_onnx_1.0.0"
                logger.info(f"Loaded real-time YOLOX-Nano ONNX model ({self.model_version}) from: {self.model_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load ONNX Runtime session: {e}")

        # Fallback to OpenCV HOG People Detector
        try:
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.backend = "opencv_hog"
            self.model_version = "opencv_hog_people_v1"
            logger.info("Using OpenCV HOG people fallback detector.")
        except Exception:
            self.backend = "geometric_motion_detector"
            self.model_version = "fallback_motion_v1"

    def detect(
        self,
        frame: np.ndarray,
        regions: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> List[dict]:
        """Runs accurate real-time detection on the frame or specific candidate regions.
        Returns list of dicts: {"box": (x1, y1, x2, y2), "label": str, "score": float}
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]

        # If regions are provided, detect within candidate regions and map coordinates
        if regions and len(regions) > 0:
            return self._detect_in_regions(frame, regions, h, w)

        if self.backend == "onnx_yolox_nano" and self.session is not None:
            return self._detect_yolox(frame, h, w)
        elif self.backend == "opencv_hog" and self._hog is not None:
            return self._detect_hog(frame, h, w)
        else:
            return []

    def _detect_in_regions(
        self,
        frame: np.ndarray,
        regions: List[Tuple[int, int, int, int]],
        full_h: int,
        full_w: int
    ) -> List[dict]:
        """Infers YOLOX on motion candidate regions and projects boxes back to frame coordinates."""
        raw_detections = []
        for rx1, ry1, rx2, ry2 in regions:
            rx1 = max(0, min(full_w - 1, rx1))
            ry1 = max(0, min(full_h - 1, ry1))
            rx2 = max(rx1 + 1, min(full_w, rx2))
            ry2 = max(ry1 + 1, min(full_h, ry2))

            crop = frame[ry1:ry2, rx1:rx2]
            if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
                continue

            ch, cw = crop.shape[:2]
            if self.backend == "onnx_yolox_nano" and self.session is not None:
                crop_dets = self._detect_yolox(crop, ch, cw)
            elif self.backend == "opencv_hog" and self._hog is not None:
                crop_dets = self._detect_hog(crop, ch, cw)
            else:
                crop_dets = []

            for d in crop_dets:
                bx1, by1, bx2, by2 = d["box"]
                gx1 = min(full_w, max(0, bx1 + rx1))
                gy1 = min(full_h, max(0, by1 + ry1))
                gx2 = min(full_w, max(0, bx2 + rx1))
                gy2 = min(full_h, max(0, by2 + ry1))
                raw_detections.append({
                    "box": (gx1, gy1, gx2, gy2),
                    "label": d["label"],
                    "score": d["score"]
                })

        # Apply per-label NMS across overlapping region detections
        final_results = []
        labels = set(d["label"] for d in raw_detections)
        for lbl in labels:
            lbl_dets = [d for d in raw_detections if d["label"] == lbl]
            if not lbl_dets:
                continue
            nms_boxes = [[d["box"][0], d["box"][1], d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]] for d in lbl_dets]
            scores = [d["score"] for d in lbl_dets]
            indices = cv2.dnn.NMSBoxes(nms_boxes, scores, self.confidence_threshold, self.nms_threshold)
            if len(indices) > 0:
                for idx in indices.flatten():
                    final_results.append(lbl_dets[idx])

        return final_results

    def _detect_yolox(self, frame: np.ndarray, h: int, w: int) -> List[dict]:
        """Runs YOLOX-Nano inference and decodes bounding boxes with NMS."""
        t0 = time.time()
        tw, th = self.target_size

        # 1. Letterbox resize (keeps aspect ratio)
        r = min(th / h, tw / w)
        rw, rh = int(w * r), int(h * r)
        padded_img = np.full((th, tw, 3), 114, dtype=np.uint8)
        resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
        padded_img[:rh, :rw] = resized

        # Format to float32 NCHW
        input_tensor = np.ascontiguousarray(padded_img.transpose((2, 0, 1))[None]).astype(np.float32)

        # 2. Run ONNX Session
        outputs = self.session.run(None, {self.input_name: input_tensor})[0]
        # outputs shape: (1, 3549, 85)

        # 3. Vectorized Coordinate and Score Decoding
        box_xy = (outputs[..., :2] + self._grids) * self._strides
        box_wh = np.exp(outputs[..., 2:4]) * self._strides
        pred_boxes = np.concatenate([box_xy - box_wh / 2, box_xy + box_wh / 2], -1)[0]
        # pred_boxes: (3549, 4) in (x1, y1, x2, y2) letterbox coords

        obj_conf = outputs[0, :, 4:5]
        cls_conf = outputs[0, :, 5:]
        scores_matrix = obj_conf * cls_conf  # (3549, 80)

        results: List[dict] = []

        # 4. Filter for Presence B2C Target Classes
        for cls_id, label in self.TARGET_CLASSES.items():
            if cls_id >= scores_matrix.shape[1]:
                continue
            cls_scores = scores_matrix[:, cls_id]
            mask = cls_scores >= self.confidence_threshold

            if not np.any(mask):
                continue

            cand_boxes = pred_boxes[mask]
            cand_scores = cls_scores[mask]

            # Convert to [x, y, w, h] format for cv2.dnn.NMSBoxes
            nms_boxes = []
            for b in cand_boxes:
                bx1 = max(0.0, float(b[0] / r))
                by1 = max(0.0, float(b[1] / r))
                bx2 = min(float(w), float(b[2] / r))
                by2 = min(float(h), float(b[3] / r))
                bw = max(0.0, bx2 - bx1)
                bh = max(0.0, by2 - by1)
                nms_boxes.append([int(bx1), int(by1), int(bw), int(bh)])

            scores_list = [float(s) for s in cand_scores]

            # Apply Non-Maximum Suppression
            indices = cv2.dnn.NMSBoxes(nms_boxes, scores_list, self.confidence_threshold, self.nms_threshold)

            if len(indices) > 0:
                indices = indices.flatten()
                for idx in indices:
                    bx, by, bw, bh = nms_boxes[idx]
                    sc = scores_list[idx]
                    x1 = max(0, bx)
                    y1 = max(0, by)
                    x2 = min(w, bx + bw)
                    y2 = min(h, by + bh)

                    results.append({
                        "box": (x1, y1, x2, y2),
                        "label": label,
                        "score": round(sc, 3)
                    })

        self.latency_ms = round((time.time() - t0) * 1000, 2)
        return results

    def _detect_hog(self, frame: np.ndarray, h: int, w: int) -> List[dict]:
        """Fallback OpenCV HOG people detection."""
        scale = min(1.0, 640.0 / max(h, w))
        small_frame = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        boxes, weights = self._hog.detectMultiScale(gray, winStride=(8, 8), padding=(4, 4), scale=1.05)

        results = []
        for (bx, by, bw, bh), weight in zip(boxes, weights):
            score = float(weight[0]) if hasattr(weight, "__iter__") else float(weight)
            norm_score = min(1.0, max(0.1, 1.0 / (1.0 + np.exp(-score * 2.0))))
            if norm_score >= self.confidence_threshold:
                x1 = max(0, int(bx / scale))
                y1 = max(0, int(by / scale))
                x2 = min(w, int((bx + bw) / scale))
                y2 = min(h, int((by + bh) / scale))
                results.append({
                    "box": (x1, y1, x2, y2),
                    "label": "person",
                    "score": round(norm_score, 3)
                })
        return results
