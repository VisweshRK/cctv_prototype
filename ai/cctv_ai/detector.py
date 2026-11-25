"""detector.py: Production YOLOX ONNX Person Detector (Apache 2.0).
Provides high-accuracy, real-time person detection robust to all postures
(standing, sitting, crouching, and lying on the floor).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("cctv_ai.detector")

# Default COCO classes (class 0 is 'person')
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
)


class YOLOXDetector:
    """High-performance ONNX detector for YOLOX models (Apache 2.0)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_thresh: float = 0.45,
        nms_thresh: float = 0.50,
        input_size: tuple[int, int] = (416, 416),
        target_classes: Optional[set[str]] = None,
    ):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_size
        self.target_classes = target_classes or {"person"}
        self.session = None

        if model_path is None:
            # Check default model location
            default_p = os.path.join(os.path.dirname(__file__), "..", "models", "yolox_tiny.onnx")
            if os.path.exists(default_p):
                model_path = os.path.abspath(default_p)

        self.model_path = model_path
        self._load_session()

    def _load_session(self) -> None:
        if not self.model_path or not os.path.exists(self.model_path):
            log.warning("YOLOX model path not found: %s", self.model_path)
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
            if len(inp_shape) == 4 and inp_shape[2] is not None and inp_shape[3] is not None:
                self.input_size = (int(inp_shape[2]), int(inp_shape[3]))
            log.info("Initialized YOLOX detector from %s with input size %s", self.model_path, self.input_size)
        except Exception as err:
            log.error("Failed to load YOLOX ONNX session: %s", err)
            self.session = None

    def is_available(self) -> bool:
        return self.session is not None

    def _preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """Preprocesses image with letterbox padding to preserve aspect ratio."""
        target_h, target_w = self.input_size
        h, w = img.shape[:2]
        r = min(target_h / h, target_w / w)
        resized_w = int(w * r)
        resized_h = int(h * r)

        resized_img = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        padded_img = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded_img[:resized_h, :resized_w] = resized_img

        # Transpose HWC -> CHW, float32
        padded_img = padded_img.transpose((2, 0, 1)).astype(np.float32)
        padded_img = np.ascontiguousarray(padded_img)
        return padded_img, r

    def _postprocess(
        self, outputs: np.ndarray, r: float, orig_shape: tuple[int, int]
    ) -> list[dict]:
        """Decodes YOLOX output grid and applies non-maximum suppression (NMS)."""
        # outputs shape: (1, num_anchors, 85)
        predictions = outputs[0]
        grids = []
        strides = []
        target_h, target_w = self.input_size
        strides_list = [8, 16, 32]
        hsize = [target_h // s for s in strides_list]
        wsize = [target_w // s for s in strides_list]

        for h, w, stride in zip(hsize, wsize, strides_list):
            yv, xv = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, axis=1)[0]
        strides = np.concatenate(strides, axis=1)[0]

        # Decode box centers & sizes (use copy to prevent in-place mutation of predictions)
        boxes = predictions[:, :4].copy()
        boxes[:, :2] = (boxes[:, :2] + grids) * strides
        boxes[:, 2:4] = np.exp(boxes[:, 2:4]) * strides

        # Convert [cx, cy, w, h] to [x1, y1, x2, y2]
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        # Scale back to original image coordinates
        orig_h, orig_w = orig_shape
        x1 = np.clip(x1 / r, 0, orig_w)
        y1 = np.clip(y1 / r, 0, orig_h)
        x2 = np.clip(x2 / r, 0, orig_w)
        y2 = np.clip(y2 / r, 0, orig_h)

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        scores = predictions[:, 4:5] * predictions[:, 5:]
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        # Inanimate object disambiguation (suppress bags/backpacks/suitcases/chairs falsely classified as person)
        OBJ_INDICES = [24, 26, 28, 56, 57]  # backpack, handbag, suitcase, chair, couch
        obj_competing_scores = np.max(scores[:, OBJ_INDICES], axis=1)

        is_person = class_ids == 0
        bag_confusion = is_person & (confidences < 0.65) & (obj_competing_scores >= 0.65 * confidences)

        mask = (confidences >= self.conf_thresh) & (~bag_confusion)
        filtered_boxes = boxes_xyxy[mask]
        filtered_scores = confidences[mask]
        filtered_classes = class_ids[mask]

        if len(filtered_boxes) == 0:
            return []

        # NMS per class
        results = []
        indices = cv2.dnn.NMSBoxes(
            bboxes=[[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in filtered_boxes],
            scores=[float(s) for s in filtered_scores],
            score_threshold=self.conf_thresh,
            nms_threshold=self.nms_thresh,
        )

        if len(indices) == 0:
            return []

        for idx in indices.flatten():
            cls_name = COCO_CLASSES[filtered_classes[idx]] if filtered_classes[idx] < len(COCO_CLASSES) else "unknown"
            if self.target_classes and cls_name not in self.target_classes:
                continue

            b = filtered_boxes[idx]
            bw = b[2] - b[0]
            bh = b[3] - b[1]
            # Minimum dimension threshold (reject tiny bags or stray patches)
            if bw < 25 or bh < 30:
                continue

            results.append({
                "label": cls_name,
                "box": (int(b[0]), int(b[1]), int(b[2]), int(b[3])),
                "score": float(filtered_scores[idx]),
            })

        return self._suppress_nested_boxes(results)

    @staticmethod
    def _suppress_nested_boxes(dets: list[dict], iou_thresh: float = 0.55) -> list[dict]:
        """Disentangles composite group boxes covering multiple individuals (e.g. people sitting together at a table)
        and preserves depth-occluded individuals (e.g. barista standing behind a counter person).
        """
        if len(dets) <= 1:
            return dets

        dets = sorted(dets, key=lambda d: d["score"], reverse=True)
        kept = []

        for d in dets:
            b1 = list(d["box"])
            w1 = b1[2] - b1[0]
            h1 = b1[3] - b1[1]
            a1 = max(1, w1 * h1)
            cx1 = (b1[0] + b1[2]) / 2.0
            cy1 = (b1[1] + b1[3]) / 2.0

            suppressed = False
            for k in kept:
                b2 = list(k["box"])
                w2 = b2[2] - b2[0]
                h2 = b2[3] - b2[1]
                a2 = max(1, w2 * h2)
                cx2 = (b2[0] + b2[2]) / 2.0
                cy2 = (b2[1] + b2[3]) / 2.0

                ix1 = max(b1[0], b2[0])
                iy1 = max(b1[1], b2[1])
                ix2 = min(b1[2], b2[2])
                iy2 = min(b1[3], b2[3])

                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if inter <= 0:
                    continue

                union = a1 + a2 - inter
                iou = inter / union if union > 0 else 0.0
                containment1 = inter / a1  # proportion of d inside k
                containment2 = inter / a2  # proportion of k inside d

                # 1. Internal Torso/Chest/Limb Sub-box Suppression:
                # If d is substantially inside k (>= 60%), and d starts below k's shoulders (>= 15% down k's height),
                # it is physically impossible to be a separate person. It is a floating chest/torso/limb sub-box on the same person.
                is_inner_sub_box = (containment1 >= 0.60) and (b1[1] >= b2[1] + 0.15 * h2)
                if is_inner_sub_box:
                    suppressed = True
                    break

                # 2. Side-by-side Group Box Resolution (e.g. cafe table):
                # ONLY applies if both individuals have heads at a similar vertical height (seated/standing beside each other)
                heads_aligned = abs(b1[1] - b2[1]) <= 0.35 * min(h1, h2)
                if heads_aligned and containment1 >= 0.70 and w2 >= 1.35 * w1:
                    if cx1 < cx2:
                        k["box"] = (int(max(b2[0] + w1 * 0.45, cx1 + w1 * 0.25)), b2[1], b2[2], b2[3])
                    else:
                        k["box"] = (b2[0], b2[1], int(min(b2[2] - w1 * 0.45, cx1 - w1 * 0.25)), b2[3])
                    continue

                elif heads_aligned and containment2 >= 0.70 and w1 >= 1.35 * w2:
                    if cx2 < cx1:
                        b1[0] = int(max(b1[0] + w2 * 0.45, cx2 + w2 * 0.25))
                    else:
                        b1[2] = int(min(b1[2] - w2 * 0.45, cx2 - w2 * 0.25))
                    d["box"] = tuple(b1)
                    continue

                # 3. Depth Occlusion Preservation:
                # Background person (e.g. barista) occluded behind foreground person
                # Background person must have head visible near the top (b1[1] <= b2[1] + 0.15 * h2)
                is_depth_occluded_person = (b1[1] <= b2[1] + 0.15 * h2) and (
                    abs(cx1 - cx2) > 0.15 * min(w1, w2) or abs(b1[1] - b2[1]) > 15
                )
                if is_depth_occluded_person and iou < 0.48:
                    continue

                # 4. Only suppress true duplicate jitter on the exact same person
                if iou >= iou_thresh or (containment1 >= 0.85 and abs(cx1 - cx2) < 0.15 * min(w1, w2)):
                    suppressed = True
                    break

            if not suppressed:
                kept.append(d)

        return kept

    def detect(self, img: np.ndarray) -> list[dict]:
        """Runs detection on a BGR image.
        Returns: list of {"label": str, "box": (x1, y1, x2, y2), "score": float}
        """
        if not self.is_available() or img is None or img.size == 0:
            return []

        orig_shape = img.shape[:2]
        padded_img, r = self._preprocess(img)
        input_tensor = np.expand_dims(padded_img, axis=0)

        outputs = self.session.run(None, {self.input_name: input_tensor})
        return self._postprocess(outputs[0], r, orig_shape)
