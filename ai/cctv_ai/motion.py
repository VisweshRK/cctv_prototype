"""Motion Detection and Region Generator for Presence B2C CCTV AI Platform.
Implements Frigate Section 3 (motion -> regions -> shared detector):
1. Fast frame differencing on downscaled blurred grayscale frames.
2. Contours extraction and filtering by min_motion_area.
3. Merging and expanding motion boxes into square candidate regions for YOLOX.
4. Gating detection so YOLOX only runs on motion regions or active scenes.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class MotionResult:
    has_motion: bool
    motion_boxes: List[Tuple[int, int, int, int]]  # (x1, y1, x2, y2)
    regions: List[Tuple[int, int, int, int]]       # Merged padded detection regions
    motion_score: float                            # Fraction of moving pixels


class MotionDetector:
    """Fast motion detector identifying active regions to gate heavy CNN detection."""

    def __init__(
        self,
        motion_threshold: int = 25,
        min_motion_area: int = 150,
        scale: float = 0.5,
        blur_kernel: Tuple[int, int] = (15, 15),
        dilation_iterations: int = 2,
        padding_ratio: float = 0.20,
        make_square: bool = True,
        max_regions: int = 4,
    ):
        self.motion_threshold = motion_threshold
        self.min_motion_area = min_motion_area
        self.scale = scale
        self.blur_kernel = blur_kernel
        self.dilation_iterations = dilation_iterations
        self.padding_ratio = padding_ratio
        self.make_square = make_square
        self.max_regions = max_regions

        self._prev_gray: Optional[np.ndarray] = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def reset(self) -> None:
        self._prev_gray = None

    def detect(self, frame: np.ndarray) -> MotionResult:
        """Processes frame and returns motion status, raw motion boxes, and merged detection regions."""
        if frame is None or frame.size == 0:
            return MotionResult(has_motion=False, motion_boxes=[], regions=[], motion_score=0.0)

        h, w = frame.shape[:2]

        # 1. Downscale frame for ultra-fast motion estimation
        if self.scale != 1.0:
            sw, sh = max(1, int(w * self.scale)), max(1, int(h * self.scale))
            small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        else:
            small = frame
            sw, sh = w, h

        # 2. Convert to grayscale and blur to remove sensor noise
        if len(small.shape) == 3:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        else:
            gray = small

        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        # First frame initialization
        if self._prev_gray is None or self._prev_gray.shape != blurred.shape:
            self._prev_gray = blurred
            return MotionResult(has_motion=False, motion_boxes=[], regions=[], motion_score=0.0)

        # 3. Absolute frame difference
        frame_diff = cv2.absdiff(self._prev_gray, blurred)
        self._prev_gray = blurred

        # 4. Threshold difference
        _, thresh = cv2.threshold(frame_diff, self.motion_threshold, 255, cv2.THRESH_BINARY)

        # 5. Dilate to fill gaps between moving pixels
        dilated = cv2.dilate(thresh, self._kernel, iterations=self.dilation_iterations)

        # 6. Find contours of moving components
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        inv_scale = 1.0 / self.scale
        scaled_min_area = self.min_motion_area * (self.scale ** 2)

        motion_boxes: List[Tuple[int, int, int, int]] = []
        moving_pixels = int(cv2.countNonZero(thresh))
        motion_score = moving_pixels / float(sw * sh)

        for c in contours:
            area = cv2.contourArea(c)
            if area < scaled_min_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(c)
            # Map back to full resolution
            x1 = max(0, int(bx * inv_scale))
            y1 = max(0, int(by * inv_scale))
            x2 = min(w, int((bx + bw) * inv_scale))
            y2 = min(h, int((by + bh) * inv_scale))

            if x2 > x1 and y2 > y1:
                motion_boxes.append((x1, y1, x2, y2))

        has_motion = len(motion_boxes) > 0

        # 7. Merge and pad candidate regions
        regions = self._create_detection_regions(motion_boxes, w, h)

        return MotionResult(
            has_motion=has_motion,
            motion_boxes=motion_boxes,
            regions=regions,
            motion_score=round(motion_score, 4)
        )

    def _create_detection_regions(
        self,
        boxes: List[Tuple[int, int, int, int]],
        frame_w: int,
        frame_h: int
    ) -> List[Tuple[int, int, int, int]]:
        """Merges overlapping/proximate boxes and pads them into detection regions."""
        if not boxes:
            return []

        merged = list(boxes)
        # Cluster/merge boxes that are within 50px of each other
        merged = self._merge_overlapping_boxes(merged, distance_margin=40)

        # Pad and optionally square each region
        regions = []
        for b in merged[:self.max_regions]:
            x1, y1, x2, y2 = b
            bw, bh = x2 - x1, y2 - y1

            # Context padding
            pad_x = int(bw * self.padding_ratio)
            pad_y = int(bh * self.padding_ratio)
            rx1 = max(0, x1 - pad_x)
            ry1 = max(0, y1 - pad_y)
            rx2 = min(frame_w, x2 + pad_x)
            ry2 = min(frame_h, y2 + pad_y)

            if self.make_square:
                rw, rh = rx2 - rx1, ry2 - ry1
                side = max(rw, rh)
                cx, cy = (rx1 + rx2) // 2, (ry1 + ry2) // 2
                rx1 = max(0, cx - side // 2)
                ry1 = max(0, cy - side // 2)
                rx2 = min(frame_w, rx1 + side)
                ry2 = min(frame_h, ry1 + side)

            regions.append((rx1, ry1, rx2, ry2))

        return regions

    def _merge_overlapping_boxes(
        self,
        boxes: List[Tuple[int, int, int, int]],
        distance_margin: int = 40
    ) -> List[Tuple[int, int, int, int]]:
        """Iteratively merges boxes that overlap or are within distance_margin of each other."""
        if len(boxes) <= 1:
            return boxes

        merged = []
        used = [False] * len(boxes)

        for i in range(len(boxes)):
            if used[i]:
                continue
            cur_box = list(boxes[i])
            used[i] = True

            changed = True
            while changed:
                changed = False
                for j in range(len(boxes)):
                    if not used[j]:
                        b = boxes[j]
                        # Check proximity
                        if not (cur_box[2] + distance_margin < b[0] or
                                b[2] + distance_margin < cur_box[0] or
                                cur_box[3] + distance_margin < b[1] or
                                b[3] + distance_margin < cur_box[1]):
                            # Merge
                            cur_box[0] = min(cur_box[0], b[0])
                            cur_box[1] = min(cur_box[1], b[1])
                            cur_box[2] = max(cur_box[2], b[2])
                            cur_box[3] = max(cur_box[3], b[3])
                            used[j] = True
                            changed = True

            merged.append((cur_box[0], cur_box[1], cur_box[2], cur_box[3]))

        return merged
