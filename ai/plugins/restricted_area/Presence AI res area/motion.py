"""Cheap motion detector (inspired by Frigate's ``improved_motion.py``).

Uses OpenCV's background subtractor (MOG2) to find regions of change.
Only those regions are passed to the heavy person detector, saving a lot
of compute when the scene is static.
"""

from __future__ import annotations

import cv2
import numpy as np


class MotionDetector:
    """Lightweight frame-differencing motion detector.

    Parameters
    ----------
    threshold : int
        Foreground-mask binarisation threshold (0–255).
    min_area : int
        Minimum contour area in *downscaled* pixels to count as motion.
    history : int
        Number of frames the MOG2 background model remembers.
    downscale : float
        Factor to shrink the frame before analysis (0 < downscale <= 1).
        Smaller = faster but coarser boxes.
    """

    def __init__(
        self,
        threshold: int = 25,
        min_area: int = 500,
        history: int = 30,
        downscale: float = 0.25,
    ) -> None:
        self._threshold = threshold
        self._min_area = min_area
        self._downscale = downscale
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=threshold,
            detectShadows=False,
        )
        # Morphological kernel for closing small holes
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._frame_count = 0
        self._calibration_frames = history  # let bg model stabilise

    # ----------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return bounding boxes ``(x1, y1, x2, y2)`` of motion regions
        in full-frame coordinates. Returns an empty list if no
        significant motion is found.
        """
        h, w = frame.shape[:2]

        # 1) Downscale + grayscale
        small_w = max(1, int(w * self._downscale))
        small_h = max(1, int(h * self._downscale))
        small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # 2) Background subtraction
        fg_mask = self._bg.apply(gray)

        # Let the background model calibrate for a few frames
        self._frame_count += 1
        if self._frame_count <= self._calibration_frames:
            return []

        # 3) Threshold + morphological cleanup
        _, binary = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._kernel)

        # 4) Find contours -> bounding boxes
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        scale_x = w / small_w
        scale_y = h / small_h

        boxes: list[tuple[int, int, int, int]] = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self._min_area:
                continue
            sx, sy, sw, sh = cv2.boundingRect(cnt)
            # Scale back to full-frame coords
            x1 = int(sx * scale_x)
            y1 = int(sy * scale_y)
            x2 = int((sx + sw) * scale_x)
            y2 = int((sy + sh) * scale_y)
            boxes.append((x1, y1, x2, y2))

        return boxes
