"""RapidOCR wrapper: text boxes + text + confidence from a BGR image.

RapidOCR (rapidocr-onnxruntime, Apache 2.0) bundles the PaddleOCR
det/cls/rec ONNX models, so there is nothing to download. The engine call is
heavy (hundreds of ms on a laptop CPU), which is why the plugin only runs it
on vehicle crops and throttles it.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger("cctv_ai.vehicle_lpr")


def enhance_contrast(bgr: np.ndarray, brightness_threshold: float = 90.0) -> np.ndarray:
    """CLAHE on the L channel for dark (night) crops, otherwise unchanged."""
    if bgr.mean() >= brightness_threshold:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(light), a, b]), cv2.COLOR_LAB2BGR)


def prepare_crop(crop: np.ndarray, min_height: int = 200, max_zoom: float = 2.0) -> np.ndarray:
    """Upscale a small vehicle crop so plate characters reach OCR-able size.

    The detector finds plate text reliably only above roughly 30 px per text
    line; on gate cameras the vehicle is often far away, so zoom in (capped:
    magnifying a blur creates no detail) with cubic interpolation.
    """
    if crop is None or crop.size == 0:
        return crop
    h = crop.shape[0]
    if h >= min_height:
        return crop
    zoom = min(max_zoom, min_height / max(1, h))
    if zoom <= 1.05:
        return crop
    return cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)


class PlateOCR:
    """Lazy RapidOCR engine: load once in setup(), never at import time."""

    def __init__(self) -> None:
        self.engine = None

    def load(self) -> bool:
        if self.engine is not None:
            return True
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            log.warning("rapidocr-onnxruntime not installed (see ai/requirements.txt)")
            return False
        try:
            # The package defaults to -1 (all host cores) for each of three
            # ONNX sessions. Cap each session so LPR cannot starve live video.
            self.engine = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
            return True
        except Exception:
            log.exception("cannot start RapidOCR")
            return False

    def detect_text(self, bgr: np.ndarray) -> list[tuple[tuple[float, float, float, float], str, float]]:
        """Text lines as ((x, y, w, h), text, confidence), best first."""
        if self.engine is None or bgr is None or bgr.size == 0:
            return []
        result = self.engine(bgr)
        if isinstance(result, tuple):  # (result, elapse) in rapidocr-onnxruntime
            result = result[0]
        lines: list[tuple[tuple[float, float, float, float], str, float]] = []
        for box, text, conf in result or []:
            if not text or not conf:
                continue
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            lines.append(((min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)),
                          str(text), float(conf)))
        lines.sort(key=lambda l: l[2], reverse=True)
        return lines
