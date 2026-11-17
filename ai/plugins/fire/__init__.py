"""Fire and smoke detection using cheap, explainable visual signals.

This is intentionally model-optional: HSV flame candidates and diffuse smoke
coverage work without downloaded weights, while the event contract stays the
same as every other main plugin.
"""
from __future__ import annotations
from collections import deque
from datetime import datetime
import cv2
import numpy as np
from cctv_ai.plugin import Event, Plugin as Base
from cctv_ai.zones import load_zones_config, point_in_polygon

class Plugin(Base):
    use_case = "fire"
    enabled = True
    needs = {"video"}
    interval = 0.5
    persistence = 3
    def __init__(self):
        self.history: dict[str, deque[bool]] = {}
        self.zones: dict = {}
        self.last_event: dict[str, datetime] = {}
    def setup(self, cameras: list[str]) -> None:
        self.history = {c: deque(maxlen=self.persistence) for c in cameras}
        self.last_event = {}
        for path in ("zones.yaml", "ai/zones.yaml"):
            try:
                self.zones = load_zones_config(path)
                if self.zones:
                    break
            except Exception:
                pass
    def _cooking(self, camera: str, point: tuple[float, float], shape: tuple[int, int]) -> bool:
        h, w = shape
        for zone in self.zones.get(camera, {}).values():
            if zone.name.lower() not in {"cooking", "stove", "kitchen_stove"}:
                continue
            poly = np.array([[p[0] * w, p[1] * h] for p in zone.points], np.float32)
            if len(poly) and point_in_polygon(point, poly):
                return True
        return False
    def on_frame(self, camera: str, frame: np.ndarray, ts: datetime) -> list[Event]:
        if frame is None or frame.size == 0:
            return []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        flame = cv2.inRange(hsv, np.array([0, 110, 180], np.uint8), np.array([38, 255, 255], np.uint8))
        flame = cv2.morphologyEx(flame, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        smoke = cv2.inRange(hsv, np.array([0, 0, 115], np.uint8), np.array([180, 60, 235], np.uint8))
        contours, _ = cv2.findContours(flame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < max(80.0, frame.shape[0] * frame.shape[1] * 0.0003):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            candidates.append((x, y, x + w, y + h, area))
        smoke_ratio = float(cv2.countNonZero(smoke)) / float(frame.shape[0] * frame.shape[1])
        dangerous = bool(candidates) or smoke_ratio >= 0.30
        hist = self.history.setdefault(camera, deque(maxlen=self.persistence))
        hist.append(dangerous)
        if len(hist) < self.persistence or sum(hist) < self.persistence:
            return []
        last = self.last_event.get(camera)
        if last and (ts - last).total_seconds() < 10:
            return []
        x1, y1, x2, y2, area = max(candidates, key=lambda x: x[4]) if candidates else (0, 0, frame.shape[1], frame.shape[0], 0)
        box = [round(x1 / frame.shape[1], 3), round(y1 / frame.shape[0], 3), round(x2 / frame.shape[1], 3), round(y2 / frame.shape[0], 3)]
        in_cooking = bool(candidates) and self._cooking(camera, ((x1 + x2) / 2, (y1 + y2) / 2), frame.shape[:2])
        if in_cooking and area < frame.shape[0] * frame.shape[1] * 0.03 and smoke_ratio < 0.30:
            return [self.event(camera, "cooking_flame", 0.85, time=ts, data={"area": "Cooking zone", "detail": "Expected cooking flame", "box": box})]
        self.last_event[camera] = ts
        label = "fire" if candidates else "smoke"
        confidence = min(0.99, 0.65 + (0.25 if candidates else smoke_ratio * 0.5))
        detail = f"Flame candidate ({area:.0f} px)" if candidates else f"Smoke coverage {smoke_ratio:.0%}"
        return [self.event(camera, label, confidence, time=ts, data={"area": "Whole view", "detail": detail, "box": box, "smoke_coverage": round(smoke_ratio, 3)})]
