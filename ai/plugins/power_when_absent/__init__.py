"""Camera-only estimate of appliances left on while a room is empty."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import yaml
from cctv_ai.plugin import Event, Plugin as Base

WATTS = {"light": 15.0, "fan": 65.0, "screen": 110.0, "tv": 110.0, "ac": 1400.0}
class Plugin(Base):
    use_case = "power_when_absent"
    enabled = True
    needs = {"video", "objects"}
    interval = 2.0
    def __init__(self):
        self.rois: dict[str, list[dict]] = {}
        self.state: dict[str, dict] = {}
        self.previous: dict[tuple[str, str], np.ndarray] = {}
        self.empty_minutes = 10.0
        self.hysteresis = 60.0
        self.tariff = 8.0
    def setup(self, cameras: list[str]) -> None:
        self.state = {}
        self.rois = {}
        for path in (Path("ai/zones.yaml"), Path("zones.yaml")):
            if path.exists():
                try:
                    raw = yaml.safe_load(path.read_text()) or {}
                    for cam, entries in raw.get("appliances", {}).items() if isinstance(raw.get("appliances"), dict) else []:
                        self.rois.setdefault(cam, []).extend(entries)
                    # Also accept a top-level list, as used by the source branch.
                    for entry in raw.get("appliances", []) if isinstance(raw.get("appliances"), list) else []:
                        self.rois.setdefault(entry.get("camera", entry.get("camera_id", "cam1")), []).append(entry)
                except Exception:
                    pass
    def configure_rois(self, camera: str, rois: list[dict]) -> None:
        self.rois[camera] = rois
    def _st(self, camera: str) -> dict:
        return self.state.setdefault(camera, {"last_person": None, "empty_since": None, "alerted": False})
    def on_objects(self, camera: str, frame: np.ndarray, ts: datetime, objects, ended) -> list[Event]:
        st = self._st(camera)
        people = [o for o in objects if o.label == "person" and o.true_positive]
        if people:
            st.update(last_person=ts, empty_since=None, alerted=False)
            return []
        if st["last_person"] is None:
            st["empty_since"] = st["empty_since"] or ts
        elif (ts - st["last_person"]).total_seconds() >= self.hysteresis:
            st["empty_since"] = st["empty_since"] or st["last_person"]
        return []
    def _active(self, camera: str, frame: np.ndarray) -> list[dict]:
        active = []
        if frame is None or frame.size == 0:
            return active
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        for roi in self.rois.get(camera, []):
            box = roi.get("box", [0, 0, 100, 100]); x1, y1, x2, y2 = map(int, box)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1: continue
            crop = gray[y1:y2, x1:x2]; typ = roi.get("type", "light").lower(); key = str(roi.get("id", typ))
            mean = float(crop.mean()); on = mean > float(roi.get("threshold", 140 if typ == "light" else 90))
            if typ == "fan":
                prev = self.previous.get((camera, key)); self.previous[(camera, key)] = crop.copy()
                on = prev is not None and prev.shape == crop.shape and float(cv2.absdiff(prev, crop).mean()) > float(roi.get("motion_diff_threshold", 4))
            if on: active.append({"id": key, "type": typ, "watts": WATTS.get(typ, 50.0)})
        return active
    def on_frame(self, camera: str, frame: np.ndarray, ts: datetime) -> list[Event]:
        st = self._st(camera); active = self._active(camera, frame)
        if not active or st["empty_since"] is None or st["alerted"]:
            if not active: st["alerted"] = False
            return []
        minutes = (ts - st["empty_since"]).total_seconds() / 60.0
        if minutes < self.empty_minutes: return []
        st["alerted"] = True
        watts = sum(a["watts"] for a in active); kwh = watts * minutes / 60000.0
        return [self.event(camera, "power_wasted", 0.88, time=ts, data={"area": "Room", "detail": f"Power on for {minutes:.1f} minutes with nobody present", "minutes": round(minutes, 1), "watts": round(watts, 1), "kwh": round(kwh, 4), "active_appliances": [a["id"] for a in active], "cost_inr": round(kwh * self.tariff, 2)})]
