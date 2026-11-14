"""Use Case 8 — Fire & Smoke Detection Plugin for Presence B2C.
Provides flame and smoke candidate detection, cooking-zone context awareness,
and temporal fire persistence tracking.
"""

from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from ai.cctv_ai.objects import TrackedObject
from ai.cctv_ai.plugin import BasePlugin
from ai.cctv_ai.zones import MonitoringZone, ZoneManager
import os
from packages.event_schemas.events import StandardAIEvent, StandardEventType, EventSeverity


class FireSmokePlugin(BasePlugin):
    """Plugin for visual flame and smoke detection."""

    def __init__(self, zone_manager: Optional[ZoneManager] = None, model_path: Optional[str] = None):
        super().__init__(plugin_id="fire", name="Fire & Smoke Detection", enabled=True)
        self.zone_manager = zone_manager or ZoneManager()
        self.flame_persistence: Dict[str, deque] = {}  # camera_id -> deque of recent flame detections

        # Optional ONNX deep-learning fire classification model
        default_paths = [
            model_path,
            "e:/cctv/ai/models/fire_detection.onnx",
            "ai/models/fire_detection.onnx",
            "e:/cctv/backend/models/fire_detection.onnx",
            "backend/models/fire_detection.onnx"
        ]
        self.model_path = next((os.path.abspath(p) for p in default_paths if p and os.path.exists(p)), None)
        self.session = None
        if self.model_path:
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
                self.input_name = self.session.get_inputs()[0].name
            except Exception:
                self.session = None

    def on_frame(
        self,
        camera: str,
        frame: np.ndarray,
        timestamp: datetime
    ) -> List[StandardAIEvent]:
        events: List[StandardAIEvent] = []
        if not self.enabled or frame is None or frame.size == 0:
            return events

        if camera not in self.flame_persistence:
            self.flame_persistence[camera] = deque(maxlen=10)

        # 1. Color and High-Luminance Flame Segmentation (HSV Space)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Flame color mask: low hue (red/orange/yellow), high saturation and value
        lower_flame = np.array([0, 120, 200], dtype=np.uint8)
        upper_flame = np.array([35, 255, 255], dtype=np.uint8)
        flame_mask = cv2.inRange(hsv, lower_flame, upper_flame)

        # 2. Smoke Segmentation: Low saturation, mid-to-high brightness with diffuse edges
        lower_smoke = np.array([0, 0, 140], dtype=np.uint8)
        upper_smoke = np.array([180, 40, 230], dtype=np.uint8)
        smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)

        # Find flame contours
        contours, _ = cv2.findContours(flame_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        flame_candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 100:  # Minimum pixel cluster
                x, y, w, h = cv2.boundingRect(c)
                flame_candidates.append({"box": (x, y, x + w, y + h), "area": area})

        self.flame_persistence[camera].append(len(flame_candidates) > 0)

        # Check cooking zones
        zones = self.zone_manager.get_camera_zones(camera)
        cooking_zones = [z for z in zones if z.zone_type == "cooking"]

        for cand in flame_candidates:
            box = cand["box"]
            cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)

            in_cooking_zone = False
            for cz in cooking_zones:
                if cz.contains_point((cx, cy)):
                    in_cooking_zone = True
                    break

            if in_cooking_zone:
                # Normal gas stove / expected flame
                if cand["area"] < 1200:
                    events.append(StandardAIEvent(
                        household_id=self.config.get("household_id", "household_default"),
                        camera_id=camera,
                        module_id=self.plugin_id,
                        event_type=StandardEventType.EXPECTED_COOKING_FLAME.value,
                        timestamp=timestamp,
                        confidence=0.85,
                        evidence_type="video_bbox",
                        severity=EventSeverity.INFO,
                        metadata={"box": box, "area": cand["area"], "in_kitchen_stove": True},
                        monitoring_uncertainty=0.15
                    ))
                else:
                    # Unusually large flame inside kitchen
                    events.append(StandardAIEvent(
                        household_id=self.config.get("household_id", "household_default"),
                        camera_id=camera,
                        module_id=self.plugin_id,
                        event_type=StandardEventType.POSSIBLE_DANGEROUS_FIRE.value,
                        timestamp=timestamp,
                        confidence=0.91,
                        evidence_type="video_bbox",
                        severity=EventSeverity.CRITICAL,
                        metadata={"box": box, "area": cand["area"], "escalation": "Large kitchen flare"},
                        monitoring_uncertainty=0.10
                    ))
            else:
                # Flame outside cooking zone -> Urgent dangerous fire alert
                events.append(StandardAIEvent(
                    household_id=self.config.get("household_id", "household_default"),
                    camera_id=camera,
                    module_id=self.plugin_id,
                    event_type=StandardEventType.POSSIBLE_DANGEROUS_FIRE.value,
                    timestamp=timestamp,
                    confidence=0.93,
                    evidence_type="video_bbox",
                    severity=EventSeverity.CRITICAL,
                    metadata={"box": box, "area": cand["area"], "location": "Living area or prohibited zone"},
                    monitoring_uncertainty=0.07
                ))

        # Check heavy smoke plume
        smoke_pixel_count = cv2.countNonZero(smoke_mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        smoke_ratio = smoke_pixel_count / max(1, total_pixels)

        if smoke_ratio > 0.35:
            events.append(StandardAIEvent(
                household_id=self.config.get("household_id", "household_default"),
                camera_id=camera,
                module_id=self.plugin_id,
                event_type=StandardEventType.SMOKE_CANDIDATE.value,
                timestamp=timestamp,
                confidence=0.87,
                evidence_type="scene_metric",
                severity=EventSeverity.HIGH,
                metadata={"smoke_coverage_pct": round(smoke_ratio * 100, 1)},
                monitoring_uncertainty=0.13
            ))

        return events
