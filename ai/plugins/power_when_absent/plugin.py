"""Use Case 7 — Electricity Consumption During Room Absence Analytics for Presence B2C.
Implements Plan P1 Mohanraj (Electricity used when nobody is there).

Rule: power is on AND no person for N minutes -> one event "power_wasted".
Architecture:
  on_objects -> people in the room (tracked persons, including stationary ones)
  on_frame   -> "power on?" signals: lights on, screen on, fan moving (per-camera ROIs)
             -> timer: empty AND power on for N min -> event "power_wasted" -> re-arm when someone enters or turns off.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set
import cv2
import numpy as np
import yaml

from ai.cctv_ai.objects import TrackedObject
from ai.cctv_ai.plugin import BasePlugin
from ai.cctv_ai.zones import ZoneManager
from packages.event_schemas.events import StandardAIEvent, StandardEventType, EventSeverity


# Versioned typical appliance reference ranges (Watts)
APPLIANCE_WATTAGE_BENCHMARKS = {
    "light": {"min_w": 9.0, "typical_w": 15.0, "max_w": 28.0, "name": "Room Lighting"},
    "fan": {"min_w": 50.0, "typical_w": 65.0, "max_w": 80.0, "name": "Ceiling Fan"},
    "tv": {"min_w": 70.0, "typical_w": 110.0, "max_w": 160.0, "name": "Television Display"},
    "screen": {"min_w": 70.0, "typical_w": 110.0, "max_w": 160.0, "name": "Monitor / Screen Display"},
    "ac_indicator": {"min_w": 900.0, "typical_w": 1400.0, "max_w": 2000.0, "name": "Air Conditioner"},
}


class PowerWhenAbsentPlugin(BasePlugin):
    """Plugin for visual appliance detection and avoidable electricity estimates during room absence."""

    def __init__(
        self,
        empty_minutes: float = 10.0,
        absence_trigger_seconds: Optional[float] = None,
        hysteresis_seconds: float = 60.0,
        check_ir_night_mode: bool = False,
        ir_saturation_threshold: float = 15.0,
        assumed_tariff_inr_per_kwh: float = 8.0,
        zone_manager: Optional[ZoneManager] = None,
        zones_file: Optional[str] = "ai/zones.yaml",
    ):
        super().__init__(plugin_id="power_when_absent", name="Electricity During Absence Analytics", enabled=True)
        # Support both empty_minutes and absence_trigger_seconds
        if absence_trigger_seconds is not None:
            self.empty_minutes = absence_trigger_seconds / 60.0
            self.absence_trigger_seconds = absence_trigger_seconds
        else:
            self.empty_minutes = empty_minutes
            self.absence_trigger_seconds = empty_minutes * 60.0

        self.hysteresis_seconds = hysteresis_seconds
        self.check_ir_night_mode = check_ir_night_mode
        self.ir_saturation_threshold = ir_saturation_threshold
        self.assumed_tariff = assumed_tariff_inr_per_kwh
        self.zone_manager = zone_manager
        self.zones_file = zones_file

        # Appliance ROIs per camera: camera_id -> list of {"id": str, "type": str, "box": (x1, y1, x2, y2), ...}
        self.appliance_rois: Dict[str, List[dict]] = {}

        # Tracking absence and alert state per camera
        self._camera_states: Dict[str, dict] = {}

        # Previous frame crop for fan ROI frame differencing: (camera_id, roi_id) -> np.ndarray
        self._prev_fan_crops: Dict[Tuple[str, str], np.ndarray] = {}

        # Auto-load ROIs from zones.yaml if available
        if self.zones_file:
            self._load_appliances_from_yaml(self.zones_file)

    def _get_camera_state(self, camera_id: str) -> dict:
        if camera_id not in self._camera_states:
            self._camera_states[camera_id] = {
                "last_person_seen": None,
                "is_empty": False,
                "empty_since": None,
                "alerted": False,
                "alerted_appliances": set(),
                "power_signals": {"lights": False, "screen": False, "fan": False},
                "is_ir_night_mode": False,
            }
        return self._camera_states[camera_id]

    def _load_appliances_from_yaml(self, filepath: str) -> None:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                for app in data.get("appliances", []):
                    cam = app.get("camera_id", "camera_main")
                    if cam not in self.appliance_rois:
                        self.appliance_rois[cam] = []
                    self.appliance_rois[cam].append({
                        "id": app.get("id", f"app_{len(self.appliance_rois[cam])}"),
                        "name": app.get("name", "Appliance"),
                        "type": app.get("type", "light"),
                        "box": tuple(app.get("box", [0, 0, 100, 100])),
                        "brightness_off": app.get("brightness_off"),
                        "brightness_on": app.get("brightness_on"),
                        "threshold": app.get("threshold", 140.0),
                        "motion_diff_threshold": app.get("motion_diff_threshold", 4.0),
                    })
        except (FileNotFoundError, yaml.YAMLError):
            pass

    def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().setup(config)
        if config:
            if "empty_minutes" in config:
                self.empty_minutes = float(config["empty_minutes"])
                self.absence_trigger_seconds = self.empty_minutes * 60.0
            if "absence_trigger_seconds" in config:
                self.absence_trigger_seconds = float(config["absence_trigger_seconds"])
                self.empty_minutes = self.absence_trigger_seconds / 60.0
            if "hysteresis_seconds" in config:
                self.hysteresis_seconds = float(config["hysteresis_seconds"])
            if "zones_file" in config:
                self._load_appliances_from_yaml(config["zones_file"])
            if "appliance_rois" in config:
                self.appliance_rois.update(config["appliance_rois"])

    def configure_rois(self, camera_id: str, rois: List[dict]) -> None:
        """Manually configure appliance ROIs for a specific camera."""
        self.appliance_rois[camera_id] = rois

    def _update_person_presence(self, camera_id: str, has_people: bool, timestamp: datetime) -> None:
        """Update occupancy state respecting 60s departure hysteresis and stationary tracking."""
        state = self._get_camera_state(camera_id)

        if has_people:
            # Person detected in the room (tracked or stationary)
            state["last_person_seen"] = timestamp
            state["is_empty"] = False
            state["empty_since"] = None
            # Re-arm when someone enters
            state["alerted"] = False
            state["alerted_appliances"].clear()
        else:
            # No people detected in this frame
            if state["last_person_seen"] is None:
                # No person ever seen in session, mark empty
                state["is_empty"] = True
                if state["empty_since"] is None:
                    state["empty_since"] = timestamp
            else:
                elapsed_since_seen = (timestamp - state["last_person_seen"]).total_seconds()
                if elapsed_since_seen >= self.hysteresis_seconds:
                    # Confirmed empty after hysteresis window
                    state["is_empty"] = True
                    if state["empty_since"] is None:
                        state["empty_since"] = timestamp
                else:
                    # Still within hysteresis window: someone leaving momentarily does NOT start timer
                    state["is_empty"] = False
                    state["empty_since"] = None

    def update_occupancy_state(self, camera_id: str, is_occupied: bool, timestamp: datetime) -> None:
        """External hook for occupancy status (e.g. from pipeline or tests)."""
        state = self._get_camera_state(camera_id)
        if not is_occupied:
            # When externally set to unoccupied with a timestamp, respect empty duration directly
            state["is_empty"] = True
            state["empty_since"] = timestamp
            state["last_person_seen"] = None
        else:
            self._update_person_presence(camera_id, True, timestamp)

    def on_objects(
        self,
        camera: str,
        frame: np.ndarray,
        timestamp: datetime,
        objects: List[TrackedObject],
        ended: List[TrackedObject]
    ) -> List[StandardAIEvent]:
        """Step 1: Check tracked people in the room using shared tracker.
        A person sitting still is still tracked (stationary), so room is NOT empty.
        Counts only true_positive persons. Uses zone helper if a room zone is defined.
        """
        if not self.enabled:
            return []

        # 1. Filter only true_positive persons
        persons = [obj for obj in objects if obj.label == "person" and obj.true_positive]

        # 2. Zone containment: if a room/occupancy zone is defined, only count persons inside it
        if self.zone_manager:
            room_zones = [
                z for z in self.zone_manager.get_camera_zones(camera)
                if z.zone_type in ("room", "occupancy", "indoor")
            ]
            if room_zones:
                persons = [p for p in persons if any(rz.contains_object(p) for rz in room_zones)]

        # Note: persons includes stationary people (p.stationary is True), keeping room occupied!
        has_people = len(persons) > 0
        self._update_person_presence(camera, has_people, timestamp)

        return []

    def _evaluate_appliances(
        self,
        camera: str,
        frame: np.ndarray
    ) -> Tuple[Dict[str, bool], List[dict], bool]:
        """Evaluates lights, screens, and fan movements across configured ROIs.
        Returns:
            (power_signals, active_rois, is_ir_night_mode)
        """
        power_signals = {"lights": False, "screen": False, "fan": False}
        active_rois = []

        if frame is None or frame.size == 0:
            return power_signals, active_rois, False

        # Step 2: Night mode detection via color saturation (e.g. Q-Tech camera IR mode)
        # When camera switches to IR, image turns grey (saturation ~ 0).
        # A grey IR image means room lights are off.
        is_ir_night_mode = False
        ir_enabled = self.check_ir_night_mode or self.config.get("check_ir_night_mode", False) or (self.config.get("camera_model", "").lower() == "q-tech")

        if ir_enabled:
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mean_saturation = float(np.mean(hsv[:, :, 1]))
                is_ir_night_mode = mean_saturation < self.ir_saturation_threshold
            else:
                is_ir_night_mode = True

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        rois = self.appliance_rois.get(camera, [])

        for r in rois:
            roi_id = r.get("id", "appliance")
            app_type = r.get("type", "light").lower()
            if app_type == "tv":
                app_type = "screen"

            box = r.get("box", (0, 0, 100, 100))
            x1, y1, x2, y2 = box
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = gray[y1:y2, x1:x2]
            mean_lum = float(np.mean(crop))
            var_lum = float(np.var(crop))

            is_active = False

            if app_type == "light":
                # If camera is in IR night mode, room lights are definitively OFF!
                if is_ir_night_mode:
                    is_active = False
                else:
                    # Calibrated comparison or threshold
                    b_on = r.get("brightness_on")
                    b_off = r.get("brightness_off")
                    if b_on is not None and b_off is not None:
                        is_active = abs(mean_lum - b_on) < abs(mean_lum - b_off)
                    else:
                        is_active = mean_lum > r.get("threshold", 140.0)

                if is_active:
                    power_signals["lights"] = True

            elif app_type == "screen":
                # Screen / TV: bright rectangle or active display variance
                b_on = r.get("brightness_on")
                b_off = r.get("brightness_off")
                if b_on is not None and b_off is not None:
                    is_active = abs(mean_lum - b_on) < abs(mean_lum - b_off)
                else:
                    is_active = mean_lum > r.get("threshold", 90.0) or var_lum > 80.0

                if is_active:
                    power_signals["screen"] = True

            elif app_type == "fan":
                # Step 3: Constant motion inside fan ROI using frame differences inside ROI only
                prev_crop = self._prev_fan_crops.get((camera, roi_id))
                self._prev_fan_crops[(camera, roi_id)] = crop.copy()

                if prev_crop is not None and prev_crop.shape == crop.shape:
                    diff = cv2.absdiff(crop, prev_crop)
                    motion_score = float(np.mean(diff))
                    diff_thresh = r.get("motion_diff_threshold", 4.0)
                    is_active = motion_score > diff_thresh
                else:
                    is_active = False

                if is_active:
                    power_signals["fan"] = True

            if is_active:
                active_rois.append({
                    "id": roi_id,
                    "type": app_type,
                    "mean_lum": mean_lum,
                    "name": r.get("name", app_type),
                })

        return power_signals, active_rois, is_ir_night_mode

    def on_frame(
        self,
        camera: str,
        frame: np.ndarray,
        timestamp: datetime
    ) -> List[StandardAIEvent]:
        """Hook called for video frames to evaluate visual power and absence timer."""
        events: List[StandardAIEvent] = []
        if not self.enabled or frame is None or frame.size == 0:
            return events

        state = self._get_camera_state(camera)
        power_signals, active_rois, is_ir_night_mode = self._evaluate_appliances(camera, frame)
        state["power_signals"] = power_signals
        state["is_ir_night_mode"] = is_ir_night_mode

        any_power_on = any(power_signals.values())

        # Step 4: Re-arm when everything turns off
        if not any_power_on:
            if state["alerted"]:
                state["alerted"] = False
                state["alerted_appliances"].clear()
            return events

        # If room is not confirmed empty or timer hasn't started, no alarm
        if not state["is_empty"] or state["empty_since"] is None:
            return events

        # Measure empty duration
        absence_duration = (timestamp - state["empty_since"]).total_seconds()
        empty_minutes = absence_duration / 60.0

        if empty_minutes < self.empty_minutes:
            return events

        # Step 4: Empty AND power on for N minutes -> one event "power_wasted"
        if not state["alerted"]:
            state["alerted"] = True

            total_typical_w = sum(
                APPLIANCE_WATTAGE_BENCHMARKS.get(r["type"], {}).get("typical_w", 30.0)
                for r in active_rois
            )
            hours = absence_duration / 3600.0
            est_kwh = (total_typical_w * hours) / 1000.0
            est_cost = est_kwh * self.assumed_tariff

            # Main power_wasted event matching Plan P1 contract
            main_event = StandardAIEvent(
                household_id=self.config.get("household_id", "household_default"),
                camera_id=camera,
                module_id=self.plugin_id,
                event_type=StandardEventType.POTENTIAL_ENERGY_WASTE.value,
                timestamp=timestamp,
                confidence=0.88,
                evidence_type="scene_metric",
                severity=EventSeverity.MEDIUM,
                metadata={
                    "label": "power_wasted",
                    "data": {
                        "lights": power_signals["lights"],
                        "screen": power_signals["screen"],
                        "fan": power_signals["fan"],
                        "empty_minutes": round(empty_minutes, 1),
                    },
                    "lights": power_signals["lights"],
                    "screen": power_signals["screen"],
                    "fan": power_signals["fan"],
                    "empty_minutes": round(empty_minutes, 1),
                    "active_appliances": [r["id"] for r in active_rois],
                    "assumed_wattage_typical": total_typical_w,
                    "estimated_waste_kwh": round(est_kwh, 4),
                    "estimated_cost_inr": round(est_cost, 2),
                    "disclaimer": "Camera-only visual estimate. Not a direct electrical meter measurement.",
                },
                monitoring_uncertainty=0.15
            )
            events.append(main_event)

            # Specific appliance events for backward compatibility
            for r in active_rois:
                roi_id = r["id"]
                app_type = r["type"]
                if roi_id not in state["alerted_appliances"]:
                    state["alerted_appliances"].add(roi_id)
                    benchmark = APPLIANCE_WATTAGE_BENCHMARKS.get(
                        app_type,
                        {"min_w": 10.0, "typical_w": 50.0, "max_w": 100.0, "name": app_type}
                    )
                    roi_kwh = (benchmark["typical_w"] * hours) / 1000.0
                    roi_cost = roi_kwh * self.assumed_tariff

                    if app_type == "light":
                        event_type = StandardEventType.LIGHT_POSSIBLY_LEFT_ON.value
                    elif app_type == "fan":
                        event_type = StandardEventType.FAN_POSSIBLY_LEFT_ON.value
                    elif app_type in ("tv", "screen"):
                        event_type = StandardEventType.TV_POSSIBLY_LEFT_ON.value
                    else:
                        event_type = StandardEventType.POTENTIAL_ENERGY_WASTE.value

                    events.append(StandardAIEvent(
                        household_id=self.config.get("household_id", "household_default"),
                        camera_id=camera,
                        module_id=self.plugin_id,
                        event_type=event_type,
                        timestamp=timestamp,
                        confidence=0.85,
                        evidence_type="scene_metric",
                        severity=EventSeverity.MEDIUM,
                        metadata={
                            "label": "power_wasted",
                            "appliance_id": roi_id,
                            "appliance_type": app_type,
                            "appliance_name": benchmark.get("name", app_type),
                            "absence_duration_minutes": round(empty_minutes, 1),
                            "assumed_wattage_typical": benchmark["typical_w"],
                            "estimated_waste_kwh": round(roi_kwh, 4),
                            "estimated_cost_inr": round(roi_cost, 2),
                            "disclaimer": "Camera-only visual estimate. Not a direct electrical meter measurement.",
                        },
                        monitoring_uncertainty=0.20
                    ))

        return events

    def shutdown(self) -> None:
        self._prev_fan_crops.clear()
        self._camera_states.clear()
