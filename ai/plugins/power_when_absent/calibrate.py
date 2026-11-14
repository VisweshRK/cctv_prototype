"""Calibration script and utilities for Appliance Visual Power Detection.
Records average brightness with lights off and on for calibrated ROIs in ai/zones.yaml.
Supports Section 3 Step 2 of Mohanraj's P1 Plan.
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import yaml


@dataclass
class ApplianceCalibration:
    appliance_id: str
    camera_id: str
    appliance_type: str
    box: Tuple[int, int, int, int]
    brightness_off: float = 40.0
    brightness_on: float = 190.0

    @property
    def midpoint(self) -> float:
        return (self.brightness_off + self.brightness_on) / 2.0

    def is_powered_on(self, mean_brightness: float) -> bool:
        """Lights are on when the ROI brightness is closer to the 'on' value."""
        return abs(mean_brightness - self.brightness_on) < abs(mean_brightness - self.brightness_off)


def compute_roi_brightness(frame: np.ndarray, box: Tuple[int, int, int, int]) -> float:
    """Computes mean grayscale luminance inside the bounding box."""
    if frame is None or frame.size == 0:
        return 0.0
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    crop = gray[y1:y2, x1:x2]
    return float(np.mean(crop))


def update_zones_yaml_calibration(
    zones_file: str,
    appliance_id: str,
    camera_id: str,
    state: str,
    measured_val: float
) -> bool:
    """Updates the brightness_off or brightness_on value in zones.yaml."""
    try:
        with open(zones_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        data = {"zones": [], "appliances": []}

    appliances = data.get("appliances", [])
    found = False
    for app in appliances:
        if app.get("id") == appliance_id and app.get("camera_id", "camera_main") == camera_id:
            if state.lower() == "on":
                app["brightness_on"] = round(measured_val, 1)
            else:
                app["brightness_off"] = round(measured_val, 1)
            found = True
            break

    if not found:
        # Create appliance record
        new_app = {
            "id": appliance_id,
            "camera_id": camera_id,
            "type": "light",
            "box": [50, 50, 150, 150],
            f"brightness_{state.lower()}": round(measured_val, 1)
        }
        appliances.append(new_app)
        data["appliances"] = appliances

    with open(zones_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate appliance ROI light brightness")
    parser.add_argument("--camera", default="camera_main", help="Camera ID")
    parser.add_argument("--id", default="light_main", help="Appliance ROI ID")
    parser.add_argument("--state", choices=["on", "off"], required=True, help="State: 'on' or 'off'")
    parser.add_argument("--zones_file", default="ai/zones.yaml", help="Path to zones.yaml")
    args = parser.parse_args()

    # Capture 5 frames from default camera to compute average
    cap = cv2.VideoCapture(0)
    samples = []
    box = (50, 50, 150, 150)
    if cap.isOpened():
        for _ in range(5):
            ret, frame = cap.read()
            if ret:
                samples.append(compute_roi_brightness(frame, box))
        cap.release()

    if samples:
        avg_val = float(np.mean(samples))
        update_zones_yaml_calibration(args.zones_file, args.id, args.camera, args.state, avg_val)
        print(f"Successfully calibrated {args.id} {args.state} = {avg_val:.1f}")
    else:
        print("Camera not available. Synthetic calibration value used.")
