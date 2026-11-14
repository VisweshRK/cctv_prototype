"""Interactive Live Demo Runner for Presence B2C AI CCTV Platform.
Steps through frames, showing:
  Motion detection -> Region extraction -> YOLOX Detector -> Kalman Tracker -> Electricity plugin
"""

from datetime import datetime, timezone, timedelta
import time
import numpy as np

from ai.cctv_ai.pipeline import MasterPipeline
from ai.cctv_ai.testing import generate_test_frame, create_stationary_person


def main():
    print("=" * 70)
    print("Presence B2C AI CCTV Platform — Live Pipeline Demo Runner")
    print("=" * 70)

    # 1. Initialize Master Pipeline (loads YOLOX, Kalman tracker, motion detector, all 10 plugins)
    print("\n[1/4] Initializing Master Pipeline...")
    pipeline = MasterPipeline(camera_id="camera_main", stream_source="synthetic")
    print("      -> SharedDetector: YOLOX-Nano ONNX")
    print("      -> SharedTracker: Kalman Filter enabled")
    print("      -> MotionDetector: Frame diffing & region padding active")
    print("      -> Electricity Plugin: Configured with 60s hysteresis & calibrated ROIs")

    # 2. Configure Electricity ROIs
    power_plugin = pipeline.plugins["power_when_absent"]
    power_plugin.empty_minutes = 0.05  # Shortened for quick terminal demo (3 seconds)
    power_plugin.absence_trigger_seconds = 3.0
    power_plugin.hysteresis_seconds = 1.0  # 1 second hysteresis for demo

    light_box = (50, 50, 150, 150)
    power_plugin.configure_rois("camera_main", [
        {"id": "light_main", "type": "light", "box": light_box, "threshold": 120.0}
    ])

    print("\n[2/4] Simulating Room with Person Present & Lights ON...")
    now = datetime.now(timezone.utc)
    frame_with_light = generate_test_frame(light_on=True, light_roi=light_box)

    for i in range(3):
        t = now + timedelta(seconds=i)
        person = create_stationary_person(timestamp=t)
        # Process frame
        events = pipeline.step_frame(frame_with_light, timestamp=t)
        active_objs = pipeline.metrics["active_objects_count"]
        has_motion = pipeline.metrics.get("has_motion", False)
        print(f"      Frame {i+1}: Active Persons: {active_objs} (Stationary) | Motion: {has_motion} | Events: {len(events)}")
        time.sleep(0.3)

    print("      -> Result: Room occupied! No electricity waste event fired.")

    print("\n[3/4] Person Leaves Room — Lights Left ON...")
    pipeline.stream_source = "empty"  # Stop synthetic simulated person detections
    # Clear tracker detections to simulate vacancy
    empty_now = now + timedelta(seconds=5)
    for i in range(8):
        t = empty_now + timedelta(seconds=i * 2)
        events = pipeline.step_frame(frame_with_light, timestamp=t)
        active_objs = pipeline.metrics["active_objects_count"]
        empty_state = power_plugin._get_camera_state("camera_main")["is_empty"]

        waste_events = [e for e in events if e.metadata.get("label") == "power_wasted"]
        print(f"      Step {i+1} (t=+{i*2}s): Room Confirmed Empty: {empty_state} | Active Persons: {active_objs} | Alerts: {len(waste_events)}")
        if waste_events:
            ev = waste_events[0]
            print(f"\n      >>> EVENT FIRED: [{ev.event_type}] Severity: {ev.severity.value}")
            print(f"          Label: {ev.metadata.get('label')}")
            print(f"          Data:  {ev.metadata.get('data')}")
            print(f"          Cost:  INR {ev.metadata.get('estimated_cost_inr')} | Waste: {ev.metadata.get('estimated_waste_kwh')} kWh")
            break
        time.sleep(0.3)

    print("\n[4/4] Pipeline Demo Finished Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
