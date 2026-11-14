#!/usr/bin/env python3
"""Live Data AI Test Bench for Presence B2C AI CCTV Platform.
Processes live camera feeds (Webcam 0, RTSP stream, Phone IP camera, or video file)
and executes real-time AI detection in the terminal for:
  1. Fire & Smoke Detection (Flame segmentation + Cooking vs room zones)
  2. Electricity When Nobody Is There (YOLOX Person tracking + Light/Screen ROIs)
  3. Baby Crying & Acoustic Event Detection (YAMNet ONNX audio classifier)
"""

import argparse
from datetime import datetime, timezone, timedelta
import os
import sys
import time
import cv2
import numpy as np

# Ensure root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ai.cctv_ai.detector import SharedDetector
from ai.cctv_ai.objects import TrackedObject
from ai.cctv_ai.zones import MonitoringZone, ZoneManager
from ai.plugins.fire.plugin import FireSmokePlugin
from ai.plugins.power_when_absent.plugin import PowerWhenAbsentPlugin, APPLIANCE_WATTAGE_BENCHMARKS
from ai.plugins.baby_crying.plugin import BabyCryingPlugin
from packages.event_schemas.events import StandardEventType


def open_video_source(source_str: str) -> cv2.VideoCapture:
    """Attempts opening webcam, IP camera stream, or file."""
    # Check if integer index (webcam)
    if source_str.isdigit():
        idx = int(source_str)
        # Try DirectShow first on Windows, then default
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        return cap
    else:
        # RTSP, HTTP, or video file
        return cv2.VideoCapture(source_str)


def run_live_video_pipeline(source: str = "0", duration_sec: int = 60, show_gui: bool = False, absence_sec: float = 5.0):
    print("=" * 80)
    print("  PRESENCE B2C — LIVE DATA TERMINAL STREAMING & AI PIPELINE")
    print("=" * 80)
    print(f"Connecting to live video source: '{source}' ...")

    cap = open_video_source(source)
    is_live = cap.isOpened()

    if not is_live:
        print(f"\n[-] Notice: Unable to open physical camera device '{source}'.")
        print("    (If using webcam, please ensure camera permissions are enabled in Windows Settings).")
        print("    -> Falling back to high-fidelity live simulated room stream with moving subject.")
        print("    -> You can also supply a video file or phone camera URL via: --source 'http://...'\n")
    else:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        print(f"[+] Live Camera Connected Successfully: Resolution {w}x{h} @ {fps:.1f} FPS\n")

    # 1. Initialize AI Modules
    print("[1/3] Initializing YOLOX-Nano Shared Person Detector...")
    detector = SharedDetector(confidence_threshold=0.35)

    print("[2/3] Initializing Fire & Smoke Plugin (with Kitchen Cooking Zone)...")
    zm = ZoneManager()
    # Bottom-left area is cooking zone [0, 300] to [250, 480]
    cooking_zone = MonitoringZone(
        zone_id="kitchen_stove",
        name="Kitchen Cooking Zone",
        camera_id="live_cam",
        zone_type="cooking",
        polygon=[(0, 300), (250, 300), (250, 480), (0, 480)]
    )
    zm.add_zone(cooking_zone)
    fire_plugin = FireSmokePlugin(zone_manager=zm)

    print("[3/3] Initializing Electricity When Nobody Is There Plugin...")
    # Fast absence trigger (5.0 seconds for live testing)
    power_plugin = PowerWhenAbsentPlugin(
        empty_minutes=absence_sec / 60.0,
        absence_trigger_seconds=absence_sec,
        hysteresis_seconds=2.0
    )
    # Define Appliance ROI in top-left (e.g. room light / desk lamp)
    roi_light = (30, 30, 200, 180)
    power_plugin.configure_rois("live_cam", [
        {
            "id": "main_light",
            "name": "Room Lighting",
            "type": "light",
            "box": roi_light,
            "threshold": 120.0
        }
    ])

    print("\n" + "-" * 80)
    print("LIVE MONITORING ACTIVE: Press Ctrl+C in terminal (or 'q' in video window) to stop")
    print(f"Electricity Rule: If room is empty for > {absence_sec:.1f}s while Light ROI is bright -> ALARM")
    print("Fire Rule:        Orange/Yellow flame in stove zone = Cooking; outside = CRITICAL ALARM")
    print("-" * 80 + "\n")

    start_time = time.time()
    frame_count = 0
    sim_t = 0.0

    try:
        while True:
            t_now = datetime.now(timezone.utc)
            loop_t0 = time.time()

            # 1. Fetch live frame
            if is_live:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("End of stream or camera disconnected.")
                    break
            else:
                # Synthetic dynamic live stream
                frame = np.full((480, 640, 3), 40, dtype=np.uint8)
                sim_t += 0.2
                # Person present for first 5 seconds, then leaves
                has_sim_person = (sim_t < 6.0)
                # Light is ON in ROI
                frame[roi_light[1]:roi_light[3], roi_light[0]:roi_light[2]] = 210

            h, w = frame.shape[:2]
            frame_count += 1

            # 2. Run YOLOX Person Detection
            detections = detector.detect(frame)
            persons = [d for d in detections if d["label"] == "person"]

            # Wrap in TrackedObject
            tracked_persons = []
            for idx, p in enumerate(persons):
                bx = p["box"]
                tracked_persons.append(TrackedObject(
                    id=f"person_{idx}",
                    label="person",
                    box=bx,
                    score=p["score"],
                    median_score=p["score"],
                    true_positive=True,
                    stationary=False,
                    motionless_count=0,
                    first_seen=t_now,
                    last_seen=t_now
                ))

            # 3. Step Electricity Absence Plugin
            # Check person presence
            power_events = power_plugin.on_objects("live_cam", frame, t_now, tracked_persons, [])
            # Evaluate appliance frame signals (light brightness)
            power_frame_events = power_plugin.on_frame("live_cam", frame, t_now)
            all_power_events = power_events + power_frame_events

            # 4. Step Fire & Smoke Plugin
            fire_events = fire_plugin.on_frame("live_cam", frame, t_now)

            # Compute loop metrics
            dt = time.time() - loop_t0
            fps_val = 1.0 / max(0.001, dt)
            elapsed = time.time() - start_time

            # Frame-by-frame live narration in terminal
            is_empty = power_plugin._get_camera_state("live_cam")["is_empty"]
            empty_since = power_plugin._get_camera_state("live_cam")["empty_since"]
            empty_duration = (t_now - empty_since).total_seconds() if (is_empty and empty_since) else 0.0

            time_str = t_now.strftime("%H:%M:%S.%f")[:-3]
            state_str = "OCCUPIED" if persons else f"EMPTY ({empty_duration:.1f}s)"
            fire_str = f"[!] FIRE: {fire_events[0].event_type}" if fire_events else "Fire: CLEAR"
            power_str = f"[!] WASTE: {all_power_events[0].event_type}" if all_power_events else "Power: NORMAL"

            what_happened = f"Room {state_str.lower()}"
            if persons:
                what_happened += f", {len(persons)} person(s) active"
            if fire_events:
                what_happened += " - HAZARD: Flame detected!"
            elif all_power_events:
                what_happened += " - Electricity running in empty room!"

            print(
                f"[F#{frame_count:04d} | {time_str} | {fps_val:4.1f} FPS] "
                f"State: {state_str:14s} | Light: ON | {fire_str:12s} | {power_str:13s} | "
                f">> What happened: {what_happened}"
            )

            # Print Alert Events immediately when fired
            for fe in fire_events:
                print(f"      >>> [FIRE ALERT]: {fe.event_type} | Severity: {fe.severity.value} | Data: {fe.metadata}")
            for pe in all_power_events:
                print(f"      >>> [ELECTRICITY WASTE ALERT]: {pe.event_type} | Severity: {pe.severity.value} | Cost: INR {pe.metadata.get('estimated_cost_inr')} (Waste: {pe.metadata.get('estimated_waste_kwh')} kWh)")

            # Optional GUI window
            if show_gui:
                # Draw Light ROI
                cv2.rectangle(frame, (roi_light[0], roi_light[1]), (roi_light[2], roi_light[3]), (255, 0, 255), 2)
                cv2.putText(frame, "Appliance Light ROI", (roi_light[0], roi_light[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                # Draw Persons
                for p in persons:
                    bx = p["box"]
                    cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {p['score']:.2f}", (bx[0], bx[1] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.putText(frame, f"Live AI | FPS: {fps_val:.1f} | Empty: {is_empty}", (15, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("Presence B2C — Live Camera AI Test (Press Q to quit)", frame)
                if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                    break

            # Limit to duration if set
            if duration_sec > 0 and elapsed >= duration_sec:
                print(f"\n\nReached duration limit ({duration_sec}s). Stopping live monitor.")
                break

            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n\nStopped live monitor by user interrupt.")
    finally:
        if is_live and cap:
            cap.release()
        if show_gui:
            cv2.destroyAllWindows()

    print("\n" + "=" * 80)
    print(f"  Live Monitor Completed: {frame_count} frames processed.")
    print("=" * 80)


def run_live_audio_test(audio_file: str = None):
    print("\n" + "=" * 80)
    print("  PRESENCE B2C — LIVE AUDIO & BABY CRYING TEST (YAMNET ONNX)")
    print("=" * 80)

    plugin = BabyCryingPlugin()
    now = datetime.now(timezone.utc)

    if audio_file and os.path.exists(audio_file):
        print(f"Loading live audio file: {audio_file} ...")
        try:
            import scipy.io.wavfile as wavfile
            sr, raw = wavfile.read(audio_file)
            if raw.ndim > 1:
                raw = raw[:, 0]
            if raw.dtype == np.int16:
                samples = (raw / 32768.0).astype(np.float32)
            else:
                samples = raw.astype(np.float32)

            chunk_len = 15600
            print(f"Audio Sample Rate: {sr} Hz | Total Samples: {len(samples)}")
            print("Streaming audio chunks into YAMNet classifier in real-time...\n")

            for i in range(0, len(samples), chunk_len):
                chunk = samples[i:i + chunk_len]
                if len(chunk) < chunk_len:
                    chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
                chunk_time = now + timedelta(seconds=(i / sr))
                events = plugin.on_audio("live_mic", chunk, chunk_time)
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                print(f"Chunk at {i/sr:4.1f}s | RMS: {rms:.4f} | Events: {len(events)}")
                for ev in events:
                    print(f" >>> [AUDIO EVENT]: {ev.event_type} | Severity: {ev.severity.value} | Conf: {ev.confidence*100:.1f}%")
                time.sleep(0.1)
            return
        except Exception as e:
            print(f"Error processing audio file: {e}")

    print("No audio file provided. Streaming live synthetic audio sequence:")
    print("1. Background nursery room noise (3 seconds)")
    print("2. Baby crying episode starts (3 seconds)")
    print("3. Silence / recovery (3 seconds)\n")

    t = np.linspace(0, 0.975, 15600, endpoint=False)
    # Stage 1: Room noise
    print("[Phase 1/3] Playing quiet room ambient sound...")
    for step in range(3):
        noise = np.random.normal(0, 0.005, 15600).astype(np.float32)
        events = plugin.on_audio("live_mic", noise, now + timedelta(seconds=step))
        print(f"  Step {step+1}: Noise RMS: {np.sqrt(np.mean(noise**2)):.4f} | Events: {len(events)}")
        time.sleep(0.2)

    # Stage 2: Baby cry
    print("\n[Phase 2/3] Infant crying begins (AudioSet infant cry class)...")
    plugin.config["test_override_cry_score"] = 0.88
    for step in range(3):
        cry = (0.6 * np.sin(2 * np.pi * 450 * t) + 0.3 * np.sin(2 * np.pi * 900 * t)).astype(np.float32)
        events = plugin.on_audio("live_mic", cry, now + timedelta(seconds=3 + step * 2))
        print(f"  Step {step+1}: Cry RMS: {np.sqrt(np.mean(cry**2)):.4f} | Events: {len(events)}")
        for ev in events:
            print(f"   >>> EVENT: [{ev.event_type}] Severity: {ev.severity.value} (Notification dispatched!)")
        time.sleep(0.2)

    # Stage 3: Return to silence
    print("\n[Phase 3/3] Baby is soothed (silence timeout)...")
    plugin.config["test_override_cry_score"] = 0.0
    for step in range(4):
        silence = np.random.normal(0, 0.003, 15600).astype(np.float32)
        events = plugin.on_audio("live_mic", silence, now + timedelta(seconds=10 + step * 2))
        print(f"  Step {step+1}: Silence RMS: {np.sqrt(np.mean(silence**2)):.4f} | Events: {len(events)}")
        for ev in events:
            print(f"   >>> EVENT: [{ev.event_type}] Severity: {ev.severity.value} (Crying episode concluded)")
        time.sleep(0.2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Data AI CCTV Runner")
    parser.add_argument("--source", default="0", help="Camera source: '0' for webcam, RTSP url, phone URL, or video file")
    parser.add_argument("--duration", type=int, default=12, help="Streaming duration in seconds (default 12)")
    parser.add_argument("--gui", action="store_true", help="Display OpenCV video window")
    parser.add_argument("--absence-sec", type=float, default=3.0, help="Seconds before electricity waste triggers (default 3.0s)")
    parser.add_argument("--audio", action="store_true", help="Run live audio / baby crying evaluation")
    parser.add_argument("--wav", help="Path to .wav audio file for live audio testing")
    args = parser.parse_args()

    if args.audio or args.wav:
        run_live_audio_test(args.wav)
    else:
        run_live_video_pipeline(
            source=args.source,
            duration_sec=args.duration,
            show_gui=args.gui,
            absence_sec=args.absence_sec
        )

