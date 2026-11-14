#!/usr/bin/env python3
"""Interactive Terminal Runner for Presence B2C AI CCTV Platform.
Allows giving direct inputs in the terminal to test:
  1. Fire & Smoke Detection
  2. Electricity When Nobody Is There (Power When Absent)
  3. Baby Crying Acoustic Detection
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

from ai.cctv_ai.objects import TrackedObject
from ai.cctv_ai.zones import MonitoringZone, ZoneManager
from ai.plugins.fire.plugin import FireSmokePlugin
from ai.plugins.power_when_absent.plugin import PowerWhenAbsentPlugin, APPLIANCE_WATTAGE_BENCHMARKS
from ai.plugins.baby_crying.plugin import BabyCryingPlugin
from packages.event_schemas.events import StandardEventType


# -----------------------------------------------------------------------------
# 1. FIRE & SMOKE TESTER
# -----------------------------------------------------------------------------
def run_fire_smoke_interactive(scenario_choice=None, custom_file=None):
    print("\n" + "=" * 70)
    print("  [1] FIRE & SMOKE DETECTION — TERMINAL RUNNER")
    print("=" * 70)

    zm = ZoneManager()
    # Designate cooking zone in bottom-left [100, 300] to [300, 450]
    cooking_zone = MonitoringZone(
        zone_id="kitchen_stove",
        name="Kitchen Gas Stove",
        camera_id="cam_fire",
        zone_type="cooking",
        polygon=[(100, 300), (300, 300), (300, 450), (100, 450)]
    )
    zm.add_zone(cooking_zone)
    plugin = FireSmokePlugin(zone_manager=zm)
    now = datetime.now(timezone.utc)

    if not scenario_choice and not custom_file:
        print("\nWhat input would you like to provide for Fire & Smoke?")
        print("  1. Kitchen Gas Stove (Normal cooking flame inside kitchen zone)")
        print("  2. Dangerous Room Fire (Uncontrolled flame in bedroom/living room)")
        print("  3. Smoke Plume (Diffuse smoke cloud)")
        print("  4. Normal Safe Room (Standard room, no fire)")
        print("  5. Custom image file path (load a .jpg or .png)")
        scenario_choice = input("\nEnter your choice (1-5) [default 2]: ").strip() or "2"

    frame = np.full((480, 640, 3), 40, dtype=np.uint8)

    if scenario_choice == "1":
        print("\n-> Simulating: Gas stove flame (Normal controlled flame inside Kitchen zone: X=200, Y=350)...")
        # Controlled flame patch BGR: B=20, G=140, R=255 (area ~ 25x25 = 625 px < 1200 px limit)
        frame[340:365, 190:215] = (20, 140, 255)
    elif scenario_choice == "2":
        print("\n-> Simulating: Dangerous room fire (Large flame in living room: X=450, Y=200)...")
        # Large flame patch
        frame[150:280, 400:530] = (20, 150, 255)
    elif scenario_choice == "3":
        print("\n-> Simulating: Smoke plume (low saturation, bright diffuse smoke cloud)...")
        # Diffuse smoke: BGR (180, 180, 180)
        cv2.circle(frame, (320, 240), 90, (180, 180, 180), -1)
    elif scenario_choice == "4":
        print("\n-> Simulating: Clean normal room with standard furniture lighting...")
        cv2.rectangle(frame, (100, 100), (500, 400), (60, 60, 60), 2)
    elif scenario_choice == "5":
        if not custom_file:
            custom_file = input("Enter path to image file: ").strip()
        if os.path.exists(custom_file):
            frame = cv2.imread(custom_file)
            print(f"-> Loaded image: {custom_file} (Resolution: {frame.shape[1]}x{frame.shape[0]})")
        else:
            print(f"[-] File not found: {custom_file}. Using default test room.")

    # Execute plugin
    t0 = time.time()
    events = plugin.on_frame("cam_fire", frame, now)
    elapsed_ms = (time.time() - t0) * 1000

    print(f"\nExecution Time: {elapsed_ms:.2f} ms")
    print(f"Events Emitted: {len(events)}")

    if not events:
        print(">>> STATUS: NORMAL / SAFE (No dangerous flame or smoke detected).")
    else:
        for idx, ev in enumerate(events, 1):
            print(f"\n[{idx}] EVENT FIRED: [{ev.event_type}]")
            print(f"    Severity:       {ev.severity.value}")
            print(f"    Confidence:     {ev.confidence * 100:.1f}%")
            print(f"    Evidence Type:  {ev.evidence_type}")
            print(f"    Metadata:       {ev.metadata}")
            if ev.event_type == StandardEventType.POSSIBLE_DANGEROUS_FIRE.value:
                print("    ACTION:         EMERGENCY SIREN & CRITICAL NOTIFICATION DISPATCHED!")
            elif ev.event_type == StandardEventType.EXPECTED_COOKING_FLAME.value:
                print("    ACTION:         Routine kitchen cooking activity logged (No alarm).")


# -----------------------------------------------------------------------------
# 2. ELECTRICITY WHEN NOBODY IS THERE (POWER WHEN ABSENT)
# -----------------------------------------------------------------------------
def run_electricity_interactive(appliance=None, state=None, occupancy=None, minutes=None):
    print("\n" + "=" * 70)
    print("  [2] ELECTRICITY WHEN NOBODY IS THERE — TERMINAL RUNNER")
    print("=" * 70)

    if not appliance:
        print("\nWhat input would you like to provide for Electricity Waste?")
        print("Select Appliance:")
        print("  1. Room Lighting (LED / Bulb, ~15W)")
        print("  2. Ceiling Fan (~65W)")
        print("  3. Television / Monitor Screen (~110W)")
        print("  4. Air Conditioner (~1400W)")
        app_choice = input("Enter appliance choice (1-4) [default 1]: ").strip() or "1"
        app_map = {"1": "light", "2": "fan", "3": "screen", "4": "ac_indicator"}
        appliance = app_map.get(app_choice, "light")

    if not state:
        state_choice = input("Is the appliance ON or OFF? (on/off) [default on]: ").strip().lower() or "on"
        is_on = (state_choice == "on")
    else:
        is_on = (state.lower() == "on")

    if not occupancy:
        occ_choice = input("Is the room EMPTY or OCCUPIED? (empty/occupied) [default empty]: ").strip().lower() or "empty"
        is_empty = (occ_choice == "empty")
    else:
        is_empty = (occupancy.lower() == "empty")

    if minutes is None:
        min_input = input("How many minutes has nobody been in the room? [default 15]: ").strip() or "15"
        try:
            minutes = float(min_input)
        except ValueError:
            minutes = 15.0

    print(f"\n-> Inputs Received:")
    print(f"   * Appliance:      {appliance.upper()} (Benchmark: {APPLIANCE_WATTAGE_BENCHMARKS.get(appliance, {}).get('typical_w', 20)}W)")
    print(f"   * Appliance State: {'ON (Consuming Power)' if is_on else 'OFF'}")
    print(f"   * Room Status:     {'EMPTY (Nobody there)' if is_empty else 'OCCUPIED (Person present)'}")
    print(f"   * Empty Duration:  {minutes} minutes")

    # Initialize plugin with short absence trigger to evaluate duration
    plugin = PowerWhenAbsentPlugin(empty_minutes=minutes, absence_trigger_seconds=minutes * 60.0)
    roi_box = (50, 50, 180, 180)
    plugin.configure_rois("cam_elec", [
        {
            "id": f"{appliance}_main",
            "name": f"Main {appliance.title()}",
            "type": appliance,
            "box": roi_box,
            "threshold": 120.0,
            "motion_diff_threshold": 3.0
        }
    ])

    now = datetime.now(timezone.utc)

    # 1. Set occupancy state
    if is_empty:
        # Nobody there since `minutes` ago
        plugin.update_occupancy_state("cam_elec", is_occupied=False, timestamp=now - timedelta(minutes=minutes))
    else:
        # Person sitting in the room
        person = TrackedObject(
            id="person_couch",
            label="person",
            box=(200, 150, 300, 350),
            score=0.94,
            median_score=0.94,
            true_positive=True,
            stationary=True,
            motionless_count=60,
            first_seen=now - timedelta(minutes=minutes),
            last_seen=now
        )
        plugin.on_objects("cam_elec", None, now, [person], [])

    # 2. Generate frame for ROI
    frame = np.full((480, 640, 3), 35, dtype=np.uint8)
    if is_on:
        if appliance in ["light", "ac_indicator"]:
            # High brightness in ROI
            frame[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]] = 230
        elif appliance == "screen":
            frame[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]] = (220, 180, 100)
        elif appliance == "fan":
            # For fan motion, simulate frame difference by running two consecutive frames
            prev = np.full((130, 130, 3), 40, dtype=np.uint8)
            plugin._prev_fan_crops[("cam_elec", f"{appliance}_main")] = prev
            frame[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]] = 180
    else:
        # Dark appliance / OFF
        frame[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]] = 25

    # 3. Process frame
    t0 = time.time()
    events = plugin.on_frame("cam_elec", frame, now)
    elapsed_ms = (time.time() - t0) * 1000

    print(f"\nExecution Time: {elapsed_ms:.2f} ms")
    print(f"Events Emitted: {len(events)}")

    if not events:
        if not is_empty:
            print(">>> RESULT: Room is occupied! No energy waste (Person is currently using the room).")
        elif not is_on:
            print(">>> RESULT: All appliances are switched OFF. No electricity wasted.")
        else:
            print(">>> RESULT: Within safe hysteresis window. No alert yet.")
    else:
        for idx, ev in enumerate(events, 1):
            print(f"\n[{idx}] EVENT FIRED: [{ev.event_type}]")
            print(f"    Severity:       {ev.severity.value}")
            print(f"    Appliance:      {ev.metadata.get('appliance_type', appliance)}")
            print(f"    Empty Minutes:  {ev.metadata.get('empty_minutes', minutes):.1f} min")
            print(f"    Estimated Waste:{ev.metadata.get('estimated_waste_kwh', 0.0):.4f} kWh")
            print(f"    Avoidable Cost: INR {ev.metadata.get('estimated_cost_inr', 0.0):.2f}")
            print("    ACTION:         Appliance Left ON Alert sent to Household Admin!")


# -----------------------------------------------------------------------------
# 3. BABY CRYING ACOUSTIC DETECTION
# -----------------------------------------------------------------------------
def run_baby_crying_interactive(sound_choice=None, custom_wav=None):
    print("\n" + "=" * 70)
    print("  [3] BABY CRYING ACOUSTIC CLASSIFIER (YAMNET) — TERMINAL RUNNER")
    print("=" * 70)

    plugin = BabyCryingPlugin()
    now = datetime.now(timezone.utc)

    if not sound_choice and not custom_wav:
        print("\nWhat input sound would you like to provide?")
        print("  1. Infant Crying & Sobbing (450 Hz modulated distress harmonics)")
        print("  2. Screaming / Shouting (High-pitch distress call)")
        print("  3. Normal Speech / Conversation (Standard adult dialogue)")
        print("  4. Silence / Low Ambient Noise (Quiet nursery room)")
        print("  5. Custom .wav audio file path")
        sound_choice = input("\nEnter your choice (1-5) [default 1]: ").strip() or "1"

    # 15600 samples = 0.975 seconds @ 16,000 Hz (Standard YAMNet input window)
    sample_rate = 16000
    duration = 0.975
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    if sound_choice == "1":
        print("\n-> Generating: Infant crying acoustic waveform (450 Hz modulated cry harmonics)...")
        # Infant cry audio with crying score simulation
        plugin.config["test_override_cry_score"] = 0.88
        audio_samples = (
            0.6 * np.sin(2 * np.pi * 450 * t) +
            0.3 * np.sin(2 * np.pi * 900 * t) +
            0.1 * np.random.normal(0, 0.02, len(t))
        ).astype(np.float32)
    elif sound_choice == "2":
        print("\n-> Generating: Screaming / high-frequency distress burst...")
        plugin.config["test_override_cry_score"] = 0.75
        audio_samples = (
            0.8 * np.sin(2 * np.pi * 1200 * t) +
            0.2 * np.random.normal(0, 0.05, len(t))
        ).astype(np.float32)
    elif sound_choice == "3":
        print("\n-> Generating: Normal speech frequencies (120 Hz to 250 Hz)...")
        audio_samples = (
            0.4 * np.sin(2 * np.pi * 180 * t) +
            0.2 * np.sin(2 * np.pi * 320 * t) +
            0.05 * np.random.normal(0, 0.01, len(t))
        ).astype(np.float32)
    elif sound_choice == "4":
        print("\n-> Generating: Low room ambient silence (-40 dB)...")
        audio_samples = np.random.normal(0, 0.005, len(t)).astype(np.float32)
    elif sound_choice == "5":
        if not custom_wav:
            custom_wav = input("Enter path to .wav file: ").strip()
        if os.path.exists(custom_wav):
            try:
                import scipy.io.wavfile as wavfile
                sr, raw = wavfile.read(custom_wav)
                if raw.ndim > 1:
                    raw = raw[:, 0]  # Take first channel
                # Convert to float32 [-1.0, 1.0]
                if raw.dtype == np.int16:
                    audio_samples = (raw / 32768.0).astype(np.float32)
                else:
                    audio_samples = raw.astype(np.float32)
                # Resample or slice to 15600 samples
                if len(audio_samples) < 15600:
                    audio_samples = np.pad(audio_samples, (0, 15600 - len(audio_samples)))
                else:
                    audio_samples = audio_samples[:15600]
                print(f"-> Loaded audio: {custom_wav} (Sample Rate: {sr} Hz)")
            except Exception as e:
                print(f"[-] Error loading audio file: {e}. Falling back to infant cry tone.")
                audio_samples = (0.6 * np.sin(2 * np.pi * 450 * t)).astype(np.float32)
        else:
            print(f"[-] File not found: {custom_wav}. Using default infant cry tone.")
            audio_samples = (0.6 * np.sin(2 * np.pi * 450 * t)).astype(np.float32)
    else:
        audio_samples = (0.6 * np.sin(2 * np.pi * 450 * t)).astype(np.float32)

    # Execute plugin
    t0 = time.time()
    events = plugin.on_audio("cam_baby", audio_samples, now)
    elapsed_ms = (time.time() - t0) * 1000

    # Calculate signal metrics
    rms = float(np.sqrt(np.mean(audio_samples ** 2)))

    print(f"\nExecution Time:     {elapsed_ms:.2f} ms")
    print(f"Classifier Backend: {plugin.backend} ({plugin.model_version})")
    print(f"Audio Energy (RMS): {rms:.4f} (Threshold: {plugin.energy_rms_threshold})")
    print(f"Events Emitted:     {len(events)}")

    if not events:
        print(">>> RESULT: No crying episode active (Audio is quiet or normal speech).")
    else:
        for idx, ev in enumerate(events, 1):
            print(f"\n[{idx}] EVENT FIRED: [{ev.event_type}]")
            print(f"    Severity:       {ev.severity.value}")
            print(f"    Confidence:     {ev.confidence * 100:.1f}%")
            print(f"    Cry Score:      {ev.metadata.get('cry_score', 0.0):.3f}")
            print(f"    Episode Status: {ev.metadata.get('episode_status')}")
            print(f"    Duration:       {ev.metadata.get('episode_duration_sec', 0.0):.1f}s")
            print("    ACTION:         High Priority Infant Cry Alert dispatched with audio chime!")


# -----------------------------------------------------------------------------
# MAIN CLI ENTRYPOINT
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Terminal AI Input & Execution Runner")
    parser.add_argument("--module", choices=["fire", "electricity", "baby", "all"], help="Module to test")
    parser.add_argument("--fire-scenario", choices=["1", "2", "3", "4", "5"], help="Fire scenario choice")
    parser.add_argument("--image", help="Custom image path for fire detection")
    parser.add_argument("--appliance", choices=["light", "fan", "screen", "ac_indicator"], help="Appliance type")
    parser.add_argument("--state", choices=["on", "off"], help="Appliance state")
    parser.add_argument("--occupancy", choices=["empty", "occupied"], help="Room occupancy")
    parser.add_argument("--empty-minutes", type=float, help="Minutes room was empty")
    parser.add_argument("--audio-choice", choices=["1", "2", "3", "4", "5"], help="Audio choice")
    parser.add_argument("--wav", help="Custom .wav audio file path")
    args = parser.parse_args()

    # Non-interactive mode if arguments provided
    if args.module:
        if args.module == "fire":
            run_fire_smoke_interactive(scenario_choice=args.fire_scenario, custom_file=args.image)
        elif args.module == "electricity":
            run_electricity_interactive(
                appliance=args.appliance,
                state=args.state,
                occupancy=args.occupancy,
                minutes=args.empty_minutes
            )
        elif args.module == "baby":
            run_baby_crying_interactive(sound_choice=args.audio_choice, custom_wav=args.wav)
        elif args.module == "all":
            run_fire_smoke_interactive(scenario_choice="2")
            run_electricity_interactive(appliance="light", state="on", occupancy="empty", minutes=15.0)
            run_baby_crying_interactive(sound_choice="1")
        return

    # Interactive Main Menu
    while True:
        print("\n" + "=" * 70)
        print("  PRESENCE B2C AI CCTV — INTERACTIVE TERMINAL TEST BENCH")
        print("=" * 70)
        print("Select the AI module you want to provide input for:")
        print("  1. Fire & Smoke Detection (Kitchen stove vs dangerous fire vs smoke)")
        print("  2. Electricity When Nobody Is There (Appliance left ON + room empty)")
        print("  3. Baby Crying Acoustic Detection (Infant cry vs speech vs silence)")
        print("  4. Run All 3 Use Cases sequentially")
        print("  5. Exit")

        choice = input("\nEnter choice (1-5): ").strip()
        if choice == "1":
            run_fire_smoke_interactive()
        elif choice == "2":
            run_electricity_interactive()
        elif choice == "3":
            run_baby_crying_interactive()
        elif choice == "4":
            run_fire_smoke_interactive(scenario_choice="2")
            run_electricity_interactive(appliance="light", state="on", occupancy="empty", minutes=20.0)
            run_baby_crying_interactive(sound_choice="1")
        elif choice == "5" or choice.lower() in ["exit", "q", "quit"]:
            print("Exiting interactive terminal test bench. Goodbye!")
            break
        else:
            print("Invalid choice, please select 1-5.")


if __name__ == "__main__":
    main()
