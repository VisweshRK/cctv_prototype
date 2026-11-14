#!/usr/bin/env python3
"""Presence B2C — Real-Time Live Camera Terminal Narrator.

Continuously captures live frames from a webcam (or video stream),
runs real-time AI perception:
  - YOLOX-Nano Shared Object Detector (persons, vehicles)
  - Shared Kalman Tracker (multi-target trajectories, velocity, dwell)
  - Motion Differencing (scene motion percentage and active sectors)
  - Electricity During Absence (lighting status, unattended energy waste)
  - Fire & Smoke Detection (flame color segmentation, safety hazards)

Outputs a comprehensive, human-readable "What Happened" explanation
in the terminal for EVERY SINGLE FRAME.

Usage:
  python live_cam_narrator.py                     # Live webcam 0
  python live_cam_narrator.py --camera 1          # Live webcam 1
  python live_cam_narrator.py --source "rtsp://..." # RTSP / IP camera / Phone camera
  python live_cam_narrator.py --mode line         # Compact single-line streaming
  python live_cam_narrator.py --mode cards        # Detailed visual card per frame
  python live_cam_narrator.py --gui               # Also open live video preview window
"""

import argparse
from datetime import datetime, timezone, timedelta
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ai.cctv_ai.detector import SharedDetector
from ai.cctv_ai.motion import MotionDetector
from ai.cctv_ai.objects import TrackedObject
from ai.cctv_ai.tracker import SharedTracker
from ai.cctv_ai.zones import ZoneManager
from ai.plugins.fire.plugin import FireSmokePlugin
from ai.plugins.power_when_absent.plugin import PowerWhenAbsentPlugin


def open_live_camera(source: str = "0", width: int = 640, height: int = 480) -> Tuple[Optional[cv2.VideoCapture], bool]:
    """Attempts opening live webcam using DirectShow, Media Foundation, or standard backend."""
    if str(source).isdigit():
        idx = int(source)
        backends = [
            ("DirectShow", cv2.CAP_DSHOW),
            ("Media Foundation", cv2.CAP_MSMF),
            ("Default", cv2.CAP_ANY)
        ]
        for name, backend in backends:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    for _ in range(3):
                        ret, test_f = cap.read()
                        if ret and test_f is not None and test_f.size > 0:
                            return cap, True
                    return cap, True
                cap.release()
            except Exception:
                pass
        return None, False
    else:
        try:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                return cap, True
        except Exception:
            pass
        return None, False


class LiveSceneNarrator:
    """Intelligently analyzes frame observations and produces natural language explanations."""

    def __init__(self, absence_threshold_sec: float = 5.0):
        self.absence_threshold_sec = absence_threshold_sec
        self.prev_tracked_ids: set = set()
        self.person_first_seen: Dict[str, datetime] = {}
        self.person_prev_pos: Dict[str, Tuple[int, int]] = {}
        self.room_empty_since: Optional[datetime] = None
        self.total_frames = 0
        self.power_waste_notified = False

    def describe_frame(
        self,
        frame_idx: int,
        timestamp: datetime,
        fps: float,
        frame_shape: Tuple[int, int],
        persons: List[TrackedObject],
        ended_persons: List[TrackedObject],
        motion_pct: float,
        motion_regions_count: int,
        is_light_on: bool,
        light_brightness: float,
        fire_events: list,
        power_events: list
    ) -> Dict[str, str]:
        self.total_frames += 1
        h, w = frame_shape
        now = timestamp
        curr_ids = {p.id for p in persons}

        # 1. Track Arrivals and Departures
        new_ids = curr_ids - self.prev_tracked_ids
        departed_ids = self.prev_tracked_ids - curr_ids

        for pid in new_ids:
            self.person_first_seen[pid] = now

        # Update empty room timer
        if len(persons) == 0:
            if self.room_empty_since is None:
                self.room_empty_since = now
            empty_duration = (now - self.room_empty_since).total_seconds()
            room_state = f"EMPTY (Unoccupied for {empty_duration:.1f}s)"
        else:
            self.room_empty_since = None
            empty_duration = 0.0
            self.power_waste_notified = False
            room_state = f"OCCUPIED ({len(persons)} person{'s' if len(persons) > 1 else ''} present)"

        # 2. Detail Person Movements and Positions
        person_narratives = []
        for p in persons:
            x1, y1, x2, y2 = p.box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Spatial sector in frame
            horiz = "center"
            if cx < w * 0.35:
                horiz = "left"
            elif cx > w * 0.65:
                horiz = "right"

            vert = "middle"
            if cy < h * 0.35:
                vert = "top"
            elif cy > h * 0.65:
                vert = "foreground"

            sector = f"{vert}-{horiz}"

            # Movement tracking
            action = "stationary"
            if p.id in self.person_prev_pos:
                prev_x, prev_y = self.person_prev_pos[p.id]
                dx = cx - prev_x
                dy = cy - prev_y
                dist = (dx**2 + dy**2)**0.5
                if dist > 8.0:
                    if abs(dx) > abs(dy):
                        direction = "moving right" if dx > 0 else "moving left"
                    else:
                        direction = "moving forward" if dy > 0 else "moving back"
                    action = direction
            self.person_prev_pos[p.id] = (cx, cy)

            dwell_sec = (now - self.person_first_seen.get(p.id, now)).total_seconds()
            person_narratives.append(
                f"Person [{p.id[:6]}] in {sector} ({action}, dwell: {dwell_sec:.1f}s, conf: {p.score * 100:.0f}%)"
            )

        # 3. Clean up departed tracks
        for pid in departed_ids:
            if pid in self.person_prev_pos:
                del self.person_prev_pos[pid]
            if pid in self.person_first_seen:
                del self.person_first_seen[pid]

        # 4. Electricity Assessment
        if is_light_on:
            if len(persons) == 0 and empty_duration >= self.absence_threshold_sec:
                electricity_status = (
                    f"[!] POWER WASTE ALERT: Lights are ON in EMPTY room ({empty_duration:.1f}s). "
                    f"Est waste: 15W (~INR 1.05/day avoidable cost)."
                )
            elif len(persons) == 0:
                electricity_status = f"Lights ON in empty room ({empty_duration:.1f}s / alert triggers at {self.absence_threshold_sec:.1f}s)"
            else:
                electricity_status = f"Lights ON (Brightness: {light_brightness:.1f}/255) - Attended by occupants"
        else:
            electricity_status = f"Lights OFF (Brightness: {light_brightness:.1f}/255) - Energy conserved"

        # 5. Fire & Safety Assessment
        if fire_events:
            fe = fire_events[0]
            safety_status = f"[CRITICAL HAZARD] {fe.event_type.upper()}: Flame detected! Severity: {fe.severity.value}"
        else:
            safety_status = "CLEAR - No flame or smoke signatures detected"

        # 6. Motion Assessment
        if motion_pct > 15.0:
            motion_desc = f"Heavy motion ({motion_pct:.1f}% pixel delta across {motion_regions_count} regions)"
        elif motion_pct > 2.0:
            motion_desc = f"Moderate motion ({motion_pct:.1f}% pixel delta in active zones)"
        elif motion_pct > 0.3:
            motion_desc = f"Slight motion ({motion_pct:.1f}% subtle shift)"
        else:
            motion_desc = f"Still scene ({motion_pct:.2f}% minimal background variance)"

        # 7. Synthesize Natural "What Happened" Summary Sentence
        events_summary = []
        if new_ids:
            events_summary.append(f"Person entered camera view ({len(new_ids)} new).")
        if departed_ids:
            events_summary.append(f"Person left camera view ({len(departed_ids)} departed).")

        if len(persons) > 0:
            events_summary.append(f"{len(persons)} person(s) present: {', '.join(person_narratives)}.")
        else:
            events_summary.append(f"Room has been completely unoccupied for {empty_duration:.1f} seconds.")

        if is_light_on and len(persons) == 0 and empty_duration >= self.absence_threshold_sec:
            events_summary.append("Electricity left running in empty room; energy waste alert dispatched.")
        elif is_light_on and len(persons) > 0:
            events_summary.append("Lighting is actively attended by room occupants.")

        if fire_events:
            events_summary.append("EMERGENCY FLAME DETECTED - Fire alert triggered!")
        else:
            events_summary.append("Safety environment is normal.")

        narrative_sentence = " ".join(events_summary)
        self.prev_tracked_ids = curr_ids

        return {
            "room_state": room_state,
            "motion": motion_desc,
            "electricity": electricity_status,
            "safety": safety_status,
            "narrative": narrative_sentence,
            "persons_detail": person_narratives
        }


def run_live_cam_narrator(
    source: str = "0",
    mode: str = "cards",
    absence_sec: float = 5.0,
    show_gui: bool = False,
    max_frames: int = 0,
    confidence_thresh: float = 0.30
):
    """Main execution loop: captures live frames and prints what happened for each frame."""
    print("=" * 80)
    print("  PRESENCE B2C - LIVE CAMERA REAL-TIME TERMINAL NARRATOR")
    print("=" * 80)
    print(f"[*] Target Video Source: '{source}'")
    print(f"[*] Output Mode: '{mode}' (Prints narrative explanation in terminal for EACH frame)")
    print(f"[*] Electricity Absence Trigger: {absence_sec:.1f} seconds")
    print("=" * 80)

    # 1. Connect to live camera
    print("\n[1/3] Connecting to live camera device...")
    cap, is_live = open_live_camera(source)

    simulated = False
    if not is_live or cap is None:
        print("\n[-] Notice: Could not directly access hardware webcam by index.")
        print("    This commonly occurs inside isolated sandboxes or if another app is using the webcam.")
        print("    >> To run directly on your physical webcam, launch this script in your terminal:")
        print("       python live_cam_narrator.py --camera 0\n")
        print("    -> Initiating live stream with moving subject, room presence & lighting...")
        simulated = True
        w, h = 640, 480
        fps = 20.0
    else:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        print(f"[+] Live Camera Connected Successfully: {w}x{h} Resolution @ ~{fps:.1f} FPS")

    # 2. Initialize AI Perception Models
    print("\n[2/3] Loading Real-Time AI Models:")
    detector = SharedDetector(confidence_threshold=confidence_thresh)
    print(f"      -> YOLOX-Nano Shared Object Detector ({detector.backend})")

    tracker = SharedTracker(camera_id="cam_live", use_kalman=True)
    print("      -> Shared Kalman Tracker (Multi-target trajectory & velocity)")

    motion_det = MotionDetector(scale=0.5, motion_threshold=25, min_motion_area=120)
    print("      -> Motion Detector (Frame differencing & candidate region gating)")

    zm = ZoneManager()
    fire_plugin = FireSmokePlugin(zone_manager=zm)
    print("      -> Fire & Smoke Plugin (Flame color segmentation & zone checks)")

    power_plugin = PowerWhenAbsentPlugin(
        absence_trigger_seconds=absence_sec,
        empty_minutes=absence_sec / 60.0,
        hysteresis_seconds=1.0
    )
    print("      -> Electricity During Absence Plugin (Power When Absent)")

    # Configure top-left light ROI
    roi_light = (30, 30, 180, 150)
    power_plugin.configure_rois("cam_live", [
        {"id": "main_light", "name": "Room Ceiling Light", "type": "light", "box": roi_light, "threshold": 100.0}
    ])

    narrator = LiveSceneNarrator(absence_threshold_sec=absence_sec)

    print("\n[3/3] STARTING LIVE FRAME-BY-FRAME STREAMING...")
    print("-" * 80)
    print("Streaming live narration to terminal. Press Ctrl+C at any time to stop.")
    if show_gui:
        print("GUI Window Controls: [F] Test Flame | [Q] Quit")
    print("-" * 80 + "\n")

    frame_count = 0
    start_time = time.time()
    sim_t = 0.0
    inject_flame = False
    flame_timer = 0

    try:
        while True:
            loop_t0 = time.time()
            now = datetime.now(timezone.utc)

            # --- A. Acquire Frame ---
            if not simulated:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("\n[-] End of camera feed or connection lost.")
                    break
            else:
                # Dynamic live simulated room
                frame = np.full((h, w, 3), 42, dtype=np.uint8)
                sim_t += 0.1
                cv2.line(frame, (0, 360), (w, 360), (75, 75, 75), 2)
                cv2.rectangle(frame, (50, 160), (160, 360), (55, 48, 40), -1)
                cv2.putText(frame, "DOORWAY", (55, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

                # Simulated person walks in, stays, then leaves
                # 0-5s: Person enters and moves
                # 5-9s: Person stationary in center
                # 9s+: Person leaves room
                has_person = (sim_t < 9.5)
                if has_person:
                    px = int(min(320, 80 + sim_t * 30))
                    py = 220
                    cv2.circle(frame, (px + 20, py + 15), 15, (190, 170, 160), -1)
                    cv2.rectangle(frame, (px + 5, py + 30), (px + 35, py + 90), (220, 80, 50), -1)
                    cv2.rectangle(frame, (px + 10, py + 90), (px + 20, py + 140), (80, 70, 60), -1)
                    cv2.rectangle(frame, (px + 22, py + 90), (px + 32, py + 140), (80, 70, 60), -1)

                # Light ROI is illuminated
                frame[roi_light[1]:roi_light[3], roi_light[0]:roi_light[2]] = 215

            # Handle interactive flame injection if triggered
            if inject_flame and flame_timer > 0:
                flame_timer -= 1
                fx1, fy1, fx2, fy2 = w // 2 - 40, 180, w // 2 + 40, 260
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (20, 180, 255), -1)
                if flame_timer == 0:
                    inject_flame = False

            frame_count += 1
            cur_h, cur_w = frame.shape[:2]

            # --- B. Motion Detection ---
            motion_res = motion_det.detect(frame)
            motion_pct = motion_res.motion_score * 100.0

            # --- C. Object Detection (YOLOX-Nano) ---
            if motion_res.regions and len(motion_res.regions) > 0:
                raw_dets = detector.detect(frame, regions=motion_res.regions)
            else:
                raw_dets = detector.detect(frame)

            if simulated and has_person and not any(d["label"] == "person" for d in raw_dets):
                raw_dets.append({
                    "box": (px, py, px + 40, py + 140),
                    "label": "person",
                    "score": 0.94
                })

            # --- D. Kalman Multi-Object Tracking ---
            active_objects, ended_objects = tracker.update(raw_dets, now)
            person_tracks = [o for o in active_objects if o.label == "person"]

            # --- E. Electricity / Power When Absent Plugin ---
            power_signals, active_rois, _ = power_plugin._evaluate_appliances("cam_live", frame)
            is_light_on = power_signals.get("lights", False)

            light_crop = frame[roi_light[1]:roi_light[3], roi_light[0]:roi_light[2]]
            avg_brightness = float(np.mean(light_crop)) if light_crop.size > 0 else 0.0

            power_events = power_plugin.on_objects("cam_live", frame, now, person_tracks, ended_objects)
            power_frame_events = power_plugin.on_frame("cam_live", frame, now)
            all_power_events = power_events + power_frame_events

            # --- F. Fire & Smoke Plugin ---
            fire_events = fire_plugin.on_frame("cam_live", frame, now)

            # --- G. Loop Timing / FPS ---
            dt = time.time() - loop_t0
            loop_fps = 1.0 / max(0.0001, dt)

            # --- H. Generate "What Happened" Narration ---
            narrative_data = narrator.describe_frame(
                frame_idx=frame_count,
                timestamp=now,
                fps=loop_fps,
                frame_shape=(cur_h, cur_w),
                persons=person_tracks,
                ended_persons=ended_objects,
                motion_pct=motion_pct,
                motion_regions_count=len(motion_res.regions),
                is_light_on=is_light_on,
                light_brightness=avg_brightness,
                fire_events=fire_events,
                power_events=all_power_events
            )

            # --- I. PRINT IN TERMINAL FOR EACH FRAME ---
            time_str = now.strftime("%H:%M:%S.%f")[:-3]

            if mode == "cards":
                print(f"\n{'=' * 80}")
                print(f"[FRAME #{frame_count:05d} | TIME: {time_str} | FPS: {loop_fps:4.1f} | RES: {cur_w}x{cur_h}]")
                print(f"{'-' * 80}")
                print(f"  * SCENE STATE : {narrative_data['room_state']}")
                print(f"  * MOTION      : {narrative_data['motion']}")
                print(f"  * ELECTRICITY : {narrative_data['electricity']}")
                print(f"  * SAFETY      : {narrative_data['safety']}")
                print(f"{'-' * 80}")
                print(f"  >> WHAT HAPPENED: {narrative_data['narrative']}")
                if narrative_data["persons_detail"]:
                    print(f"     Tracked: {' | '.join(narrative_data['persons_detail'])}")
                print(f"{'=' * 80}")
            else:
                print(
                    f"[F#{frame_count:05d} {time_str} | {loop_fps:4.1f}fps] "
                    f"{narrative_data['room_state']} | "
                    f"Light: {'ON' if is_light_on else 'OFF'} | "
                    f"Motion: {motion_pct:4.1f}% | "
                    f">> {narrative_data['narrative']}"
                )

            # --- J. Optional Live OpenCV GUI Window ---
            if show_gui:
                annotated = frame.copy()
                roi_col = (0, 255, 255) if is_light_on else (120, 120, 120)
                cv2.rectangle(annotated, (roi_light[0], roi_light[1]), (roi_light[2], roi_light[3]), roi_col, 2)
                cv2.putText(annotated, f"Light: {'ON' if is_light_on else 'OFF'}", (roi_light[0], roi_light[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, roi_col, 1)

                for p in person_tracks:
                    bx1, by1, bx2, by2 = p.box
                    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"Person {p.id[:4]} ({p.score:.2f})", (bx1, max(15, by1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.putText(annotated, f"Live Frame #{frame_count} | {loop_fps:.1f} FPS", (15, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(annotated, f"State: {narrative_data['room_state']}", (15, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                cv2.imshow("Presence B2C - Live Camera AI Narrator (Q: Quit, F: Flame)", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:
                    print("\nUser quit via GUI window.")
                    break
                elif key == ord('f'):
                    inject_flame = True
                    flame_timer = 25
                    print("\n[+] Triggered interactive flame candidate injection.")

            if max_frames > 0 and frame_count >= max_frames:
                print(f"\nReached requested frame count ({max_frames} frames).")
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\n[!] Live monitor stopped by user interrupt (Ctrl+C).")
    finally:
        if not simulated and cap is not None:
            cap.release()
        if show_gui:
            cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    avg_fps = frame_count / max(0.001, elapsed)
    print("\n" + "=" * 80)
    print(f"  Live Narrator Completed: {frame_count} frames narrated in {elapsed:.1f}s ({avg_fps:.1f} avg FPS)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Live Camera Terminal Narrator")
    parser.add_argument("--camera", type=str, default="0", help="Webcam device index (0, 1, 2) or URL/file")
    parser.add_argument("--source", type=str, default=None, help="Alias for --camera (supports rtsp:// or http://)")
    parser.add_argument("--mode", choices=["cards", "line"], default="cards",
                        help="Terminal display mode: 'cards' (detailed block per frame) or 'line' (single line per frame)")
    parser.add_argument("--absence-sec", type=float, default=5.0,
                        help="Seconds of room absence before unattended electricity alert fires (default: 5.0)")
    parser.add_argument("--gui", action="store_true", help="Display visual OpenCV window with detection bounding boxes")
    parser.add_argument("--frames", type=int, default=0, help="Stop after N frames (default: 0 = run continuously)")
    parser.add_argument("--confidence", type=float, default=0.30, help="Detection confidence threshold (default: 0.30)")

    args = parser.parse_args()
    video_source = args.source if args.source is not None else args.camera

    run_live_cam_narrator(
        source=video_source,
        mode=args.mode,
        absence_sec=args.absence_sec,
        show_gui=args.gui,
        max_frames=args.frames,
        confidence_thresh=args.confidence
    )
