"""draw_zones.py: Interactive zone polygon drawing and exclusion area calibration tool.
Allows users to visually draw and save:
  - Monitoring zones (e.g. 'room', 'door', 'corridor')
  - Furniture exclusion zones (e.g. 'bed', 'sofa') to suppress intentional lying down.

Usage:
    python -m cctv_ai.draw_zones cam1 --zone sofa --exclude --save
    python -m cctv_ai.draw_zones cam1 --zone bed --exclude --video path/to/clip.mp4 --save
    python -m cctv_ai.draw_zones cam1 --zone room --save

Controls:
    - Left Click: Add polygon vertex
    - Enter / Space: Complete polygon
    - 's': Save directly to ai/zones.yaml
    - 'c': Clear points
    - 'q' / Esc: Quit
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import cv2
import numpy as np
import yaml

from .client import Go2rtc


def save_zone_to_yaml(
    yaml_path: str,
    camera: str,
    zone_name: str,
    norm_points: list[list[float]],
    is_exclude: bool = False,
    objects: Optional[list[str]] = None,
) -> None:
    """Safely append or update a zone entry in zones.yaml."""
    data = {}
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as err:
            print(f"Warning: could not read existing {yaml_path}: {err}", file=sys.stderr)
            data = {}

    if camera not in data or not isinstance(data[camera], dict):
        data[camera] = {}

    zone_entry = {
        "points": [[float(f"{pt[0]:.3f}"), float(f"{pt[1]:.3f}")] for pt in norm_points],
        "objects": objects or ["person"],
        "inertia": 3,
        "loitering_time": 0,
    }
    if is_exclude:
        zone_entry["exclude"] = True

    data[camera][zone_name] = zone_entry

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=None)
    print(f"\n[SUCCESS] Successfully saved zone '{zone_name}' for {camera} to {yaml_path}!")


def run_draw_tool(
    camera: str,
    zone_name: str,
    url: str,
    video_path: Optional[str] = None,
    is_exclude: bool = False,
    auto_save: bool = False,
    yaml_path: Optional[str] = None,
) -> None:
    frame = None

    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            print(f"Warning: Could not read frame from video {video_path}", file=sys.stderr)
            frame = None

    if frame is None:
        client = Go2rtc(url)
        try:
            frame = client.frame(camera)
        except Exception:
            frame = None

    if frame is None:
        print(f"Note: Camera {camera} is not streaming. Opening a 1280x720 canvas.")
        frame = np.full((720, 1280, 3), 30, dtype=np.uint8)
        cv2.putText(
            frame,
            f"Canvas for {camera} ({zone_name}). Click points to calibrate.",
            (50, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 200),
            2,
        )

    h, w = frame.shape[:2]
    points_px: list[tuple[int, int]] = []
    clone = frame.copy()
    zone_type_label = "EXCLUSION ZONE (Bed/Sofa)" if is_exclude else "MONITORING ZONE"

    def redraw():
        nonlocal clone
        clone = frame.copy()
        # Draw header bar
        cv2.rectangle(clone, (0, 0), (w, 40), (20, 25, 35), -1)
        color = (0, 140, 255) if is_exclude else (0, 255, 120)
        cv2.putText(
            clone,
            f"Camera: {camera} | Zone: '{zone_name}' [{zone_type_label}] | Click points | ENTER=done, S=save, C=clear",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
        for i, pt in enumerate(points_px):
            cv2.circle(clone, pt, 5, color, -1)
            cv2.circle(clone, pt, 7, (255, 255, 255), 1)
            if i > 0:
                cv2.line(clone, points_px[i - 1], pt, color, 2)
        if len(points_px) >= 3:
            # Ghost line back to start
            cv2.line(clone, points_px[-1], points_px[0], (color[0] // 2, color[1] // 2, color[2] // 2), 1)

    win_name = f"Zone Drawer: {camera} - {zone_name}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, min(1280, w), min(720, h))

    def mouse_callback(event, x, y, flags, param):
        nonlocal points_px
        if event == cv2.EVENT_LBUTTONDOWN:
            points_px.append((x, y))
            redraw()
            cv2.imshow(win_name, clone)

    cv2.setMouseCallback(win_name, mouse_callback)
    redraw()
    cv2.imshow(win_name, clone)

    print(f"\n--- Zone Drawing Tool: {camera} / {zone_name} ---")
    print(f"Type: {zone_type_label}")
    print("Click at least 3 points to outline the area.")
    print("Press ENTER to finish, 's' to save directly to zones.yaml, 'c' to clear, 'q' to quit.\n")

    finished = False
    saved = False

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):  # Enter or Space
            if len(points_px) >= 3:
                finished = True
                redraw()
                cv2.line(clone, points_px[-1], points_px[0], (0, 255, 120) if not is_exclude else (0, 140, 255), 2)
                cv2.imshow(win_name, clone)
                cv2.waitKey(100)
                if auto_save:
                    break
                print("\nPolygon defined! Press 's' to save to zones.yaml, or close window.")
            else:
                print("Please click at least 3 points.")
        elif key in (ord("s"), ord("S")):
            if len(points_px) >= 3:
                finished = True
                norm_points = [[round(x / float(w), 3), round(y / float(h), 3)] for x, y in points_px]
                target_yaml = yaml_path or os.path.join(os.path.dirname(__file__), "..", "zones.yaml")
                save_zone_to_yaml(
                    yaml_path=os.path.abspath(target_yaml),
                    camera=camera,
                    zone_name=zone_name,
                    norm_points=norm_points,
                    is_exclude=is_exclude,
                )
                saved = True
                break
            else:
                print("Cannot save: click at least 3 points first.")
        elif key in (ord("c"), ord("C")):
            points_px.clear()
            redraw()
            cv2.imshow(win_name, clone)
            print("Cleared points.")
        elif key in (27, ord("q"), ord("Q")):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()

    if finished and not saved:
        norm_points = [[round(x / float(w), 3), round(y / float(h), 3)] for x, y in points_px]
        print(f"\nGenerated YAML block:")
        print(f"  {zone_name}:")
        print(f"    points:")
        for pt in norm_points:
            print(f"      - [{pt[0]:.3f}, {pt[1]:.3f}]")
        print(f"    objects: [person]")
        print(f"    inertia: 3")
        print(f"    loitering_time: 0")
        if is_exclude:
            print(f"    exclude: true")


def main():
    parser = argparse.ArgumentParser(description="Interactive zone polygon and exclusion zone drawing tool")
    parser.add_argument("camera", default="cam1", nargs="?", help="Camera name (e.g. cam1)")
    parser.add_argument("--zone", default="sofa", help="Zone name (e.g. sofa, bed, desk, room)")
    parser.add_argument("--exclude", action="store_true", help="Mark zone as furniture exclusion zone (bed/sofa)")
    parser.add_argument("--video", default=None, help="Optional video file to calibrate from")
    parser.add_argument("--url", default="http://localhost:1984", help="go2rtc URL")
    parser.add_argument("--save", action="store_true", help="Auto-save to zones.yaml on Enter")
    parser.add_argument("--yaml", default=None, help="Custom path to zones.yaml")
    args = parser.parse_args()

    # If zone name is bed or sofa, default to exclude=True unless specified
    is_ex = args.exclude or args.zone.lower() in ("bed", "sofa", "couch", "recliner")
    run_draw_tool(
        camera=args.camera,
        zone_name=args.zone,
        url=args.url,
        video_path=args.video,
        is_exclude=is_ex,
        auto_save=args.save,
        yaml_path=args.yaml,
    )


if __name__ == "__main__":
    main()
