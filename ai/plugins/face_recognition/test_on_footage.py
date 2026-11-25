"""Test the face plugin on footage without go2rtc: feed a video file (or a
folder of images) straight into on_frame() and print every event.

Run from ai/ (gallery must be built first):
    .venv/Scripts/python -m plugins.face_recognition.test_on_footage --video clip.mp4
    .venv/Scripts/python -m plugins.face_recognition.test_on_footage --images frames/

Faces are recognised with data/faces.npz (see build_gallery.py); snapshots go
to data/snapshots/face_recognition/. A stride is applied so you don't wait
for YuNet+SFace on all 30 frames of every second.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from plugins.face_recognition import Plugin

log = logging.getLogger("cctv_ai.face_recognition.footage")


def frames_from_video(path: Path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        yield frame
    capture.release()


def frames_from_images(folder: Path):
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            data = np.fromfile(str(path), np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is not None:
                yield frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="video file (mp4/avi/...)")
    source.add_argument("--images", type=Path, help="folder of jpg/png frames")
    parser.add_argument("--stride", type=int, default=10,
                        help="feed every Nth frame (default 10, ~3 fps on 30 fps video)")
    parser.add_argument("--camera", default="cam1")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    plugin = Plugin()
    plugin.setup([args.camera])
    if not plugin.ready:
        print("models missing: run plugins.face_recognition.download_models", file=sys.stderr)
        return 1
    if not plugin.people_known:
        print("no gallery: put photos in plugins/face_recognition/faces/<name>/ "
              "and run plugins.face_recognition.build_gallery", file=sys.stderr)
        return 1
    print(f"known people: {plugin.engine.names}")

    source_frames = (frames_from_video(args.video) if args.video
                     else frames_from_images(args.images))
    events_seen = 0
    fed = 0
    for index, frame in enumerate(source_frames):
        if index % args.stride:
            continue
        fed += 1
        for event in plugin.on_frame(args.camera, frame, datetime.now(timezone.utc)):
            events_seen += 1
            name = event.data.get("name", "")
            print(f"[{fed:4d}] {event.label:<15} {name:<15} "
                  f"conf {event.confidence:.2f}  snapshot={event.snapshot}")
    # end of footage: force-close every open visit so unknown people are reported
    for track_id in list(plugin.votes):
        for event in plugin._events_for(args.camera, track_id, ended=True):
            events_seen += 1
            name = event.data.get("name", "")
            print(f"[ end] {event.label:<15} {name:<15} "
                  f"conf {event.confidence:.2f}  snapshot={event.snapshot}")
    print(f"fed {fed} frames, {events_seen} event(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
