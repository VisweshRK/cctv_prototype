"""End-to-end check of face recognition + LPR on a synthetic clip, no go2rtc.

Builds a short 720p clip from the gallery photos (known people), a stranger
photo, and a rendered Indian plate, then runs it through:
  1. the face plugin's demo path (on_frame)  -> expect known_person events
  2. the LPR plugin's demo path (on_frame)   -> expect a vehicle event
  3. the test bench runner (process_video_frames) -> expect both decisions

Run from ai/:
    .venv/Scripts/python -m tests.integration_check

This is a manual acceptance check (models + gallery needed), not part of the
unittest run. When it passes on synthetic footage, feed real clips through
the test bench UI; the thresholds may still need tuning per camera.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from plugins.face_recognition import Plugin as FacePlugin
from plugins.vehicle_lpr import Plugin as LprPlugin

log = logging.getLogger("cctv_ai.integration")

W, H = 1280, 720
MODELS = Path(__file__).resolve().parents[1] / "models"
FACES_DIR = Path(__file__).resolve().parents[1] / "plugins" / "face_recognition" / "faces"
STRANGER = Path(__file__).resolve().parents[0] / "test_images" / "face_sample.jpg"


def cv2_imread(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def build_frames() -> list[np.ndarray]:
    """20 frames with two known people, then 20 frames adding a stranger and
    a plate on a car-colour rectangle."""
    people = sorted(p for p in FACES_DIR.iterdir() if p.is_dir())[:2]
    photos = []
    for person in people:
        photo = next((p for p in sorted(person.iterdir())
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"}), None)
        if photo is None:
            continue
        image = cv2_imread(photo)
        if image is not None:
            photos.append((person.name, image))
    stranger = cv2_imread(STRANGER)
    if len(photos) < 1 or stranger is None:
        raise SystemExit("need >= 1 gallery person and tests/test_images/face_sample.jpg")

    def paste(frame: np.ndarray, image: np.ndarray, x: int, y: int, height: int) -> None:
        scale = height / image.shape[0]
        resized = cv2.resize(image, (int(image.shape[1] * scale), height))
        h, w = resized.shape[:2]
        frame[y:y + h, x:x + w] = resized

    def paste_plate(frame: np.ndarray, x: int, y: int, text: str = "TN09AB1234") -> None:
        cv2.rectangle(frame, (x, y), (x + 360, y + 100), (240, 240, 240), -1)
        cv2.putText(frame, text, (x + 12, y + 72),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, (30, 30, 30), 3, cv2.LINE_AA)

    frames: list[np.ndarray] = []
    for i in range(20):  # scene 1: the known people walk past
        frame = np.full((H, W, 3), 60, np.uint8)
        name0, image0 = photos[0]
        paste(frame, image0, 150 + i * 4, 120, 320)
        if len(photos) > 1:
            paste(frame, photos[1][1], 700 - i * 3, 150, 300)
        frames.append(frame)
    for i in range(20):  # scene 2: stranger + plate drive through
        frame = np.full((H, W, 3), 60, np.uint8)
        paste(frame, stranger, 200 + i * 5, 100, 340)
        cv2.rectangle(frame, (600 + i * 8, 380), (1150 + i * 8, 640), (40, 40, 90), -1)
        paste_plate(frame, 640 + i * 8, 470)
        frames.append(frame)
    return frames


def run_plugins(frames: list[np.ndarray], stranger_after: int = 20) -> tuple[list, list]:
    """Run both plugins' demo paths over the clip.

    After frame `stranger_after` the face engine's gallery is swapped for
    random embeddings ("ghost" people): a real photo cannot serve as a
    stranger because SFace is mirror- and squeeze-invariant, so any face we
    can paste is in the gallery. The random gallery guarantees the same face
    reads as unknown, which exercises the unknown-person reporting path.
    """
    ts = datetime.now(timezone.utc)
    face = FacePlugin()
    face.setup(["cam1"])
    lpr = LprPlugin()
    lpr.setup(["cam1"])
    face_events: list = []
    plate_events: list = []
    rng = np.random.default_rng(42)
    for i, frame in enumerate(frames):
        if i == stranger_after:
            face.engine.names = ["ghost_a", "ghost_b"]
            face.engine.means = rng.normal(size=(2, 128)).astype(np.float32)
            face.engine.means /= np.linalg.norm(face.engine.means, axis=1, keepdims=True)
        face_events += [(i, e) for e in face.on_frame("cam1", frame, ts)]
        plate_events += [(i, e) for e in lpr.on_frame("cam1", frame, ts)]
    # end of footage: force-close the open face visits
    for track_id in list(face.votes):
        face_events += [(len(frames), e) for e in face._events_for("cam1", track_id, ended=True)]
    return face_events, plate_events


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    frames = build_frames()
    clip_path = Path(tempfile.gettempdir()) / "cctv_integration_clip.mp4"
    writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
    for frame in frames:
        writer.write(frame)
    writer.release()

    failures: list[str] = []

    # 1 + 2: the plugin paths the worker will use
    face_events, plate_events = run_plugins(frames)
    known = {e.data.get("name") for _, e in face_events if e.label == "known_person"}
    unknown = sum(1 for _, e in face_events if e.label == "unknown_person")
    plates = {e.data["plate"] for _, e in plate_events if e.label == "vehicle"}
    print(f"face events: known={sorted(known)} unknown_count={unknown}")
    print(f"plate events: {sorted(plates)}")
    if "krithick2" not in known:
        failures.append(f"no known_person event for krithick2 (got {sorted(known)})")
    if "TN09AB1234" not in plates:
        failures.append(f"plate TN09AB1234 not announced (got {sorted(plates)})")
    if unknown != 1:
        failures.append(f"expected exactly 1 unknown_person event, got {unknown}")
    if any(name and name.startswith("ghost") for name in known):
        failures.append("a ghost gallery name was accepted as known")

    # 3: the test bench runner path
    from test_bench.runner import VisionPipeline

    pipeline = VisionPipeline()
    summary = pipeline.process_video_frames(clip_path, stride=1, max_frames=40)
    vote = summary["face_vote_summary"]
    cluster = summary["plate_cluster_summary"]
    print(f"runner face vote: {vote['decision_name']} conf {vote['confidence']} "
          f"({vote['total_attempts']} attempts, {len(vote.get('tracks', []))} tracks)")
    print(f"runner plate cluster: {cluster['best_plate']} conf {cluster['confidence']} "
          f"({cluster['agreeing_reads']} agreeing reads)")
    if not (vote["decision_name"] and vote["confidence"] >= 0.8):
        failures.append(f"runner vote did not accept a known person: {vote['decision_name']}")
    if cluster["best_plate"] != "TN09AB1234":
        failures.append(f"runner plate cluster wrong: {cluster['best_plate']}")

    clip_path.unlink(missing_ok=True)
    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nintegration check PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
