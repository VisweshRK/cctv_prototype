"""Face recognition: name known people, flag strangers (plan: docs/plans/p2-krithick.md).

Models: OpenCV Zoo YuNet (MIT) + SFace (Apache 2.0), fetched by download_models.py.
Gallery: faces/<name>/*.jpg -> data/faces.npz, built by build_gallery.py.

Today's worker calls on_frame() with the whole frame, so this plugin runs in a
demo mode: YuNet on the full frame, one pseudo-track ("visit") per face,
matched across frames by box overlap, ending visit_timeout seconds after the
face disappears. When Viswesh's pipeline
lands (TrackedObject + on_objects, see docs/plans/README.md), set
needs = {"objects"} and the real path below takes over: one vote history per
tracked person, one event per person track, exactly like Frigate.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cctv_ai.plugin import Event, Plugin as Base

from .engine import FaceEngine, iou
from .voting import FaceVote

log = logging.getLogger("cctv_ai.face_recognition")


class Plugin(Base):
    use_case = "face_recognition"
    enabled = True  # stays silent until the models and a gallery are present
    needs = {"video"}  # switch to {"objects"} when the shared tracker lands
    interval = 1.0
    heavy = True  # the pipeline should run this plugin on its own thread

    # tuning (see docs/plans/p2-krithick.md)
    detection_threshold = 0.7  # YuNet face score
    min_face_area = 750  # px; smaller faces are not recognisable
    crop_top_fraction = 0.5  # faces live in the top half of a person box
    unknown_score = 0.8
    recognition_threshold = 0.8  # Frigate uses 0.9; tune on our cameras
    min_faces = 3
    max_attempts = 12
    max_attempts_recognised = 6
    visit_timeout = 10.0  # demo mode: seconds without a face before a new "visit"
    duplicate_window = 120.0  # demo mode: no repeated event for the same name

    def setup(self, cameras: list[str]) -> None:
        root = Path(__file__).resolve().parents[3]
        self.engine = FaceEngine(root / "ai" / "models", self.detection_threshold)
        self.ready = self.engine.load_models()
        self.people_known = 0
        if self.ready:
            gallery = root / "data" / "faces.npz"
            if gallery.exists():
                try:
                    self.people_known = self.engine.load_gallery(gallery)
                    log.info("face gallery loaded: %d known people", self.people_known)
                except Exception as err:
                    log.warning("cannot load %s: %s", gallery, err)
            else:
                log.warning("no gallery at %s; everyone will be unknown "
                            "(run plugins.face_recognition.build_gallery)", gallery)
        self.votes: dict[str, FaceVote] = {}  # track id -> vote history
        self.snapshots: dict[str, str | None] = {}  # track id -> best face crop
        self.last_seen: dict[str, float] = {}  # demo visit id -> monotonic time
        self.boxes: dict[str, tuple] = {}  # demo visit id -> last face box (x, y, w, h)
        self.announced: dict[str, float] = {}  # demo camera:name -> time
        self.reported: set[str] = set()  # visits already announced
        self.box_fractions: dict[str, list[float]] = {}  # track id -> [x1, y1, x2, y2] 0..1
        self.visits = 0

    # ---- shared helpers -------------------------------------------------

    def _vote(self, track_id: str) -> FaceVote:
        if track_id not in self.votes:
            self.votes[track_id] = FaceVote(
                unknown_score=self.unknown_score,
                recognition_threshold=self.recognition_threshold,
                min_faces=self.min_faces,
                max_attempts=self.max_attempts,
                max_attempts_recognised=self.max_attempts_recognised,
            )
        return self.votes[track_id]

    def _save_snapshot(self, crop: np.ndarray, tag: str) -> str | None:
        if crop is None or crop.size == 0 or min(crop.shape[:2]) < 20:
            return None
        folder = Path(__file__).resolve().parents[3] / "data" / "snapshots" / self.use_case
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{tag}-{int(time.time() * 1000)}.jpg"
        cv2.imwrite(str(path), crop)
        return str(path)

    def _events_for(self, camera: str, track_id: str, ended: bool) -> list[Event]:
        """What this track reports now: one known or unknown event, or nothing."""
        if track_id in self.reported:  # already announced: never report twice
            return []
        vote = self.votes.get(track_id)
        if vote is None or not vote.attempts:
            return []
        name, score = vote.decision()
        if name is not None and vote.accepted == name:
            label, confidence = "known_face", score
        elif (ended or vote.exhausted()) and vote.accepted is None:
            # stranger left, or used up the attempts, without a match
            label = "unknown_face"
            confidence = max(s for _, s, _ in vote.attempts)
        else:
            return []

        if ":visit" in track_id and label == "known_face":  # demo mode only
            last = self.announced.get(f"{camera}:{name}")
            if last is not None and time.monotonic() - last < self.duplicate_window:
                return []
            self.announced[f"{camera}:{name}"] = time.monotonic()

        data: dict[str, Any] = {}
        if label == "known_face":
            data["name"] = name
        box = self.box_fractions.get(track_id)
        if box:
            data["box"] = box

        event = self.event(camera, label, min(1.0, max(0.0, confidence)),
                           snapshot=self.snapshots.get(track_id))
        if data:
            event.data = data
        log.info("%s: %s %.2f after %d attempts", track_id, label, confidence, len(vote.attempts))
        self.reported.add(track_id)
        if label == "known_face":
            self._forget(track_id)
        return [event]

    def _forget(self, track_id: str) -> None:
        """Drop every trace of a finished visit (called when it times out)."""
        self.votes.pop(track_id, None)
        self.snapshots.pop(track_id, None)
        self.boxes.pop(track_id, None)
        self.box_fractions.pop(track_id, None)
        self.reported.discard(track_id)

    def _record(self, camera: str, track_id: str, frame: np.ndarray, name: str | None,
                score: float, face) -> list[Event]:
        """Store one attempt and report the track if it is decided."""
        vote = self._vote(track_id)
        vote.add(name, score, face.area)
        if score > 0 and vote.best_area == face.area:
            self.snapshots[track_id] = self._save_snapshot(frame, name or "unknown")
        return self._events_for(camera, track_id, ended=False)

    # ---- demo path (whole frame, one pseudo-track per face) -------------

    def _match_visit(self, camera: str, box: tuple, now: float) -> str:
        """The active visit whose face overlaps this box most, or a new visit."""
        best_id: str | None = None
        best_iou = 0.3  # below this it is a different (or new) face
        for track_id, last in self.boxes.items():
            if (track_id.startswith(f"{camera}:visit")
                    and now - self.last_seen.get(track_id, 0) <= self.visit_timeout):
                overlap = iou(last, box)
                if overlap > best_iou:
                    best_id, best_iou = track_id, overlap
        if best_id is None:
            self.visits += 1
            best_id = f"{camera}:visit{self.visits}"
        return best_id

    def on_frame(self, camera: str, frame: np.ndarray, ts: datetime) -> list[Event]:
        if not self.ready or not self.people_known:
            return []
        now = time.monotonic()

        events: list[Event] = []
        for track_id in [t for t in self.boxes if t.startswith(f"{camera}:visit")
                         and now - self.last_seen.get(t, 0) > self.visit_timeout]:
            events += self._events_for(camera, track_id, ended=True)
            self._forget(track_id)  # the visit is over either way

        # every face gets its own visit, matched across frames by box overlap,
        # so two people in frame never mix their votes
        fh, fw = frame.shape[:2]
        for match in self.engine.recognise_all(frame, min_area=self.min_face_area):
            current = self._match_visit(camera, match.face.box, now)
            self.last_seen[current] = now
            self.boxes[current] = match.face.box
            x, y, bw, bh = match.face.box
            self.box_fractions[current] = [
                max(0.0, x / fw), max(0.0, y / fh),
                min(1.0, (x + bw) / fw), min(1.0, (y + bh) / fh),
            ]
            vote = self.votes.get(current)
            if vote is not None and not vote.should_attempt():
                continue  # already reported or out of attempts: keep the visit, stop voting
            if not match.usable:
                continue  # face is there but too blurry/off-angle: keep the visit, no attempt
            aligned = self.engine.align_crop(frame, match.face)
            events += self._record(camera, current, aligned, match.name,
                                   match.confidence, match.face)
        return events

    # ---- real path (tracked objects, once the shared pipeline lands) ----

    def on_objects(self, camera: str, frame: np.ndarray, ts: datetime,
                   objects: list, ended: list) -> list[Event]:
        """objects/ended are TrackedObject instances from cctv_ai.objects."""
        if not self.ready or not self.people_known:
            return []
        events: list[Event] = []

        for obj in ended:  # person left: report what we have
            key = f"{camera}:{obj.id}"
            if key in self.votes:
                events += self._events_for(camera, key, ended=True)
                self._forget(key)

        for obj in objects:
            if obj.label != "person" or not obj.true_positive:
                continue
            key = f"{camera}:{obj.id}"
            vote = self.votes.get(key)
            if vote is not None and not vote.should_attempt():
                continue
            x1, y1, x2, y2 = obj.box
            crop = frame[y1:y2, x1:x2]
            top = max(1, int(crop.shape[0] * self.crop_top_fraction))
            match = next(iter(self.engine.recognise_all(
                crop[:top], min_area=self.min_face_area)), None)
            if match is None or not match.usable:
                continue  # no usable face this frame: do not burn an attempt
            fh, fw = frame.shape[:2]
            fx, fy, fbw, fbh = match.face.box
            self.box_fractions[key] = [
                max(0.0, (x1 + fx) / fw), max(0.0, (y1 + fy) / fh),
                min(1.0, (x1 + fx + fbw) / fw), min(1.0, (y1 + fy + fbh) / fh),
            ]
            aligned = self.engine.align_crop(crop[:top], match.face)
            events += self._record(camera, key, aligned, match.name,
                                   match.confidence, match.face)
        return events
