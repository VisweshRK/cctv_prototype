"""Camera tampering (image side): covered, blurred / sprayed, moved / turned away.

Disconnect is done in Go (internal/cctv/status.go). See README.md for how it works
and what the settings mean.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np

from cctv_ai.plugin import Event, Plugin as Base

from .checks import Look, Matcher, Reference, look

log = logging.getLogger("cctv_ai.tampering")


@dataclass
class Camera:
    """What the plugin remembers about one camera."""

    learning: int  # checks left before the baseline is ready (no alerts until then)
    samples: deque = field(default_factory=lambda: deque(maxlen=300))  # (brightness, contrast, sharpness)
    reference: Optional[Reference] = None  # view to compare with for "moved"
    since_reference: int = 0
    colour: Optional[bool] = None  # False while the camera is in IR (grey) mode
    bad: int = 0  # tampered checks in a row
    good: int = 0  # normal checks in a row during an incident
    incident: Optional[str] = None  # label of the event already sent
    incident_since: Optional[datetime] = None


class Plugin(Base):
    use_case = "tampering"
    enabled = True
    needs = {"video"}
    interval = 1.0

    learn_checks = 60  # checks (1 s each) to learn what this camera normally looks like
    relearn_checks = 10  # after a day/night (colour/IR) switch
    persist = 3  # checks in a row before an event (and before "restored")
    dark = 15.0  # mean brightness below this = covered
    flat = 8.0  # contrast below this = one flat colour = covered
    plain = 12.0  # a normal view with less contrast than this is too plain for the flat and blur checks
    blur_ratio = 0.3  # sharpness below this share of normal = blurred
    min_features = 30  # reference views with fewer ORB features are not checked for "moved"
    min_inliers = 12  # fewer matching features than this (or than inlier_ratio) = turned away
    inlier_ratio = 0.15
    max_shift = 0.1  # the view's corners moved more than this share of the width = moved
    colour_on, colour_off = 16.0, 8.0  # mean saturation that means colour / IR mode (hysteresis)
    reference_every = 600  # checks between reference refreshes, so slow light changes don't drift

    def setup(self, cameras: list[str]) -> None:
        self.matcher = Matcher()
        self.cameras: dict[str, Camera] = {c: Camera(self.learn_checks) for c in cameras}

    def on_frame(self, camera: str, frame: np.ndarray, ts: datetime) -> list[Event]:
        cam = self.cameras.setdefault(camera, Camera(self.learn_checks))
        v = look(frame)
        visible = v.brightness >= self.dark and v.contrast >= self.flat

        # Day/night: a camera switching to IR loses its colour. That is not tampering,
        # so learn the new normal. A covered lens is colourless too, hence "visible".
        if visible:
            colour = self.colour_mode(cam.colour, v.saturation)
            if cam.colour is not None and colour != cam.colour:
                log.info("%s: switched to %s mode, re-learning", camera, "colour" if colour else "IR")
                self.relearn(cam)
            cam.colour = colour

        if cam.learning > 0:
            cam.samples.append((v.brightness, v.contrast, v.sharpness))
            cam.learning -= 1
            if cam.learning == 0:
                self.set_reference(cam, v)
            return []

        problem = self.problem(cam, v, visible)
        if problem is None:
            cam.bad = 0
            if cam.incident is None:
                cam.samples.append((v.brightness, v.contrast, v.sharpness))
                cam.since_reference += 1
                if cam.since_reference >= self.reference_every:
                    self.set_reference(cam, v)
                return []
            cam.good += 1
            if cam.good < self.persist:
                return []
            was, since = cam.incident, cam.incident_since
            cam.incident, cam.incident_since, cam.good = None, None, 0
            seconds = round((ts - since).total_seconds()) if since else 0
            return [self.event(camera, "restored", 1.0, time=ts, data={
                "area": "Whole view", "detail": f"View back to normal after {was} ({seconds} s)",
                "was": was, "seconds": seconds})]

        cam.good = 0
        if cam.incident is not None:
            return []  # one event per incident
        cam.bad += 1
        if cam.bad < self.persist:
            return []
        label, confidence, data = problem
        cam.incident, cam.incident_since, cam.bad = label, ts, 0
        return [self.event(camera, label, confidence, time=ts, data={"area": "Whole view", **data})]

    def problem(self, cam: Camera, v: Look, visible: bool) -> Optional[tuple[str, float, dict[str, Any]]]:
        """(label, confidence, data) if this check looks tampered, else None."""
        brightness, contrast, sharpness = np.median(np.array(cam.samples), axis=0)
        numbers = {"brightness": round(v.brightness, 1), "contrast": round(v.contrast, 1)}

        # Covered: only if the normal view is not itself dark or plain.
        if v.brightness < self.dark and brightness >= 2 * self.dark:
            return "covered", 0.95, {"detail": "View blocked: image went dark", **numbers}
        if v.contrast < self.flat and contrast >= self.plain:
            return "covered", 0.95, {"detail": "View blocked: one flat colour", **numbers}
        if not visible:
            return None  # dark or plain, but so is the normal view

        if contrast >= self.plain and sharpness > 0 and v.sharpness < self.blur_ratio * sharpness:
            share = v.sharpness / sharpness
            return "blurred", min(1.0, max(0.5, 1 - share)), {
                "detail": f"Image blurred: sharpness {share:.0%} of normal",
                "sharpness": round(v.sharpness, 3), "normal_sharpness": round(float(sharpness), 3)}

        ref = cam.reference
        if ref is None or len(ref.keypoints) < self.min_features:
            return None  # too few features to tell (e.g. a blank wall)
        m = self.matcher.compare(ref, v.grey)
        matching = {"features": m.features, "inliers": m.inliers, "shift": round(m.shift, 3)}
        if m.inliers < max(self.min_inliers, self.inlier_ratio * m.features):
            return "moved", 0.9, {
                "detail": f"View turned away: {m.inliers} of {m.features} features match", **matching}
        if m.shift > self.max_shift:
            return "moved", min(1.0, 0.5 + m.shift), {
                "detail": f"View moved by {m.shift:.0%} of the width", **matching}
        return None

    def colour_mode(self, previous: Optional[bool], saturation: float) -> bool:
        if saturation > self.colour_on:
            return True
        if saturation < self.colour_off:
            return False
        if previous is not None:
            return previous
        return saturation >= (self.colour_on + self.colour_off) / 2

    def relearn(self, cam: Camera) -> None:
        """Start a new baseline. An incident already sent stays open until the view is normal."""
        cam.learning = self.relearn_checks
        cam.samples.clear()
        cam.reference = None
        cam.bad = cam.good = 0

    def set_reference(self, cam: Camera, v: Look) -> None:
        cam.reference = self.matcher.reference(v.grey)
        cam.since_reference = 0
