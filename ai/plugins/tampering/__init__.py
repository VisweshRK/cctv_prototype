"""Camera tampering (image side): covered, blurred / sprayed, moved / turned away, shaking.

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

from .checks import OVERLAY, TILES, Look, Match, Matcher, Reference, changed, jump, look, offset

log = logging.getLogger("cctv_ai.tampering")

# A worse problem during an open incident still gets its own event (shaken, then turned, then covered).
SEVERITY = {"shaking": -1, "moved": 0, "blurred": 1, "covered": 2}

# problem() result for "can't tell yet": the picture is still changing a lot (someone moving close to
# the lens, or the camera still swinging). Counts as neither tampered nor normal.
HOLD = "hold"

# Tiles that carry the burned-in clock and camera name are left out of the per-tile blur check.
INNER = np.zeros((TILES[1], TILES[0]), bool)
INNER[round(OVERLAY * TILES[1]):TILES[1] - round(OVERLAY * TILES[1])] = True


@dataclass
class Camera:
    """What the plugin remembers about one camera."""

    learning: int  # checks left before the baseline is ready (no alerts until then)
    samples: deque = field(default_factory=lambda: deque(maxlen=600))  # (brightness, contrast, sharpness), 5 min
    tiles: deque = field(default_factory=lambda: deque(maxlen=600))  # (tile_sharpness, tile_variance), 5 min
    previous: Optional[np.ndarray] = None  # grey picture of the last check
    # Pictures spread over the learning minute (then one every 20 s of normal view): the reference is
    # their median, so people who move about are not part of it.
    views: deque = field(default_factory=lambda: deque(maxlen=30))
    reference: Optional[Reference] = None  # view to compare with for "moved"
    since_reference: int = 0
    colour: Optional[bool] = None  # False while the camera is in IR (grey) mode
    bad: int = 0  # tampered checks in a row
    good: int = 0  # normal checks in a row during an incident
    incident: Optional[str] = None  # label of the event already sent
    incident_since: Optional[datetime] = None
    moved_for: int = 0  # checks in a row the view has been moved during an incident


class Plugin(Base):
    use_case = "tampering"
    enabled = True
    needs = {"video"}
    interval = 0.5

    learn_checks = 120  # checks (0.5 s each, so 60 s) to learn what this camera normally looks like
    relearn_checks = 20  # after a day/night (colour/IR) switch, or after accepting a new position
    persist = 2  # checks in a row before an event (and before "restored"): 1 s
    unsure_persist = 6  # 3 s for a borderline "moved": people close to the lens hide most of the view for a moment
    accept_checks = 240  # moved this long (2 min) = accept the new position as normal
    dark = 15.0  # mean brightness below this = covered
    flat = 8.0  # contrast below this = one flat colour = covered
    plain = 12.0  # a normal view with less contrast than this is too plain for the flat and blur checks
    blur_ratio = 0.3  # sharpness below this share of normal = blurred
    min_features = 30  # reference views with fewer ORB features are not checked for "moved"
    min_inliers = 12  # fewer matching features than this = turned away
    inlier_ratio = 0.15  # fewer than this share matching = a borderline match (unsure_persist)
    max_shift = 0.2  # the view's corners moved more than this share of the width = moved
    still_shift = 0.01  # the reference's edges within this share of the width of where they were ...
    still_response = 0.05  # ... with at least this strength = the camera has not moved
    anchor_px = 4.0  # a feature within this many pixels (of 320) of where it was is still in place
    min_anchored = 20  # this many features still in place (and anchored_ratio of the reference) ...
    anchored_ratio = 0.1
    min_spread = 0.3  # ... spread over this share of the view = the camera has not moved
    textured = 64.0  # tiles with less grey-level variance than this (plain wall) are not used for blur
    keep_ratio = 0.5  # a tile at least this share of its normal sharpness is still sharp
    min_kept = 0.25  # this share of the detailed tiles still sharp = not blurred (a person in front)
    settle = 0.1  # more than this share of the picture changed since the last check = still moving
    shake_shift = 0.02  # the whole picture jumped more than this share of the width since the last check ...
    shake_response = 0.1  # ... with at least this phase-correlation strength = camera shaken
    colour_on, colour_off = 16.0, 8.0  # mean saturation that means colour / IR mode (hysteresis)
    reference_every = 1200  # checks (10 min) between reference refreshes, so slow light changes don't drift

    def setup(self, cameras: list[str]) -> None:
        self.matcher = Matcher()
        self.cameras: dict[str, Camera] = {c: Camera(self.learn_checks) for c in cameras}

    def on_frame(self, camera: str, frame: np.ndarray, ts: datetime) -> list[Event]:
        cam = self.cameras.setdefault(camera, Camera(self.learn_checks))
        v = look(frame)
        previous, cam.previous = cam.previous, v.grey
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
            cam.tiles.append((v.tile_sharpness, v.tile_variance))
            if cam.learning % 4 == 0:
                cam.views.append(v.grey)
            cam.learning -= 1
            if cam.learning == 0:
                self.set_reference(cam, v)
            return []

        problem = self.problem(cam, v, visible, previous)
        if problem == HOLD:
            return []
        if problem is None:
            cam.bad = cam.moved_for = 0
            if cam.incident is None:
                cam.samples.append((v.brightness, v.contrast, v.sharpness))
                cam.tiles.append((v.tile_sharpness, v.tile_variance))
                cam.since_reference += 1
                if cam.since_reference % 40 == 0:
                    cam.views.append(v.grey)
                if cam.since_reference >= self.reference_every:
                    self.set_reference(cam, v)
                return []
            cam.good += 1
            if cam.good < self.persist:
                return []
            was, since = cam.incident, cam.incident_since
            cam.incident, cam.incident_since, cam.good, cam.moved_for = None, None, 0, 0
            seconds = round((ts - since).total_seconds()) if since else 0
            return [self.event(camera, "restored", 1.0, time=ts, data={
                "area": "Whole view", "detail": f"View back to normal after {was} ({seconds} s)",
                "was": was, "seconds": seconds})]

        cam.good = 0
        label, confidence, data, sure = problem
        if cam.incident is not None:
            cam.moved_for = cam.moved_for + 1 if label == "moved" else 0
            if cam.moved_for >= self.accept_checks:
                return [self.accept(camera, cam, ts)]
            if SEVERITY[label] <= SEVERITY[cam.incident]:
                cam.bad = 0
                return []  # one event per incident
        cam.bad += 1
        if cam.bad < (self.persist if sure else self.unsure_persist):
            return []
        cam.incident, cam.bad = label, 0
        cam.incident_since = cam.incident_since or ts
        return [self.event(camera, label, confidence, time=ts, data={"area": "Whole view", **data})]

    def accept(self, camera: str, cam: Camera, ts: datetime) -> Event:
        """The camera stayed in a new position: close the incident and learn the new view,
        so tampering is caught again from there."""
        was, since = cam.incident, cam.incident_since
        seconds = round((ts - since).total_seconds()) if since else 0
        cam.incident, cam.incident_since, cam.moved_for = None, None, 0
        self.relearn(cam)
        log.info("%s: new position accepted after %d s, re-learning", camera, seconds)
        return self.event(camera, "restored", 1.0, time=ts, data={
            "area": "Whole view", "detail": f"New camera position accepted as normal after {seconds} s",
            "was": was, "seconds": seconds, "accepted": True})

    def problem(self, cam: Camera, v: Look, visible: bool, previous: Optional[np.ndarray]) -> Any:
        """(label, confidence, data, sure) if this check looks tampered, HOLD if it can't tell yet
        (the picture is still changing a lot), else None."""
        brightness, contrast, sharpness = np.median(np.array(cam.samples), axis=0)
        numbers = {"brightness": round(v.brightness, 1), "contrast": round(v.contrast, 1)}

        # Covered: only if the normal view is not itself dark or plain.
        if v.brightness < self.dark and brightness >= 2 * self.dark:
            return "covered", 0.95, {"detail": "View blocked: image went dark", **numbers}, True
        if v.contrast < self.flat and contrast >= self.plain:
            return "covered", 0.95, {"detail": "View blocked: one flat colour", **numbers}, True
        if not visible:
            return None  # dark or plain, but so is the normal view

        ref = cam.reference
        m = None
        if ref is not None and len(ref.keypoints) >= self.min_features:
            m = self.matcher.compare(ref, v.grey, self.anchor_px)
        in_place = m is not None and (self.in_place(m) or self.still(ref, v.grey))

        # Shaking: the whole picture jumped since the last check and the background is not where it was.
        # People moving, even right in front of the lens, leave the background still, so the jump is ~0.
        if m is not None and previous is not None and not in_place:
            shift, strength = jump(previous, v.grey)
            if shift >= self.shake_shift and strength >= self.shake_response:
                return "shaking", min(1.0, 0.5 + strength / 2), {
                    "detail": f"Camera being handled: picture jumped {shift:.0%} of the width",
                    "jump": round(shift, 3), "strength": round(strength, 2), "anchored": m.anchored}, True
        settled = previous is None or changed(previous, v.grey) <= self.settle

        if contrast >= self.plain and sharpness > 0 and v.sharpness < self.blur_ratio * sharpness:
            kept, detailed = self.sharp_tiles(cam, v)
            # A person's plain clothes close to the lens soften part of the view; a smeared lens all of it.
            if detailed < 4 or kept < self.min_kept * detailed:
                if not settled:
                    return HOLD
                share = v.sharpness / sharpness
                return "blurred", min(1.0, max(0.5, 1 - share)), {
                    "detail": f"Image blurred: sharpness {share:.0%} of normal",
                    "sharpness": round(v.sharpness, 3), "normal_sharpness": round(float(sharpness), 3),
                    "sharp_tiles": kept, "detailed_tiles": detailed}, True

        if m is None or in_place:
            return None  # too few features to tell (e.g. a blank wall), or the background is where it was
        if not settled:
            return HOLD  # someone moving close to the lens, or the camera still swinging: wait
        matching = {"features": m.features, "inliers": m.inliers, "shift": round(m.shift, 3),
                    "anchored": m.anchored}
        if m.inliers < self.min_inliers:
            return "moved", 0.9, {
                "detail": f"View turned away: {m.inliers} of {m.features} features match", **matching}, True
        # Few matches but a fit: people close to the lens hide most of the view, and the few matches
        # left (often on one side) give a rough fit. Trust its shift only if it lasts.
        if m.shift > self.max_shift:
            sure = m.inliers >= self.inlier_ratio * m.features
            return "moved", min(1.0, 0.5 + m.shift), {
                "detail": f"View moved by {m.shift:.0%} of the width", **matching}, sure
        return None

    def in_place(self, m: Match) -> bool:
        """Enough reference features still exactly where they were, spread over the view: the camera has
        not moved, whatever is in front of it."""
        return (m.anchored >= max(self.min_anchored, self.anchored_ratio * m.features)
                and m.spread >= self.min_spread)

    def still(self, ref: Reference, grey: np.ndarray) -> bool:
        """The reference's edges are where they were (for plain rooms where few features anchor)."""
        shift, strength = offset(ref.grey, grey)
        return shift <= self.still_shift and strength >= self.still_response

    def sharp_tiles(self, cam: Camera, v: Look) -> tuple[int, int]:
        """(tiles still at least keep_ratio of their normal sharpness, tiles with detail to judge)."""
        normal = np.median(np.array([t[0] for t in cam.tiles]), axis=0)
        detail = np.median(np.array([t[1] for t in cam.tiles]), axis=0)
        detailed = INNER & (detail >= self.textured)
        kept = detailed & (v.tile_sharpness >= self.keep_ratio * normal)
        return int(kept.sum()), int(detailed.sum())

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
        cam.tiles.clear()
        cam.views.clear()
        cam.reference = None
        cam.bad = cam.good = 0

    def set_reference(self, cam: Camera, v: Look) -> None:
        grey = np.median(np.array(cam.views), axis=0).astype(np.uint8) if cam.views else v.grey
        cam.reference = self.matcher.reference(grey)
        cam.since_reference = 0
