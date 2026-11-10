"""Image measurements for tampering: brightness, contrast, sharpness, colour,
whether the view still matches a reference frame (ORB + homography), and how the
picture changed since the last check (camera shake vs people moving)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# Every frame is shrunk to this size first: cheap, and the numbers don't depend on
# the camera's resolution. 320x180 rather than 160x90 because ORB needs the pixels.
SIZE = (320, 180)
TILES = (8, 6)  # sharpness is measured per tile (40x30 pixels each)
OVERLAY = 1 / TILES[1]  # top and bottom band (one tile row) where cameras burn the clock and name

# Cameras burn a clock and a name onto the picture. That text stays sharp and bright when the
# lens is covered or smeared, so every measure below ignores a few percent of the picture:
# percentiles instead of mean / standard deviation, and the sharpest tiles are left out.


@dataclass
class Look:
    grey: np.ndarray  # SIZE, uint8
    brightness: float  # median grey level, 0..255
    contrast: float  # spread of grey levels: (90th - 10th percentile) / 2.56, ~ standard deviation
    sharpness: float  # Laplacian variance of the 75th-percentile tile / contrast^2
    saturation: float  # mean HSV saturation, 0..255; ~0 when the camera is in IR (grey) mode
    tile_sharpness: np.ndarray  # TILES[1] x TILES[0]: each tile's Laplacian variance / its own variance
    tile_variance: np.ndarray  # TILES[1] x TILES[0]: grey-level variance (how much detail a tile has)


def look(frame: np.ndarray) -> Look:
    small = cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA)
    if small.ndim == 2:
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    p10, p50, p90 = np.percentile(grey, (10, 50, 90))
    contrast = float(p90 - p10) / 2.56
    # 75th percentile: the clock's tiles are among the sharpest and are skipped, while a view that
    # is mostly blank wall or sky still has a quarter of its tiles with real detail.
    (w, h), (tx, ty) = SIZE, TILES
    lap = cv2.Laplacian(grey, cv2.CV_32F).reshape(ty, h // ty, tx, w // tx).var(axis=(1, 3))
    sharpness = float(np.percentile(lap, 75)) / max(contrast * contrast, 1.0)
    saturation = float(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
    # Per tile, divided by the tile's own variance: a person's plain shirt over half the view makes
    # those tiles soft, but the other tiles stay as sharp as normal. A smeared lens softens all of them.
    variance = grey.astype(np.float32).reshape(ty, h // ty, tx, w // tx).var(axis=(1, 3))
    return Look(grey, float(p50), contrast, sharpness, saturation, lap / np.maximum(variance, 25.0), variance)


def changed(previous: np.ndarray, grey: np.ndarray) -> float:
    """Share of the picture that changed since the previous check (80x45 thumbnails, more than 20 grey
    levels after taking out an overall brightness change). ~0 for a still view, a turned camera at rest
    or a smeared lens; large while someone moves close to the lens or the camera is being moved."""
    a = cv2.resize(previous, (80, 45), interpolation=cv2.INTER_AREA).astype(np.int16)
    b = cv2.resize(grey, (80, 45), interpolation=cv2.INTER_AREA).astype(np.int16)
    d = b - a
    d -= int(np.median(d))
    return float((np.abs(d) > 20).mean())


_window = cv2.createHanningWindow(SIZE, cv2.CV_32F)
_band = round(OVERLAY * SIZE[1])
_inner_window = cv2.createHanningWindow((SIZE[0], SIZE[1] - 2 * _band), cv2.CV_32F)


def jump(previous: np.ndarray, grey: np.ndarray) -> tuple[float, float]:
    """How far the whole picture moved since the previous check (phase correlation): (shift as a share of
    the width, strength 0..1). People moving leave the background where it was, so the peak stays at ~0;
    a shaken camera moves everything together."""
    (dx, dy), response = cv2.phaseCorrelate(np.float32(previous), np.float32(grey), _window)
    return float(np.hypot(dx, dy)) / SIZE[0], float(response)


def offset(ref: np.ndarray, grey: np.ndarray) -> tuple[float, float]:
    """Where the reference view's edges are in the current picture (phase correlation of the gradient,
    clock and name bands left out because they stay put when the camera turns): (shift as a share of the
    width, strength). ~0 with some strength = the camera has not moved. Works on plain rooms where ORB
    finds few corners, since it uses every edge (a cupboard side, a cable, a door frame)."""
    def edges(a):
        a = np.float32(a[_band:SIZE[1] - _band])
        return cv2.magnitude(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))
    (dx, dy), response = cv2.phaseCorrelate(edges(ref), edges(grey), _inner_window)
    return float(np.hypot(dx, dy)) / SIZE[0], float(response)


@dataclass
class Reference:
    keypoints: tuple
    descriptors: Optional[np.ndarray]
    grey: Optional[np.ndarray] = None


@dataclass
class Match:
    features: int  # features in the reference
    inliers: int  # matches that agree on one homography
    shift: float  # how far the image corners moved on average, as a fraction of the width
    anchored: int = 0  # matches still where they were (background in place, even with people in front)
    spread: float = 0.0  # how far apart the anchored matches are, as a share of the width or height


class Matcher:
    """Compares the current view with a reference view of the same camera."""

    def __init__(self) -> None:
        self.orb = cv2.ORB_create(nfeatures=500, edgeThreshold=15, patchSize=15)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def reference(self, grey: np.ndarray) -> Reference:
        keypoints, descriptors = self.orb.detectAndCompute(grey, None)
        return Reference(keypoints, descriptors, grey)

    def compare(self, ref: Reference, grey: np.ndarray, anchor_px: float = 4.0) -> Match:
        features = len(ref.keypoints)
        cur = self.reference(grey)
        if ref.descriptors is None or cur.descriptors is None or len(cur.keypoints) < 4:
            return Match(features, 0, 1.0)
        matches = self.bf.match(ref.descriptors, cur.descriptors)
        if len(matches) < 4:
            return Match(features, 0, 1.0)
        src = np.float32([ref.keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([cur.keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        # Anchored: the same feature at the same spot. People in front hide some of the background but
        # can't move it; a moved camera moves all of it. The clock and name bands are left out, because
        # they stay in place when the camera turns.
        w, h = SIZE
        p = src[:, 0]
        still = np.linalg.norm(dst[:, 0] - p, axis=1) <= anchor_px
        anchored = p[still & (p[:, 1] >= OVERLAY * h) & (p[:, 1] <= (1 - OVERLAY) * h)]
        spread = max(float(np.ptp(anchored[:, 0])) / w, float(np.ptp(anchored[:, 1])) / h) if len(anchored) else 0.0
        homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if homography is None:
            return Match(features, 0, 1.0, len(anchored), spread)
        # How far the corners moved: catches a pan or tilt, and also a rotation or zoom,
        # which leave the centre in place.
        corners = np.float32([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]])
        moved = cv2.perspectiveTransform(corners, homography) - corners
        shift = float(np.linalg.norm(moved, axis=2).mean()) / w
        return Match(features, int(mask.sum()), shift, len(anchored), spread)
