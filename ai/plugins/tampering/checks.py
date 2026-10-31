"""Image measurements for tampering: brightness, contrast, sharpness, colour,
and whether the view still matches a reference frame (ORB + homography)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# Every frame is shrunk to this size first: cheap, and the numbers don't depend on
# the camera's resolution. 320x180 rather than 160x90 because ORB needs the pixels.
SIZE = (320, 180)
TILES = (8, 6)  # sharpness is measured per tile (40x30 pixels each)

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
    lap = cv2.Laplacian(grey, cv2.CV_32F).reshape(ty, h // ty, tx, w // tx)
    sharpness = float(np.percentile(lap.var(axis=(1, 3)), 75)) / max(contrast * contrast, 1.0)
    saturation = float(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
    return Look(grey, float(p50), contrast, sharpness, saturation)


@dataclass
class Reference:
    keypoints: tuple
    descriptors: Optional[np.ndarray]


@dataclass
class Match:
    features: int  # features in the reference
    inliers: int  # matches that agree on one homography
    shift: float  # how far the image corners moved on average, as a fraction of the width


class Matcher:
    """Compares the current view with a reference view of the same camera."""

    def __init__(self) -> None:
        self.orb = cv2.ORB_create(nfeatures=500, edgeThreshold=15, patchSize=15)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def reference(self, grey: np.ndarray) -> Reference:
        keypoints, descriptors = self.orb.detectAndCompute(grey, None)
        return Reference(keypoints, descriptors)

    def compare(self, ref: Reference, grey: np.ndarray) -> Match:
        features = len(ref.keypoints)
        cur = self.reference(grey)
        if ref.descriptors is None or cur.descriptors is None or len(cur.keypoints) < 4:
            return Match(features, 0, 1.0)
        matches = self.bf.match(ref.descriptors, cur.descriptors)
        if len(matches) < 4:
            return Match(features, 0, 1.0)
        src = np.float32([ref.keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([cur.keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if homography is None:
            return Match(features, 0, 1.0)
        # How far the corners moved: catches a pan or tilt, and also a rotation or zoom,
        # which leave the centre in place.
        w, h = SIZE
        corners = np.float32([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]])
        moved = cv2.perspectiveTransform(corners, homography) - corners
        shift = float(np.linalg.norm(moved, axis=2).mean()) / w
        return Match(features, int(mask.sum()), shift)
