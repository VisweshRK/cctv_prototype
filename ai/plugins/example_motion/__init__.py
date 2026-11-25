"""Example plugin (working): simple motion detection by frame differencing.

Copy this folder to start a new use case. It shows the whole contract:
keep per-camera state, look at each frame, return Events.
"""

from __future__ import annotations

from datetime import datetime

import cv2
import numpy as np

from cctv_ai.plugin import Event, Plugin as Base


class Plugin(Base):
    use_case = "example_motion"
    enabled = False  # reference plugin only; production runs the seven selected cases
    needs = {"video"}
    interval = 1.0

    threshold = 25  # pixel difference (0..255) that counts as "changed"
    min_area = 0.02  # fraction of the image that must change to report motion
    cooldown = 10.0  # seconds between events per camera

    def setup(self, cameras: list[str]) -> None:
        self.previous: dict[str, np.ndarray] = {}
        self.last_event: dict[str, datetime] = {}

    def on_frame(self, camera: str, frame: np.ndarray, ts: datetime) -> list[Event]:
        # small grey image: cheap to compare and less sensitive to noise
        small = cv2.resize(frame, (320, 180))
        grey = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)

        previous = self.previous.get(camera)
        self.previous[camera] = grey
        if previous is None:
            return []

        changed = cv2.absdiff(previous, grey) > self.threshold
        area = float(np.count_nonzero(changed)) / changed.size
        if area < self.min_area:
            return []

        last = self.last_event.get(camera)
        if last is not None and (ts - last).total_seconds() < self.cooldown:
            return []
        self.last_event[camera] = ts

        return [self.event(camera, "motion", min(1.0, area * 5), data={"changed_area": round(area, 3)})]
