"""Face recognition - SLOT (not built yet). See README.md."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from cctv_ai.plugin import Event, Plugin as Base


class Plugin(Base):
    use_case = "face_recognition"
    enabled = False  # set True when it works
    needs = {"video"}
    interval = 1.0

    def setup(self, cameras: list[str]) -> None:
        # TODO: load models here (once, not per frame)
        pass

    def on_frame(self, camera: str, frame: np.ndarray, ts: datetime) -> list[Event]:
        # TODO: look at the frame and return events, e.g.
        #   return [self.event(camera, "label", 0.9)]
        return []
