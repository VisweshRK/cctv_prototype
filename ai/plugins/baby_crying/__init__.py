"""Baby crying - SLOT (not built yet). See README.md."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from cctv_ai.plugin import Event, Plugin as Base


class Plugin(Base):
    use_case = "baby_crying"
    enabled = False  # set True when it works
    needs = {"audio"}
    interval = 1.0

    def setup(self, cameras: list[str]) -> None:
        # TODO: load models here (once, not per frame)
        pass

    def on_audio(self, camera: str, samples: np.ndarray, sample_rate: int, ts: datetime) -> list[Event]:
        # TODO: look at ~1 s of int16 mono audio and return events, e.g.
        #   return [self.event(camera, "baby_crying", 0.9)]
        return []
