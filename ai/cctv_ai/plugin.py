"""The one interface every use-case plugin implements.

A plugin receives a video frame or a chunk of audio from one camera and
returns a list of Events. The worker sends those events to go2rtc
(POST /api/cctv/events). Plugins never talk to go2rtc themselves.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np


@dataclass
class Event:
    """Same fields as the Go Event in internal/cctv/events.go."""

    camera: str
    use_case: str  # must match the plugin's use_case
    label: str  # e.g. "person", "smoke", "fall"
    confidence: float  # 0..1
    time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    snapshot: Optional[str] = None  # optional path or URL of an image
    data: dict[str, Any] = field(default_factory=dict)  # optional extra details

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["time"] = self.time.isoformat()
        if d["snapshot"] is None:
            del d["snapshot"]
        if not d["data"]:
            del d["data"]
        return d


class Plugin:
    """Base class. Copy plugins/example_motion to start a new use case.

    Class attributes a plugin sets:
      use_case  - the slot name, e.g. "intrusion" (shows up in events)
      enabled   - False until the plugin really works; the worker skips it
      needs     - {"video"}, {"audio"} or both
      interval  - seconds between frames for this plugin (video only)
    """

    use_case: str = ""
    enabled: bool = False
    needs: set[str] = {"video"}
    interval: float = 1.0

    def setup(self, cameras: list[str]) -> None:
        """Called once before the first frame. Load models here."""

    def on_frame(self, camera: str, frame: np.ndarray, ts: datetime) -> list[Event]:
        """frame is a BGR image (OpenCV format), shape (height, width, 3)."""
        return []

    def on_audio(self, camera: str, samples: np.ndarray, sample_rate: int, ts: datetime) -> list[Event]:
        """samples is mono int16 PCM, about one second long."""
        return []

    def event(self, camera: str, label: str, confidence: float, **kwargs: Any) -> Event:
        """Helper so plugins don't repeat their use_case."""
        return Event(camera=camera, use_case=self.use_case, label=label, confidence=confidence, **kwargs)
