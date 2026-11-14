"""Base Plugin Interface for Presence B2C CCTV AI Platform.
All 10 use case modules implement this contract.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

from ai.cctv_ai.objects import TrackedObject
from packages.event_schemas.events import StandardAIEvent


class BasePlugin(ABC):
    """Abstract base class for all AI module plugins."""

    def __init__(self, plugin_id: str, name: str, enabled: bool = True):
        self.plugin_id = plugin_id
        self.name = name
        self.enabled = enabled
        self.config: Dict[str, Any] = {}
        self.active_cameras: set[str] = set()

    def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize models, thresholds, zones, and resources."""
        if config:
            self.config.update(config)

    def on_frame(
        self,
        camera: str,
        frame: np.ndarray,
        timestamp: datetime
    ) -> List[StandardAIEvent]:
        """Hook called for raw video frames (used by Fire, Tampering, etc.)."""
        return []

    def on_audio(
        self,
        camera: str,
        samples: np.ndarray,
        timestamp: datetime
    ) -> List[StandardAIEvent]:
        """Hook called for audio buffers (used by Baby Crying, Glass break, etc.)."""
        return []

    def on_objects(
        self,
        camera: str,
        frame: np.ndarray,
        timestamp: datetime,
        objects: List[TrackedObject],
        ended: List[TrackedObject]
    ) -> List[StandardAIEvent]:
        """Hook called when shared detector and tracker update tracked entities."""
        return []

    def shutdown(self) -> None:
        """Clean up resources, models, buffers."""
        pass
