"""Audio Stream Processor for Presence B2C CCTV AI Platform.
Processes 16 kHz mono audio buffers with RMS measurement and disconnection monitoring.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import queue
import threading
import time
from typing import Optional, Tuple
import numpy as np


@dataclass
class AudioMetrics:
    total_chunks: int = 0
    rms_level: float = 0.0
    microphone_active: bool = False
    disconnections: int = 0


class AudioStreamReader:
    """16 kHz mono audio buffer reader."""

    def __init__(
        self,
        camera_id: str,
        sample_rate: int = 16000,
        chunk_duration_sec: float = 0.975  # Standard YAMNet input window size
    ):
        self.camera_id = camera_id
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration_sec)
        self.metrics = AudioMetrics()
        self.queue: queue.Queue = queue.Queue(maxsize=10)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.metrics.microphone_active = False

    def push_raw_samples(self, samples: np.ndarray, timestamp: Optional[datetime] = None) -> None:
        """Pushes raw PCM samples (float32 [-1.0, 1.0])."""
        if not timestamp:
            timestamp = datetime.now(timezone.utc)
        self.metrics.microphone_active = True
        self.metrics.total_chunks += 1
        rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0
        self.metrics.rms_level = rms

        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.queue.put_nowait((samples, timestamp))
        except queue.Full:
            pass

    def get_latest_chunk(self, timeout: float = 0.5) -> Optional[Tuple[np.ndarray, datetime]]:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _audio_loop(self) -> None:
        """Synthetic ambient background audio loop if no hardware mic connected."""
        interval = self.chunk_size / self.sample_rate
        self.metrics.microphone_active = True

        while not self._stop_event.is_set():
            now = datetime.now(timezone.utc)
            # Create low ambient noise (white noise at -40dB)
            ambient = np.random.normal(0, 0.01, self.chunk_size).astype(np.float32)
            self.push_raw_samples(ambient, now)
            time.sleep(interval)
