"""Shared Camera Frame Reader for Presence B2C CCTV AI Platform.
Ensures single camera decoding per stream, non-destructive multi-consumer frame distribution,
dropped-frame metrics, and synthetic test stream generator when no physical camera is connected.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import queue
import threading
import time
from typing import Generator, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameMetrics:
    total_frames: int = 0
    dropped_frames: int = 0
    fps: float = 0.0
    last_frame_time: Optional[datetime] = None
    connected: bool = False
    reconnect_count: int = 0


class CameraFrameReader:
    """Threaded frame reader with single-decode architecture and non-destructive latest-frame sharing."""

    def __init__(
        self,
        camera_id: str,
        stream_source: str,  # "rtsp://...", "file.mp4", "0" (webcam), or "synthetic"
        target_fps: int = 15,
        buffer_size: int = 2
    ):
        self.camera_id = camera_id
        self.stream_source = stream_source
        self.target_fps = target_fps
        self.buffer_size = buffer_size

        self.queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.metrics = FrameMetrics()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

        # Multi-consumer non-destructive frame storage
        self._latest_frame: Optional[Tuple[np.ndarray, datetime]] = None
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self.metrics.connected = False

    def get_latest_frame(self, timeout: float = 0.5) -> Optional[Tuple[np.ndarray, datetime]]:
        """Returns the latest frame copy non-destructively so multiple consumers
        (AI pipeline, live MJPEG streams, snapshots) can read simultaneously without starving each other.
        """
        # Fast path: return cached latest frame if available
        with self._lock:
            if self._latest_frame is not None:
                frame, ts = self._latest_frame
                return frame.copy(), ts

        # If not yet populated, wait for the first frame
        if self._new_frame_event.wait(timeout=timeout):
            with self._lock:
                if self._latest_frame is not None:
                    frame, ts = self._latest_frame
                    return frame.copy(), ts

        # Fallback to queue if event didn't fire
        try:
            item = self.queue.get(timeout=0.05)
            with self._lock:
                self._latest_frame = item
            return item[0].copy(), item[1]
        except queue.Empty:
            return None

    def _publish_frame(self, frame: np.ndarray, now: datetime) -> None:
        """Stores latest frame under lock and signals consumers."""
        with self._lock:
            self._latest_frame = (frame, now)
        self._new_frame_event.set()

        # Also maintain bounded queue for legacy queue consumers
        if self.queue.full():
            try:
                self.queue.get_nowait()
                self.metrics.dropped_frames += 1
            except queue.Empty:
                pass
        try:
            self.queue.put_nowait((frame, now))
        except queue.Full:
            self.metrics.dropped_frames += 1

    def _capture_loop(self) -> None:
        interval = 1.0 / max(1, self.target_fps)
        synthetic = (str(self.stream_source).lower() == "synthetic" or not self.stream_source)

        if synthetic:
            self._synthetic_loop(interval)
            return

        while not self._stop_event.is_set():
            try:
                if not self._cap or not self._cap.isOpened():
                    self._open_capture()
                    if not self._cap or not self._cap.isOpened():
                        time.sleep(2.0)
                        continue

                ret, frame = self._cap.read()
                now = datetime.now(timezone.utc)

                if not ret or frame is None:
                    self.metrics.connected = False
                    self.metrics.reconnect_count += 1
                    if self._cap:
                        try:
                            self._cap.release()
                        except Exception:
                            pass
                        self._cap = None
                    time.sleep(1.0)
                    continue

                self.metrics.connected = True
                self.metrics.total_frames += 1
                self.metrics.last_frame_time = now

                self._publish_frame(frame, now)
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in camera capture loop: {e}")
                time.sleep(1.0)

    def _open_capture(self) -> None:
        src = self.stream_source
        if str(src).lower() in ["webcam", "camera0", "cam0", "0"]:
            src = 0
        elif str(src).isdigit():
            src = int(src)
        try:
            self._cap = cv2.VideoCapture(src)
            if self._cap.isOpened():
                self.metrics.connected = True
        except Exception as e:
            logger.warning(f"Failed to open video capture {src}: {e}")
            self._cap = None

    def _synthetic_loop(self, interval: float) -> None:
        """Generates synthetic test camera frames continuously with robust error recovery."""
        self.metrics.connected = True
        w, h = 640, 480
        t = 0

        while not self._stop_event.is_set():
            start_t = time.time()
            try:
                now = datetime.now(timezone.utc)

                # Create clean background
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                frame[:] = (35, 30, 30)  # Dark slate background

                # Draw a simulated room (floor, door, wall)
                cv2.line(frame, (0, 350), (w, 350), (80, 80, 80), 2)  # Floor line
                cv2.rectangle(frame, (50, 150), (170, 350), (60, 50, 40), -1)  # Door
                cv2.putText(frame, "DOOR (ZONE: ENTRANCE)", (55, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

                # Draw dynamic animated test subject moving across camera view
                cycle = (t % 150)  # 10s loop at 15 FPS
                if cycle < 110:
                    px = int(60 + (cycle / 110.0) * 480)
                    py = 220
                    # Person avatar: head, body, legs
                    cv2.circle(frame, (px + 20, py + 15), 14, (180, 160, 150), -1)
                    cv2.rectangle(frame, (px + 5, py + 30), (px + 35, py + 85), (200, 70, 50), -1)
                    cv2.rectangle(frame, (px + 8, py + 85), (px + 18, py + 125), (50, 50, 150), -1)
                    cv2.rectangle(frame, (px + 22, py + 85), (px + 32, py + 125), (50, 50, 150), -1)
                    cv2.putText(frame, "MOVING PERSON (TEST)", (px - 15, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 180), 1)

                # Draw camera overlay text
                cv2.putText(frame, f"CAM: {self.camera_id} [SYNTHETIC TEST STREAM]", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1)
                cv2.putText(frame, f"TIME: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                self.metrics.total_frames += 1
                self.metrics.last_frame_time = now

                self._publish_frame(frame, now)
                t += 1
            except Exception as e:
                logger.error(f"Error in synthetic frame loop: {e}")

            elapsed = time.time() - start_t
            sleep_time = max(0.001, interval - elapsed)
            time.sleep(sleep_time)
