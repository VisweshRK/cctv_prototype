"""Master AI Pipeline for Presence B2C CCTV Platform.
Orchestrates single-stream decoding, shared object detector (YOLOX-Nano), shared tracker,
and feeds the 3 specialized AI plugins:
1. Fire & Smoke Detection
2. Electricity During Room Absence Analytics
3. Baby Crying Acoustic Classifier
"""

from datetime import datetime, timezone
import logging
import threading
import time
from typing import Dict, List, Optional
import numpy as np

from ai.cctv_ai.audio import AudioStreamReader
from ai.cctv_ai.detector import SharedDetector
from ai.cctv_ai.events import EventManager
from ai.cctv_ai.frames import CameraFrameReader
from ai.cctv_ai.motion import MotionDetector
from ai.cctv_ai.objects import TrackedObject
from ai.cctv_ai.plugin import BasePlugin
from ai.cctv_ai.tracker import SharedTracker
from ai.cctv_ai.zones import ZoneManager

# Import the 3 focused plugins
from ai.plugins.power_when_absent.plugin import PowerWhenAbsentPlugin
from ai.plugins.fire.plugin import FireSmokePlugin
from ai.plugins.baby_crying.plugin import BabyCryingPlugin

logger = logging.getLogger(__name__)


class MasterPipeline:
    """Integrated master AI pipeline executing across the 3 core safety use cases."""

    def __init__(
        self,
        camera_id: str = "camera_main",
        stream_source: str = "synthetic",
        event_manager: Optional[EventManager] = None,
        zone_manager: Optional[ZoneManager] = None
    ):
        self.camera_id = camera_id
        self.stream_source = stream_source
        self.event_manager = event_manager or EventManager()
        self.zone_manager = zone_manager or ZoneManager()

        # Shared frame & audio readers
        self.frame_reader = CameraFrameReader(camera_id=camera_id, stream_source=stream_source)
        self.audio_reader = AudioStreamReader(camera_id=camera_id)

        # Shared detector, motion detector, and Kalman tracker (for person presence/absence)
        self.detector = SharedDetector()
        self.motion_detector = MotionDetector()
        self.tracker = SharedTracker(camera_id=camera_id, use_kalman=True)

        # Initialize the 3 core safety plugins
        self.plugins: Dict[str, BasePlugin] = {
            "fire": FireSmokePlugin(zone_manager=self.zone_manager),
            "power_when_absent": PowerWhenAbsentPlugin(zone_manager=self.zone_manager),
            "baby_crying": BabyCryingPlugin(),
        }

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._audio_thread: Optional[threading.Thread] = None
        self.metrics = {
            "inference_fps": 0.0,
            "processed_frames": 0,
            "active_objects_count": 0,
            "last_active": None,
        }

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.frame_reader.start()
        self.audio_reader.start()

        # Setup plugins
        for p in self.plugins.values():
            p.setup({"camera_id": self.camera_id, "household_id": "household_default"})

        self._thread = threading.Thread(target=self._video_pipeline_loop, daemon=True)
        self._thread.start()

        self._audio_thread = threading.Thread(target=self._audio_pipeline_loop, daemon=True)
        self._audio_thread.start()
        logger.info(f"MasterPipeline started for camera '{self.camera_id}' (Fire, Electricity, Baby Crying).")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._audio_thread:
            self._audio_thread.join(timeout=2.0)

        self.frame_reader.stop()
        self.audio_reader.stop()

        for p in self.plugins.values():
            p.shutdown()
        logger.info(f"MasterPipeline stopped for camera '{self.camera_id}'.")

    def step_frame(self, frame: np.ndarray, timestamp: Optional[datetime] = None) -> list:
        """Processes a single video frame synchronously across Fire/Smoke and Electricity plugins."""
        if not timestamp:
            timestamp = datetime.now(timezone.utc)

        all_events = []

        # 1. Fire and Smoke Detection (HSV flame segmentation & cooking zone context)
        fire_plugin: FireSmokePlugin = self.plugins["fire"]
        if fire_plugin.enabled:
            all_events.extend(fire_plugin.on_frame(self.camera_id, frame, timestamp))

        # 2. Electricity / Power When Absent: Evaluate Appliance & Light ROIs
        power_plugin: PowerWhenAbsentPlugin = self.plugins["power_when_absent"]
        if power_plugin.enabled:
            all_events.extend(power_plugin.on_frame(self.camera_id, frame, timestamp))

        # 3. Motion Detection & Shared Object Detector (YOLOX): Person tracking for room occupancy/absence
        motion_res = self.motion_detector.detect(frame)
        self.metrics["has_motion"] = motion_res.has_motion
        self.metrics["motion_score"] = motion_res.motion_score
        self.metrics["motion_regions_count"] = len(motion_res.regions)

        has_active_tracks = len(self.tracker._tracks) > 0
        if motion_res.has_motion or has_active_tracks or str(self.stream_source).lower() == "synthetic":
            if motion_res.regions and len(motion_res.regions) > 0:
                detections = self.detector.detect(frame, regions=motion_res.regions)
            else:
                detections = self.detector.detect(frame)
        else:
            detections = []

        if not detections and str(self.stream_source).lower() == "synthetic":
            cycle = (self.metrics["processed_frames"] % 150)
            if cycle < 110:
                px = int(60 + (cycle / 110.0) * 480)
                py = 220
                detections = [{
                    "box": (px, py, px + 40, py + 130),
                    "label": "person",
                    "score": 0.92
                }]

        # 4. Shared Kalman tracker updates
        active_objects, ended_objects = self.tracker.update(detections, timestamp)
        self.metrics["active_objects_count"] = len(active_objects)

        # 5. Route person detections to PowerWhenAbsentPlugin to detect if room is empty
        if power_plugin.enabled:
            evs = power_plugin.on_objects(self.camera_id, frame, timestamp, active_objects, ended_objects)
            all_events.extend(evs)

        # 6. Route all events to EventManager for rule evaluation & notification
        dispatched = []
        for ev in all_events:
            alerts = self.event_manager.process_event(ev)
            dispatched.extend(alerts)

        self.metrics["processed_frames"] += 1
        self.metrics["last_active"] = timestamp
        return all_events

    def step_audio(self, samples: np.ndarray, timestamp: Optional[datetime] = None) -> list:
        """Processes audio buffer through audio plugins (Baby crying)."""
        if not timestamp:
            timestamp = datetime.now(timezone.utc)

        baby_plugin: BabyCryingPlugin = self.plugins["baby_crying"]
        events = []
        if baby_plugin.enabled:
            events = baby_plugin.on_audio(self.camera_id, samples, timestamp)
            for ev in events:
                self.event_manager.process_event(ev)
        return events

    def _video_pipeline_loop(self) -> None:
        last_t = time.time()
        while self._running:
            try:
                item = self.frame_reader.get_latest_frame(timeout=0.2)
                if item is None:
                    time.sleep(0.01)
                    continue
                frame, timestamp = item
                self.step_frame(frame, timestamp)

                now = time.time()
                fps = 1.0 / max(1e-4, now - last_t)
                self.metrics["inference_fps"] = round(fps, 1)
                last_t = now
            except Exception as e:
                logger.error(f"Error in video pipeline loop for '{self.camera_id}': {e}", exc_info=True)
                time.sleep(0.05)

    def _audio_pipeline_loop(self) -> None:
        while self._running:
            try:
                item = self.audio_reader.get_latest_chunk(timeout=0.2)
                if item is None:
                    time.sleep(0.01)
                    continue
                samples, timestamp = item
                self.step_audio(samples, timestamp)
            except Exception as e:
                logger.error(f"Error in audio pipeline loop for '{self.camera_id}': {e}", exc_info=True)
                time.sleep(0.05)
