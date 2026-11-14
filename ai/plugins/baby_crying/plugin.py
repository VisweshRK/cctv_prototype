"""Use Case 10 — Baby Crying Detection Plugin for Presence B2C.
Processes 16 kHz mono audio waveforms using YAMNet ONNX acoustic classifier.
Maintains sliding score confidence windows, identifies baby crying and raised voices,
and groups cry episodes with state-machine deduplication.
"""

from collections import deque
from datetime import datetime, timezone
import logging
import os
import time
from typing import Any, Dict, List, Optional
import numpy as np

from ai.cctv_ai.objects import TrackedObject
from ai.cctv_ai.plugin import BasePlugin
from packages.event_schemas.events import StandardAIEvent, StandardEventType, EventSeverity

logger = logging.getLogger(__name__)


class BabyCryingPlugin(BasePlugin):
    """Plugin for infant crying and acoustic event classification using YAMNet ONNX."""

    # AudioSet indices relevant to baby cry & acoustic distress
    CLASS_SPEECH = 0
    CLASS_SHOUT = 10
    CLASS_SCREAMING = 11
    CLASS_CRYING_SOBBING = 20
    CLASS_BABY_CRY = 21

    def __init__(
        self,
        model_path: Optional[str] = None,
        energy_rms_threshold: float = 0.02,
        cry_score_threshold: float = 0.40,
        continuation_interval_sec: float = 10.0,
        silence_timeout_sec: float = 8.0
    ):
        super().__init__(plugin_id="baby_crying", name="Baby Crying Monitor", enabled=True)

        # Resolve model path (supports use-case name baby_crying.onnx and legacy yamnet.onnx)
        default_paths = [
            model_path,
            "e:/cctv/ai/models/baby_crying.onnx",
            "ai/models/baby_crying.onnx",
            "e:/cctv/backend/models/baby_crying.onnx",
            "backend/models/baby_crying.onnx",
            "e:/cctv/backend/models/yamnet.onnx",
            "backend/models/yamnet.onnx",
            "e:/cctv/ai/models/yamnet.onnx",
            "ai/models/yamnet.onnx",
            r"C:\Users\HP\Downloads\yamnet.onnx"
        ]
        resolved_path = None
        for p in default_paths:
            if p and os.path.exists(p):
                resolved_path = os.path.abspath(p)
                break

        self.model_path = resolved_path
        self.session = None
        self.backend = "spectral_heuristic_fallback"
        self.model_version = "yamnet_onnx_1.0.0"
        self.energy_rms_threshold = energy_rms_threshold
        self.cry_score_threshold = cry_score_threshold
        self.continuation_interval_sec = continuation_interval_sec
        self.silence_timeout_sec = silence_timeout_sec
        self.latency_ms = 0.0

        # Episode state machine per camera: camera_id -> dict
        self.episodes: Dict[str, dict] = {}

        self._init_backend()

    def _init_backend(self) -> None:
        if self.model_path and os.path.exists(self.model_path):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                self.session = ort.InferenceSession(
                    self.model_path,
                    opts,
                    providers=["CPUExecutionProvider"]
                )
                self.input_name = self.session.get_inputs()[0].name
                self.backend = "onnx_yamnet"
                self.model_version = "yamnet_onnx_1.0.0"
                logger.info(f"Loaded YAMNet ONNX acoustic model from: {self.model_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load YAMNet ONNX model: {e}")

        logger.info("Using acoustic spectral harmonic fallback for baby crying.")

    def _get_episode(self, camera: str) -> dict:
        if camera not in self.episodes:
            self.episodes[camera] = {
                "in_episode": False,
                "episode_start": None,
                "last_cry_time": None,
                "last_continuation_alert": None,
                "sliding_scores": deque(maxlen=5),
            }
        return self.episodes[camera]

    def on_audio(
        self,
        camera: str,
        samples: np.ndarray,
        timestamp: datetime
    ) -> List[StandardAIEvent]:
        events: List[StandardAIEvent] = []
        if not self.enabled or samples is None or len(samples) == 0:
            return events

        episode = self._get_episode(camera)

        # 1. Acoustic RMS Energy
        samples_f32 = samples.astype(np.float32)
        rms = float(np.sqrt(np.mean(samples_f32 ** 2)))

        cry_score = 0.0
        shout_score = 0.0

        if rms >= self.energy_rms_threshold:
            # Check test override
            test_score = self.config.get("test_override_cry_score")
            if test_score is not None:
                cry_score = float(test_score)
            elif self.backend == "onnx_yamnet" and self.session is not None:
                # 2. Run YAMNet ONNX Model
                t0 = time.time()
                try:
                    # YAMNet accepts 1D waveform [num_samples]
                    outputs = self.session.run(None, {self.input_name: samples_f32})
                    scores = outputs[0]  # shape (N, 521)
                    if scores.ndim >= 2:
                        # Extract maximum probability of Crying / Baby Cry classes
                        cry_indices = [self.CLASS_CRYING_SOBBING, self.CLASS_BABY_CRY]
                        cry_score = float(np.max(scores[:, cry_indices]))

                        # Extract Shout / Screaming classes
                        shout_indices = [self.CLASS_SHOUT, self.CLASS_SCREAMING]
                        shout_score = float(np.max(scores[:, shout_indices]))
                    self.latency_ms = round((time.time() - t0) * 1000, 2)
                except Exception as e:
                    logger.debug(f"YAMNet inference error: {e}")
                    cry_score = self._spectral_fallback(samples_f32)
            else:
                # 3. Spectral heuristic fallback
                cry_score = self._spectral_fallback(samples_f32)

        episode["sliding_scores"].append(cry_score)
        smoothed_score = float(np.mean(episode["sliding_scores"]))
        is_crying = (smoothed_score >= self.cry_score_threshold)

        now = timestamp

        if is_crying:
            episode["last_cry_time"] = now
            with_raised_voices = (shout_score >= 0.30)

            if not episode["in_episode"]:
                # New cry episode begins
                episode["in_episode"] = True
                episode["episode_start"] = now
                episode["last_continuation_alert"] = now

                event_type = (
                    StandardEventType.BABY_CRY_WITH_RAISED_VOICES.value
                    if with_raised_voices
                    else StandardEventType.BABY_CRY_STARTED.value
                )

                events.append(StandardAIEvent(
                    household_id=self.config.get("household_id", "household_default"),
                    camera_id=camera,
                    module_id=self.plugin_id,
                    event_type=event_type,
                    timestamp=now,
                    start_time=now,
                    confidence=round(smoothed_score, 3),
                    evidence_type="audio_waveform",
                    model_version=self.model_version,
                    severity=EventSeverity.HIGH,
                    metadata={
                        "rms_energy": round(rms, 4),
                        "cry_confidence": round(smoothed_score, 2),
                        "shout_confidence": round(shout_score, 2),
                        "model": "YAMNet ONNX",
                        "note": "Acoustic infant cry pattern detected",
                    },
                    monitoring_uncertainty=round(1.0 - smoothed_score, 2)
                ))
            else:
                # Ongoing episode: check continuation interval
                time_since_last_alert = (now - episode["last_continuation_alert"]).total_seconds()
                if time_since_last_alert >= self.continuation_interval_sec:
                    episode["last_continuation_alert"] = now
                    total_dur = (now - episode["episode_start"]).total_seconds()
                    events.append(StandardAIEvent(
                        household_id=self.config.get("household_id", "household_default"),
                        camera_id=camera,
                        module_id=self.plugin_id,
                        event_type=StandardEventType.BABY_CRY_CONTINUES.value,
                        timestamp=now,
                        start_time=episode["episode_start"],
                        confidence=round(smoothed_score, 3),
                        evidence_type="audio_waveform",
                        model_version=self.model_version,
                        severity=EventSeverity.HIGH,
                        metadata={
                            "total_duration_seconds": round(total_dur, 1),
                            "cry_confidence": round(smoothed_score, 2),
                            "model": "YAMNet ONNX",
                        },
                        monitoring_uncertainty=round(1.0 - smoothed_score, 2)
                    ))
        else:
            # Silence or non-cry sound: check timeout
            if episode["in_episode"] and episode["last_cry_time"]:
                silence_duration = (now - episode["last_cry_time"]).total_seconds()
                if silence_duration >= self.silence_timeout_sec:
                    total_dur = (episode["last_cry_time"] - episode["episode_start"]).total_seconds()
                    episode["in_episode"] = False

                    events.append(StandardAIEvent(
                        household_id=self.config.get("household_id", "household_default"),
                        camera_id=camera,
                        module_id=self.plugin_id,
                        event_type=StandardEventType.BABY_CRY_ENDED.value,
                        timestamp=now,
                        start_time=episode["episode_start"],
                        end_time=episode["last_cry_time"],
                        confidence=0.90,
                        evidence_type="audio_waveform",
                        model_version=self.model_version,
                        severity=EventSeverity.INFO,
                        metadata={
                            "total_duration_seconds": round(total_dur, 1),
                            "model": "YAMNet ONNX",
                        },
                        monitoring_uncertainty=0.10
                    ))

        return events

    def _spectral_fallback(self, samples: np.ndarray) -> float:
        """Spectral fallback calculating energy ratio in typical infant cry harmonics (350-2500 Hz)."""
        fft = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(len(samples), 1.0 / 16000.0)
        cry_band = (freqs >= 350) & (freqs <= 2500)
        cry_energy = np.sum(fft[cry_band])
        total_energy = np.sum(fft) + 1e-7
        return float(min(1.0, (cry_energy / total_energy) * 1.4))
