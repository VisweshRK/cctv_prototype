"""Baby-crying audio detector for main's 16 kHz mono plugin contract."""
from __future__ import annotations
from collections import deque
from datetime import datetime
from pathlib import Path
import logging
import time
import numpy as np
from cctv_ai.plugin import Event, Plugin as Base
log = logging.getLogger("cctv_ai.baby_crying")

class Plugin(Base):
    use_case = "baby_crying"
    enabled = True
    needs = {"audio"}
    interval = 1.0
    heavy = True
    def __init__(self):
        self.episodes: dict[str, dict] = {}
        self.session = None
        self.input_name = None
        self.model_version = "harmonic-fallback"
        self.cry_threshold = 0.40
        self.silence_timeout = 8.0
        self.continuation_interval = 10.0
    def setup(self, cameras: list[str]) -> None:
        self.episodes = {}
        root = Path(__file__).resolve().parents[2] / "models"
        for path in (root / "yamnet.onnx", root / "baby_crying.onnx"):
            if not path.exists():
                continue
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions(); opts.intra_op_num_threads = 2
                self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
                self.input_name = self.session.get_inputs()[0].name
                self.model_version = "yamnet-onnx"
                break
            except Exception as exc:
                log.warning("could not load %s: %s", path, exc)
    def _state(self, camera: str) -> dict:
        return self.episodes.setdefault(camera, {"scores": deque(maxlen=5), "active": False, "start": None, "last": None, "alert": None})
    def _score(self, samples: np.ndarray, sample_rate: int) -> tuple[float, float]:
        x = samples.astype(np.float32) / (32768.0 if np.issubdtype(samples.dtype, np.integer) else 1.0)
        rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
        if self.session is not None and self.input_name:
            try:
                out = self.session.run(None, {self.input_name: x})[0]
                if out.ndim >= 2 and out.shape[-1] >= 22:
                    return float(np.max(out[..., [20, 21]])), rms
            except Exception:
                pass
        if rms < 0.02 or x.size < 8:
            return 0.0, rms
        spectrum = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(x.size, 1 / sample_rate)
        band = (freqs >= 350) & (freqs <= 2500)
        score = min(1.0, float(spectrum[band].sum() / (spectrum.sum() + 1e-7)) * 1.5)
        return score, rms
    def on_audio(self, camera: str, samples: np.ndarray, sample_rate: int, ts: datetime) -> list[Event]:
        if samples is None or len(samples) == 0 or sample_rate <= 0:
            return []
        st = self._state(camera)
        score, rms = self._score(samples, sample_rate)
        st["scores"].append(score)
        smooth = float(np.mean(st["scores"]))
        crying = smooth >= self.cry_threshold
        if crying:
            st["last"] = ts
            if not st["active"]:
                st.update(active=True, start=ts, alert=ts)
                return [self.event(camera, "crying", min(1.0, smooth), time=ts, data={"area": "Audio", "detail": "Baby crying detected", "rms": round(rms, 4), "model": self.model_version})]
            if st["alert"] and (ts - st["alert"]).total_seconds() >= self.continuation_interval:
                st["alert"] = ts
                duration = (ts - st["start"]).total_seconds()
                return [self.event(camera, "crying_continues", min(1.0, smooth), time=ts, data={"area": "Audio", "detail": f"Crying continues for {duration:.0f}s", "seconds": round(duration, 1)})]
            return []
        if st["active"] and st["last"] and (ts - st["last"]).total_seconds() >= self.silence_timeout:
            duration = (st["last"] - st["start"]).total_seconds()
            st.update(active=False, start=None, last=None, alert=None)
            return [self.event(camera, "crying_ended", 0.9, time=ts, data={"area": "Audio", "detail": "Crying episode ended", "seconds": round(duration, 1)})]
        return []
