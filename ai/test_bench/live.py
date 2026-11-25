"""Live RTSP ingestion + inference for the vision test bench.

Two background threads keep the live path honest about latency:

* the **reader** opens the source once and keeps only the newest decoded
  frame. Old frames are dropped instead of queued, so a slow inference step
  can never make the picture drift further and further behind the camera.
* the **worker** runs the same :class:`VisionPipeline` used by the upload and
  webcam paths on that newest frame, at a capped inference rate, and stores
  the JPEG + detections for the HTTP layer to hand out.

Plate OCR costs hundreds of milliseconds to seconds per full frame on a laptop
CPU, while face recognition costs tens of milliseconds. The worker therefore
runs plate OCR every ``plate_every`` frames and reuses the last plate result in
between, so faces stay smooth without hiding plates from the overlay.

The HTTP handler never runs inference: it only copies the latest published
result. Several browser tabs can poll the same session for free, and a slow
client cannot slow the model down.

RTSP is read through OpenCV's bundled FFmpeg. RTP over TCP is requested by
default because UDP loses packets on most Wi-Fi links; set
``OPENCV_FFMPEG_CAPTURE_OPTIONS`` before launching the bench to override it.
"""

from __future__ import annotations

import base64
import logging
import math
import os
import re
import threading
import time
import urllib.parse
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

log = logging.getLogger("cctv_ai.live")

SOURCE_SCHEMES = {"rtsp", "rtsps", "http", "https"}
MODES = {"both", "faces", "plates"}

# A dead camera should not block the reader forever: cap the socket I/O.
# OpenCV reads this environment variable every time it opens a capture.
DEFAULT_CAPTURE_OPTIONS = "rtsp_transport;tcp|timeout;5000000|fflags;nobuffer|max_delay;500000"

# Some cameras (Hikvision/CP Plus style) repeat the password inside the URL
# path: .../user=admin_password=SECRET_channel=0_stream=0&onvif=0.sdp
INLINE_PASSWORD_RE = re.compile(r"(?i)((?:password|passwd|pwd)=)([^&/\s]*)")


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without opening a camera)
# ---------------------------------------------------------------------------
def normalise_source(raw: str) -> str:
    """Validate a live source URL and return it trimmed.

    Accepts RTSP/RTSP-over-TLS and HTTP(S) (go2rtc's restream or an MJPEG
    URL), because all of them go through the same OpenCV capture path.
    """
    source = (raw or "").strip()
    if not source:
        raise ValueError("RTSP URL is required")
    if any(ch.isspace() for ch in source):
        raise ValueError("URL must not contain spaces (percent-encode credentials)")
    if "<" in source or ">" in source:
        raise ValueError("URL still contains a <placeholder>; replace it with the real value")
    try:
        parsed = urllib.parse.urlsplit(source)
    except ValueError as err:
        raise ValueError(f"Invalid URL: {err}") from err
    scheme = parsed.scheme.lower()
    if scheme not in SOURCE_SCHEMES:
        raise ValueError("Unsupported URL scheme: use rtsp://, rtsps://, http:// or https://")
    if not parsed.hostname:
        raise ValueError("URL is missing a host name")
    return source


def _mask_inline_passwords(text: str) -> str:
    """Redact ``password=...`` in a path/query, keeping ``_channel=...`` intact."""

    def redact(match: re.Match[str]) -> str:
        value = match.group(2)
        suffix = ""
        marker = value.lower().find("_channel=")
        if marker >= 0:  # an empty password still leaves the marker at index 0
            suffix = value[marker:]
            value = value[:marker]
        return match.group(1) + ("***" if value else "") + suffix

    return INLINE_PASSWORD_RE.sub(redact, text)


def mask_source(source: str) -> str:
    """Hide every password in a camera URL for logs and the UI.

    Covers both ``rtsp://user:pass@host/...`` and the Hikvision-style path that
    repeats the password as ``..._password=pass_channel=...``. Camera
    credentials must never reach the browser, an event log or a screenshot of
    the bench.
    """
    try:
        parsed = urllib.parse.urlsplit(source)
    except ValueError:
        return "***"
    if not parsed.password:
        return _mask_inline_passwords(source)
    host = parsed.hostname or ""
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    netloc = f"{parsed.username or ''}:***@{host}"
    if parsed.port:
        netloc += f":{parsed.port}"
    masked = urllib.parse.urlunsplit(
        (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
    )
    return _mask_inline_passwords(masked)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _number(payload: dict[str, Any], key: str, default: float,
            low: float, high: float, cast=float) -> float:
    try:
        return _clamp(cast(payload.get(key, default)), low, high)
    except (TypeError, ValueError):
        return default


def _flag(payload: dict[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class EventCooldown:
    """Suppress duplicate live events (same key) for N seconds.

    At 5 FPS the same known face would otherwise be reported five times a
    second. The cooldown keeps the event feed readable, like the per-use-case
    cooldowns in the real worker.
    """

    def __init__(self, seconds: float = 15.0) -> None:
        self.seconds = float(seconds)
        self._last: dict[str, float] = {}

    def ready(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else float(now)
        last = self._last.get(key)
        if last is not None and now - last < self.seconds:
            return False
        self._last[key] = now
        return True

    def reset(self) -> None:
        self._last.clear()


@dataclass(frozen=True)
class LiveTuning:
    """Everything the live loop can change without reconnecting."""

    mode: str = "both"
    contrast_boost: bool = False
    min_face_area: int = 750
    face_det_thresh: float = 0.6
    face_rec_thresh: float = 0.8
    min_plate_area: int = 1000
    min_plate_conf: float = 0.8
    infer_fps: float = 5.0
    max_width: int = 1280
    jpeg_quality: int = 80
    event_cooldown: float = 15.0
    # Full-frame RapidOCR is slow (hundreds of ms to seconds on a laptop CPU).
    # Scan plates on every Nth inference instead of every frame; faces still run
    # every frame.
    plate_every: int = 5

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None, base: "LiveTuning | None" = None) -> "LiveTuning":
        """Build a clamped tuning from untrusted JSON (sliders, CLI, API)."""
        payload = payload or {}
        base = base or cls()
        mode = str(payload.get("mode", base.mode)).lower()
        if mode not in MODES:
            mode = base.mode
        fps_key = "fps" if "fps" in payload else "infer_fps"
        return cls(
            mode=mode,
            contrast_boost=_flag(payload, "contrast_boost", base.contrast_boost),
            min_face_area=int(_number(payload, "min_face_area", base.min_face_area, 0, 200_000, int)),
            face_det_thresh=_number(payload, "face_det_thresh", base.face_det_thresh, 0.05, 0.99),
            face_rec_thresh=_number(payload, "face_rec_thresh", base.face_rec_thresh, 0.05, 0.99),
            min_plate_area=int(_number(payload, "min_plate_area", base.min_plate_area, 0, 200_000, int)),
            min_plate_conf=_number(payload, "min_plate_conf", base.min_plate_conf, 0.05, 0.99),
            infer_fps=_number(payload, fps_key, base.infer_fps, 0.2, 30.0),
            max_width=int(_number(payload, "max_width", base.max_width, 320, 3840, int)),
            jpeg_quality=int(_number(payload, "jpeg_quality", base.jpeg_quality, 30, 95, int)),
            event_cooldown=_number(payload, "event_cooldown", base.event_cooldown, 0.0, 600.0),
            plate_every=int(_number(payload, "plate_every", base.plate_every, 1, 30, int)),
        )


# ---------------------------------------------------------------------------
# The live session
# ---------------------------------------------------------------------------
class LiveStream:
    """One RTSP source, one reader, one inference loop, one latest result."""

    def __init__(self, pipeline: Any, source: str, tuning: LiveTuning | None = None) -> None:
        self.pipeline = pipeline
        self.source = normalise_source(source)
        self.display_source = mask_source(self.source)

        self._lock = threading.RLock()
        self._config_lock = threading.Lock()
        self._stop = threading.Event()
        self._tuning = tuning or LiveTuning()
        self._cooldown = EventCooldown(self._tuning.event_cooldown)

        # newest decoded frame, written by the reader, read by the worker
        self._latest: np.ndarray | None = None
        self._latest_at = 0.0
        self._read_seq = 0

        # latest published result, read by the HTTP layer
        self._result: dict[str, Any] | None = None
        self._seq = 0
        self._events: deque[dict[str, Any]] = deque(maxlen=30)

        self.status = "stopped"
        self.last_error: str | None = None
        self._read_times: deque[float] = deque(maxlen=32)
        self._process_times: deque[float] = deque(maxlen=20)
        self._frames_read = 0
        self._frames_processed = 0
        self._capture_lag_ms = 0.0
        self._inference_ms = 0.0

        # Dedicated background worker cache for Plate OCR
        self._plate_lock = threading.Lock()
        self._cached_plates: list[dict[str, Any]] = []
        self._cached_plate_events: list[dict[str, Any]] = []
        self._cached_plate_at = 0.0
        self._plate_inference_ms = 0.0

        self._reader: threading.Thread | None = None
        self._worker: threading.Thread | None = None
        self._plate_worker: threading.Thread | None = None

    # -- configuration -----------------------------------------------------
    @property
    def tuning(self) -> LiveTuning:
        with self._config_lock:
            return self._tuning

    def update_tuning(self, payload: dict[str, Any]) -> LiveTuning:
        """Apply slider changes to the running loop without reopening the camera."""
        with self._config_lock:
            new = LiveTuning.from_payload(payload, self._tuning)
            if new.event_cooldown != self._tuning.event_cooldown:
                self._cooldown = EventCooldown(new.event_cooldown)
            self._tuning = new
            return new

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._reader is not None and self._reader.is_alive():
            return
        self._stop.clear()
        self.status = "connecting"
        self.last_error = None
        self._reader = threading.Thread(target=self._read_loop, name="live-reader", daemon=True)
        self._worker = threading.Thread(target=self._work_loop, name="live-inference", daemon=True)
        self._plate_worker = threading.Thread(target=self._plate_loop, name="live-plates", daemon=True)
        self._reader.start()
        self._worker.start()
        self._plate_worker.start()

    def stop(self) -> None:
        self._stop.set()
        for thread in (self._reader, self._worker, self._plate_worker):
            if thread is not None and thread.is_alive():
                # A blocked RTSP read may outlive the join; daemon threads die
                # with the process and the stop flag makes them exit silently.
                thread.join(timeout=1.5)
        self.status = "stopped"

    # -- reader ------------------------------------------------------------
    def _read_loop(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            cap = self._open_capture()
            if cap is None:
                self._set_state("reconnecting", self.last_error or "could not open source")
                self._stop.wait(backoff)
                backoff = min(backoff * 2.0, 10.0)
                continue

            backoff = 1.0
            self._set_state("live", None)
            try:
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        self._set_state("reconnecting", "no frames from source")
                        break
                    now = time.monotonic()
                    with self._lock:
                        # Keep only the newest frame: this is the whole point of
                        # the reader thread. No queue, no growing latency.
                        self._latest = frame
                        self._latest_at = now
                        self._read_seq += 1
                        self._read_times.append(now)
                        self._frames_read += 1
            except Exception as err:  # noqa: BLE001 - keep the reader alive
                log.warning("live read failed: %s", err)
                self._set_state("reconnecting", "read failed (see server log)")
            finally:
                cap.release()

            if not self._stop.is_set():
                self._stop.wait(1.0)
        with self._lock:
            self.status = "stopped"

    def _open_capture(self) -> cv2.VideoCapture | None:
        if self.source.lower().startswith(("rtsp://", "rtsps://")):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", DEFAULT_CAPTURE_OPTIONS)
        self._set_state("connecting", None)
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            # CAP_FFMPEG is not available in every OpenCV build; let OpenCV
            # choose before giving up.
            cap.release()
            cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            self.last_error = f"could not open {self.display_source}"
            return None
        # Ask FFmpeg to keep at most one frame buffered. The reader thread is
        # the real queue, so this only trims decoder latency.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        log.info("live source connected: %s", self.display_source)
        return cap

    def _set_state(self, status: str, error: str | None) -> None:
        if self._stop.is_set():
            return
        with self._lock:
            self.status = status
            self.last_error = error

    # -- inference ---------------------------------------------------------
    def _plate_loop(self) -> None:
        """Asynchronous background worker for heavy Plate OCR.

        RapidOCR takes 600-1000 ms on CPU. By running it on a separate
        thread, face recognition and video display never stutter and can
        sustain 10-15 FPS with minimal latency.
        """
        last_seq = 0
        while not self._stop.is_set():
            tuning = self.tuning
            if tuning.mode not in ("plates", "both"):
                self._stop.wait(0.2)
                continue

            frame, captured_at, read_seq = self._take_frame(last_seq)
            if frame is None:
                self._stop.wait(0.04)
                continue
            last_seq = read_seq

            height, width = frame.shape[:2]
            if width > tuning.max_width:
                scale = tuning.max_width / float(width)
                frame = cv2.resize(
                    frame,
                    (tuning.max_width, max(1, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )

            try:
                t0 = time.monotonic()
                if hasattr(self.pipeline, "process_plates_bgr"):
                    result = self.pipeline.process_plates_bgr(
                        frame,
                        contrast_boost=tuning.contrast_boost,
                        min_plate_area=tuning.min_plate_area,
                        min_plate_conf=tuning.min_plate_conf,
                    )
                else:
                    result = self.pipeline.process_bgr(
                        frame,
                        mode="plates",
                        contrast_boost=tuning.contrast_boost,
                        min_plate_area=tuning.min_plate_area,
                        min_plate_conf=tuning.min_plate_conf,
                    )
                cost_ms = (time.monotonic() - t0) * 1000.0
                with self._plate_lock:
                    self._cached_plates = [asdict(p) for p in result.plates]
                    self._cached_plate_events.extend(result.events)
                    self._cached_plate_at = time.monotonic()
                    self._plate_inference_ms = cost_ms
            except Exception:
                log.exception("async plate OCR failed")

            cadence = max(1.5, (1.0 / max(tuning.infer_fps, 0.5)) * max(1, tuning.plate_every))
            self._stop.wait(cadence)

    def _work_loop(self) -> None:
        """Fast face recognition and display publisher loop (~30ms per frame)."""
        last_seq = 0
        while not self._stop.is_set():
            frame, captured_at, read_seq = self._take_frame(last_seq)
            if frame is None:
                self._stop.wait(0.015)
                continue
            last_seq = read_seq
            tuning = self.tuning
            started = time.monotonic()
            capture_lag_ms = max(0.0, (started - captured_at) * 1000.0)

            # Cap the transport/inference width. All boxes stay in this space,
            # so the browser overlay and the JPEG always agree.
            height, width = frame.shape[:2]
            if width > tuning.max_width:
                scale = tuning.max_width / float(width)
                frame = cv2.resize(
                    frame,
                    (tuning.max_width, max(1, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
                height, width = frame.shape[:2]

            faces: list[dict[str, Any]] = []
            events: list[dict[str, Any]] = []
            telemetry: dict[str, float] = {}

            if tuning.mode in ("faces", "both"):
                try:
                    if hasattr(self.pipeline, "process_faces_bgr"):
                        result = self.pipeline.process_faces_bgr(
                            frame,
                            contrast_boost=tuning.contrast_boost,
                            min_face_area=tuning.min_face_area,
                            face_det_thresh=tuning.face_det_thresh,
                            face_rec_thresh=tuning.face_rec_thresh,
                        )
                    else:
                        result = self.pipeline.process_bgr(
                            frame,
                            mode="faces",
                            contrast_boost=tuning.contrast_boost,
                            min_face_area=tuning.min_face_area,
                            face_det_thresh=tuning.face_det_thresh,
                            face_rec_thresh=tuning.face_rec_thresh,
                        )
                    faces = [asdict(face) for face in result.faces]
                    events.extend(result.events)
                    telemetry.update(result.telemetry)
                except Exception:
                    log.exception("live face inference failed")
                    self._set_state("reconnecting", "inference failed (see server log)")
                    self._stop.wait(0.5)
                    continue

            plate_hits: list[dict[str, Any]] = []
            if tuning.mode in ("plates", "both"):
                with self._plate_lock:
                    plate_age_s = time.monotonic() - self._cached_plate_at
                    # Keep plate detection box visible for 5 seconds after scan
                    if self._cached_plate_at > 0 and plate_age_s < 5.0:
                        plate_hits = list(self._cached_plates)
                        telemetry["live_plate_age_ms"] = round(plate_age_s * 1000.0, 1)
                    if self._cached_plate_events:
                        events.extend(self._cached_plate_events)
                        self._cached_plate_events.clear()
                    if self._plate_inference_ms > 0:
                        telemetry["ocr_inference_ms"] = round(self._plate_inference_ms, 1)

            ok, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, tuning.jpeg_quality]
            )
            if not ok:
                continue

            telemetry["live_capture_lag_ms"] = round(capture_lag_ms, 1)
            telemetry["live_inference_ms"] = round((time.monotonic() - started) * 1000.0, 1)
            telemetry["total_ms"] = telemetry["live_inference_ms"]

            payload = {
                "width": width,
                "height": height,
                "faces": faces,
                "plates": plate_hits,
                "telemetry": telemetry,
                "events": self._dedupe(events),
                "image": "data:image/jpeg;base64,"
                + base64.b64encode(buffer.tobytes()).decode("ascii"),
            }
            finished = time.monotonic()
            self._publish(
                payload,
                capture_lag_ms=capture_lag_ms,
                inference_ms=telemetry["live_inference_ms"],
                finished=finished,
            )

            interval = 1.0 / max(tuning.infer_fps, 0.2)
            self._stop.wait(max(0.001, interval - (time.monotonic() - started)))

    def _publish(self, payload: dict[str, Any], capture_lag_ms: float,
                 inference_ms: float, finished: float) -> None:
        """Store one finished inference result (called by the worker, or a test)."""
        with self._lock:
            self._seq += 1
            self._result = payload
            self._frames_processed += 1
            self._capture_lag_ms = capture_lag_ms
            self._inference_ms = inference_ms
            self._process_times.append(finished)
            for event in payload["events"]:
                self._events.appendleft(event)

    def _take_frame(self, last_seq: int) -> tuple[np.ndarray | None, float, int]:
        with self._lock:
            if self._latest is None or self._read_seq == last_seq:
                return None, 0.0, last_seq
            return self._latest, self._latest_at, self._read_seq

    @staticmethod
    def _plate_due(frame_count: int, every: int) -> bool:
        """True on the frames where the expensive plate OCR should run."""
        return frame_count % max(1, int(every)) == 0

    def _dedupe(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for event in events:
            kind = "plate" if event.get("type") == "vehicle_plate" else "face"
            key = f"{kind}:{event.get('label')}"
            if self._cooldown.ready(key):
                kept.append(dict(event))
        return kept

    # -- HTTP view ---------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "source_fps": self._rate(self._read_times),
                "inference_fps": self._rate(self._process_times),
                "frames_read": self._frames_read,
                "frames_processed": self._frames_processed,
                "frames_skipped": max(0, self._frames_read - self._frames_processed),
                "capture_lag_ms": round(self._capture_lag_ms, 1),
                "inference_ms": round(self._inference_ms, 1),
            }

    @staticmethod
    def _rate(times: deque[float]) -> float:
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        if span <= 0 or math.isnan(span):
            return 0.0
        return round((len(times) - 1) / span, 1)

    def status_view(self) -> dict[str, Any]:
        with self._lock:
            status = self.status
            error = self.last_error
            seq = self._seq
        view: dict[str, Any] = {
            "status": status,
            "seq": seq,
            "source": self.display_source,
            "stats": self.stats(),
        }
        if error:
            view["error"] = error
        return view

    def snapshot(self, after: int = 0) -> dict[str, Any]:
        """Latest result for the browser, or a tiny 'unchanged' reply.

        Polling at 10 Hz must be cheap: when no new inference has finished we
        return only status/stats, never the JPEG again.
        """
        with self._lock:
            result = self._result
            seq = self._seq
            events = list(self._events)
        head = self.status_view()
        head["seq"] = seq
        if result is None or seq <= after:
            return {**head, "unchanged": True}
        return {
            **result,
            **head,
            "unchanged": False,
            # `events` is the deduped feed history, not just this frame
            "events": events,
        }
