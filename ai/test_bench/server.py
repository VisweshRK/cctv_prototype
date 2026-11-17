"""HTTP Server for CCTV Prototype Vision Test Bench.

Runs on Python's built-in http.server (no extra pip dependencies required!).
Serves modern web UI and provides REST endpoints for face recognition,
vehicle license plate recognition, live webcam streaming, gallery management,
and video analysis.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import shutil
import sys
import threading
import time
import urllib.parse
from dataclasses import asdict
from http import HTTPStatus
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Ensure ai directory is on sys.path so plugins import cleanly
CURRENT_DIR = Path(__file__).resolve().parent
AI_DIR = CURRENT_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from test_bench.live import LiveStream, LiveTuning, normalise_source
from test_bench.runner import VisionPipeline, get_repo_root
from plugins.face_recognition import build_gallery

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cctv_ai.server")

WEB_DIR = CURRENT_DIR / "web"
pipeline: VisionPipeline | None = None
live: LiveStream | None = None
live_lock = threading.Lock()          # protects the `live` global itself
live_control = threading.RLock()      # serialises start/stop of the source


def get_live() -> LiveStream | None:
    with live_lock:
        return live


def stop_live() -> None:
    """Stop the running live session, if any (safe to call repeatedly)."""
    global live
    with live_control:
        with live_lock:
            session, live = live, None
        if session is not None:
            session.stop()
            log.info("live session stopped")


def start_live(source: str, tuning: LiveTuning) -> LiveStream:
    """Replace any running live source with a fresh session."""
    global live
    assert pipeline is not None
    with live_control:
        stop_live()
        session = LiveStream(pipeline, source, tuning)
        session.start()
        with live_lock:
            live = session
        return session


def update_live(payload: dict[str, Any]) -> LiveTuning | None:
    with live_lock:
        session = live
    if session is None:
        return None
    return session.update_tuning(payload)


def decode_base64_image(b64_data: str) -> np.ndarray | None:
    """Decode data:image/...;base64,string into BGR numpy array."""
    if not b64_data:
        return None
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]
    raw = base64.b64decode(b64_data)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_image_base64(bgr: np.ndarray, ext: str = ".jpg") -> str:
    """Encode BGR image to base64 data URL."""
    _, buf = cv2.imencode(ext, bgr)
    b64 = base64.b64encode(buf).decode("utf-8")
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


class TestBenchHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, format, *args):
        try:
            if args and isinstance(args[0], str) and "/api/process/frame" in args[0]:
                return
            log.info("%s - %s", self.address_string(), format % args)
        except Exception:
            pass

    def send_json(self, data: Any, status: int = 200) -> None:
        payload = json.dumps(data, indent=2, default=json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        # Live frames are polled continuously; a cached reply would freeze the
        # viewport on an old JPEG.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/api/health":
            self.send_json({"status": "healthy"})
            return

        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        if path == "/api/status":
            assert pipeline is not None
            faces_npz = pipeline.root / "data" / "faces.npz"
            names = list(pipeline.face_engine.names) if pipeline.face_engine.names else []
            self.send_json({
                "status": "ready",
                "models": {
                    "yunet_loaded": pipeline.face_engine.detector is not None,
                    "sface_loaded": pipeline.face_engine.recognizer is not None,
                    "rapidocr_loaded": pipeline.ocr.engine is not None,
                },
                "gallery": {
                    "exists": faces_npz.exists(),
                    "people_count": len(names),
                    "names": names,
                },
                "known_plates_count": len(pipeline.known_plates),
            })
            return

        if path == "/api/samples":
            assert pipeline is not None
            samples: list[dict[str, str]] = []
            test_img_dir = pipeline.root / "ai" / "tests" / "test_images"
            if test_img_dir.exists():
                for f in sorted(test_img_dir.iterdir()):
                    if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        samples.append({"name": f.name, "type": "image", "url": f"/api/sample?name={f.name}"})

            samples_dir = pipeline.root / "samples"
            if samples_dir.exists():
                for f in sorted(samples_dir.iterdir()):
                    if f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
                        samples.append({"name": f.name, "type": "video", "url": f"/api/sample?name={f.name}"})

            self.send_json({"samples": samples})
            return

        if path == "/api/sample":
            assert pipeline is not None
            name = query.get("name", [""])[0]
            if not name or ".." in name or "/" in name or "\\" in name:
                self.send_error(400, "Invalid sample name")
                return

            candidates = [
                pipeline.root / "ai" / "tests" / "test_images" / name,
                pipeline.root / "samples" / name,
                pipeline.root / "data" / name,
            ]
            file_path = next((p for p in candidates if p.exists() and p.is_file()), None)
            if not file_path:
                self.send_error(404, f"Sample {name} not found")
                return

            mime_type, _ = mimetypes.guess_type(str(file_path))
            mime_type = mime_type or "application/octet-stream"
            content = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
            return

        if path == "/api/gallery":
            assert pipeline is not None
            gallery_dir = pipeline.root / "ai" / "plugins" / "face_recognition" / "faces"
            people_info: list[dict[str, Any]] = []
            if gallery_dir.exists():
                for person in sorted(p for p in gallery_dir.iterdir() if p.is_dir()):
                    photos = [
                        p.name for p in person.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file()
                    ]
                    people_info.append({
                        "name": person.name,
                        "photo_count": len(photos),
                        "photos": photos,
                    })

            self.send_json({
                "people": people_info,
                "enrolled_in_model": list(pipeline.face_engine.names) if pipeline.face_engine.names else [],
            })
            return

        if path == "/api/plates":
            assert pipeline is not None
            self.send_json({"plates": pipeline.known_plates})
            return

        if path == "/api/live/status":
            session = get_live()
            if session is None:
                self.send_json({"status": "stopped", "seq": 0, "stats": {}})
            else:
                self.send_json(session.status_view())
            return

        if path == "/api/live/frame":
            session = get_live()
            if session is None:
                self.send_json({"status": "stopped", "seq": 0, "stats": {}, "unchanged": True})
                return
            try:
                after = int(query.get("after", ["0"])[0])
            except (TypeError, ValueError):
                after = 0
            self.send_json(session.snapshot(after=max(0, after)))
            return

        # Serve static web frontend
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        if path in ("/api/process/image", "/api/process/frame"):
            assert pipeline is not None
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception as e:
                self.send_json({"error": f"Invalid JSON payload: {e}"}, 400)
                return

            img_b64 = data.get("image", "")
            bgr = decode_base64_image(img_b64)
            if bgr is None:
                self.send_json({"error": "Failed to decode base64 image"}, 400)
                return

            mode = data.get("mode", "both")
            contrast_boost = bool(data.get("contrast_boost", False))
            min_face_area = int(data.get("min_face_area", 750))
            face_det_thresh = float(data.get("face_det_thresh", 0.6))
            face_rec_thresh = float(data.get("face_rec_thresh", 0.8))
            min_plate_area = int(data.get("min_plate_area", 1000))
            min_plate_conf = float(data.get("min_plate_conf", 0.80))
            min_face_area = int(data.get("min_face_area", 750))
            face_det_thresh = float(data.get("face_det_thresh", 0.6))
            face_rec_thresh = float(data.get("face_rec_thresh", 0.8))
            min_plate_area = int(data.get("min_plate_area", 1000))
            min_plate_conf = float(data.get("min_plate_conf", 0.80))

            res = pipeline.process_bgr(
                bgr,
                mode=mode,
                contrast_boost=contrast_boost,
                min_face_area=min_face_area,
                face_det_thresh=face_det_thresh,
                face_rec_thresh=face_rec_thresh,
                min_plate_area=min_plate_area,
                min_plate_conf=min_plate_conf,
            )

            # Convert result dataclasses to dicts
            self.send_json({
                "width": res.width,
                "height": res.height,
                "faces": [asdict(f) for f in res.faces],
                "plates": [asdict(p) for p in res.plates],
                "telemetry": res.telemetry,
                "events": res.events,
            })
            return

        if path == "/api/process/video":
            assert pipeline is not None
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception as e:
                self.send_json({"error": f"Invalid JSON payload: {e}"}, 400)
                return

            video_b64 = data.get("video", "")
            video_sample_name = data.get("sample_name", "")
            stride = int(data.get("stride", 5))
            mode = data.get("mode", "both")
            max_frames = int(data.get("max_frames", 150))
            face_rec_thresh = float(data.get("face_rec_thresh", 0.8))
            min_face_area = int(data.get("min_face_area", 750))
            face_det_thresh = float(data.get("face_det_thresh", 0.6))
            min_plate_area = int(data.get("min_plate_area", 1000))
            min_plate_conf = float(data.get("min_plate_conf", 0.80))
            contrast_boost = bool(data.get("contrast_boost", False))

            temp_video_path: Path | None = None
            if video_sample_name:
                candidates = [
                    pipeline.root / "samples" / video_sample_name,
                    pipeline.root / "data" / video_sample_name,
                ]
                file_path = next((p for p in candidates if p.exists()), None)
                if not file_path:
                    self.send_json({"error": f"Sample video {video_sample_name} not found"}, 404)
                    return
                target_path = file_path
            else:
                if not video_b64:
                    self.send_json({"error": "No video payload or sample specified"}, 400)
                    return
                if "," in video_b64:
                    video_b64 = video_b64.split(",", 1)[1]
                video_bytes = base64.b64decode(video_b64)
                upload_dir = pipeline.root / "data" / "test_bench_uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                temp_video_path = upload_dir / f"temp_{int(time.time())}.mp4"
                temp_video_path.write_bytes(video_bytes)
                target_path = temp_video_path

            try:
                summary = pipeline.process_video_frames(
                    target_path,
                    stride=stride,
                    mode=mode,
                    max_frames=max_frames,
                    face_rec_thresh=face_rec_thresh,
                    min_face_area=min_face_area,
                    min_plate_conf=min_plate_conf,
                    contrast_boost=contrast_boost,
                )
                self.send_json(summary)
            except Exception as e:
                log.exception("Video processing failed")
                self.send_json({"error": str(e)}, 500)
            finally:
                if temp_video_path and temp_video_path.exists():
                    try:
                        temp_video_path.unlink()
                    except Exception:
                        pass
            return

        if path == "/api/live/start":
            assert pipeline is not None
            try:
                data = json.loads(body.decode("utf-8") or "{}")
            except Exception as e:
                self.send_json({"error": f"Invalid JSON payload: {e}"}, 400)
                return
            try:
                source = normalise_source(str(data.get("url") or data.get("rtsp") or ""))
            except ValueError as e:
                self.send_json({"error": str(e)}, 400)
                return

            tuning = LiveTuning.from_payload(data)
            try:
                session = start_live(source, tuning)
            except Exception as e:
                log.exception("Could not start live session")
                self.send_json({"error": f"Could not start live session: {e}"}, 500)
                return

            log.info(
                "live session started: %s (mode=%s, %.1f fps, max_width=%d)",
                session.display_source, tuning.mode, tuning.infer_fps, tuning.max_width,
            )
            self.send_json({
                "status": session.status,
                "source": session.display_source,
                "tuning": asdict(tuning),
            }, 201)
            return

        if path == "/api/live/stop":
            stop_live()
            self.send_json({"status": "stopped"})
            return

        if path == "/api/live/config":
            try:
                data = json.loads(body.decode("utf-8") or "{}")
            except Exception as e:
                self.send_json({"error": f"Invalid JSON payload: {e}"}, 400)
                return
            tuning = update_live(data)
            if tuning is None:
                self.send_json({"error": "No live session is running"}, 409)
                return
            self.send_json({"status": "ok", "tuning": asdict(tuning)})
            return

        if path == "/api/gallery/upload":
            assert pipeline is not None
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception as e:
                self.send_json({"error": f"Invalid JSON payload: {e}"}, 400)
                return

            name = data.get("name", "").strip().lower()
            if not name or ".." in name or "/" in name or "\\" in name:
                self.send_json({"error": "Invalid person name"}, 400)
                return

            photos = data.get("photos", [])  # list of base64 data urls
            if not photos:
                self.send_json({"error": "No photos provided"}, 400)
                return

            person_dir = pipeline.root / "ai" / "plugins" / "face_recognition" / "faces" / name
            person_dir.mkdir(parents=True, exist_ok=True)

            saved = 0
            for i, photo_b64 in enumerate(photos):
                bgr = decode_base64_image(photo_b64)
                if bgr is not None:
                    out_path = person_dir / f"{int(time.time())}_{i}.jpg"
                    cv2.imwrite(str(out_path), bgr)
                    saved += 1

            self.send_json({"message": f"Saved {saved} photo(s) for {name}", "saved": saved})
            return

        if path == "/api/gallery/rebuild":
            assert pipeline is not None
            # Run build_gallery algorithm
            try:
                exit_code = build_gallery.main()
                if exit_code == 0:
                    count = pipeline.reload_gallery()
                    self.send_json({
                        "message": f"Gallery successfully rebuilt with {count} identities!",
                        "identities": list(pipeline.face_engine.names) if pipeline.face_engine.names else [],
                        "count": count,
                    })
                else:
                    self.send_json({"error": "build_gallery returned non-zero code. Check that people have at least 3 photos with clear faces."}, 400)
            except Exception as e:
                log.exception("Gallery rebuild error")
                self.send_json({"error": str(e)}, 500)
            return

        if path == "/api/plates":
            assert pipeline is not None
            try:
                data = json.loads(body.decode("utf-8"))
                plates_map = data.get("plates", {})
            except Exception as e:
                self.send_json({"error": f"Invalid JSON: {e}"}, 400)
                return

            plates_file = pipeline.root / "data" / "known_plates.yaml"
            plates_file.parent.mkdir(parents=True, exist_ok=True)
            import yaml
            plates_file.write_text(yaml.safe_dump(plates_map), encoding="utf-8")
            pipeline.reload_known_plates()
            self.send_json({"message": "Saved known plates successfully", "plates": pipeline.known_plates})
            return

        self.send_error(404, "Endpoint not found")


def start_server(port: int = 8080) -> ThreadingHTTPServer:
    global pipeline
    pipeline = VisionPipeline()
    server = ThreadingHTTPServer(("127.0.0.1", port), TestBenchHandler)
    log.info("=" * 65)
    log.info(" CCTV Prototype Vision Test Bench running at:")
    log.info(" -> http://localhost:%d", port)
    log.info(" Models: YuNet (Face) | SFace (Embedding) | RapidOCR (LPR)")
    log.info(" Live:   RTSP / go2rtc restream from the RTSP source tab")
    log.info("=" * 65)
    return server


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CCTV prototype vision test bench")
    parser.add_argument("port", nargs="?", type=int, default=8080,
                        help="HTTP port (default 8080)")
    parser.add_argument("--rtsp", default="",
                        help="start a live RTSP session immediately, e.g. rtsp://user:pass@host:554/stream1")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="live inference rate (default 5)")
    parser.add_argument("--plate-every", type=int, default=5,
                        help="run plate OCR every N inference frames (default 5)")
    parser.add_argument("--mode", default="both", choices=sorted(("both", "faces", "plates")),
                        help="live detection mode (default both)")
    args = parser.parse_args()

    httpd = start_server(args.port)
    if args.rtsp:
        try:
            start_live(args.rtsp, LiveTuning.from_payload({
                "mode": args.mode,
                "fps": args.fps,
                "plate_every": args.plate_every,
            }))
        except ValueError as err:
            log.error("Cannot start live RTSP session: %s", err)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down test bench.")
    finally:
        stop_live()
        httpd.server_close()
