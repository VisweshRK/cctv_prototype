# Vision test bench

A temporary browser UI over the real inference pipeline (`test_bench/runner.py`):
upload an image or video, use the webcam, or point it at a **live RTSP camera**
and watch face recognition (YuNet + SFace) and licence plate recognition
(RapidOCR) run frame by frame.

The pipeline and the live session are the long-term code; the page, sliders and
event cards are scaffolding for trying models quickly. Keep new detection
behaviour in `runner.py` / `plugins/`, not in `live.py`.

## Run

```bat
scripts\test_bench.cmd
```

or from `ai/`:

```bat
.venv\Scripts\python test_bench\server.py
```

Open http://localhost:8080. The first load takes a few seconds while the
models and the face gallery are read from disk.

Start a live source immediately (no clicking):

```bat
.venv\Scripts\python test_bench\server.py 8080 --rtsp rtsp://user:pass@192.168.1.50:554/stream1
.venv\Scripts\python test_bench\server.py 8080 --rtsp rtsp://localhost:8554/cam1 --mode faces --fps 10
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `port` | `8080` | HTTP port |
| `--rtsp` | | live source to start with (RTSP, RTSP-over-TLS or HTTP/MJPEG) |
| `--fps` | `5` | live inference rate cap |
| `--mode` | `both` | `both`, `faces` or `plates` |
| `--plate-every` | `5` | run plate OCR every N inference frames |

## Live sources

Paste the URL into the **RTSP** source tab. All of these work because they end
up in the same OpenCV capture path:

- a camera directly: `rtsp://user:pass@192.168.1.50:554/stream1`
- Hikvision/CP Plus style inline path (some brands put the credentials in the
  path instead of the URL userinfo; an empty password is left unmasked):
  `rtsp://192.168.1.50:554/user=admin_password=_channel=0_stream=0&onvif=0.sdp`
- TLS: `rtsps://camera.local/live`
- go2rtc's restream (recommended, one camera connection shared with the rest
  of the prototype): `rtsp://localhost:8554/cam1`
- an MJPEG/HTTP stream: `http://localhost:1984/api/stream.mjpg?src=cam1`

Tips:

- Use the camera **substream** (usually 640x360 or 704x576). The main stream is
  sharper than the models need and costs more to decode.
- RTSP defaults to RTP over TCP, which survives Wi-Fi packet loss:
  `OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp|timeout;5000000"` is set
  automatically (only if you have not set it yourself). Set that variable to
  use UDP or a longer timeout.
- Passwords are masked as `rtsp://user:***@host/...` in the UI, logs and API
  replies, including passwords repeated inside the URL path. The browser never
  receives the real password.
- Face size matters more than stream resolution: a face below roughly 100 px
  in the frame rarely embeds reliably, so a substream is usually fine.

## How live mode behaves

Two daemon threads per source:

1. **Reader** — connects once and keeps only the newest decoded frame. Older
   frames are discarded instead of queued, so a slow model cannot make the
   picture drift further and further behind the camera.
2. **Worker** — runs `VisionPipeline.process_bgr` on that newest frame, encodes
   the JPEG and publishes the result. The HTTP handler only copies the latest
   result; it never runs inference, so several tabs share one model pass.

The page polls `/api/live/frame?after=<seq>` about ten times a second. When the
browser already has the newest result the reply is a few hundred bytes
(`"unchanged": true`) without the JPEG.

Frames are downscaled to `max_width` (1280 by default) before inference and
encoding, so boxes and JPEG coordinates always agree.

### Why plate OCR runs every Nth frame

Full-frame RapidOCR costs hundreds of milliseconds to a few seconds per frame
on a laptop CPU, while face recognition costs tens of milliseconds. Running
plates on every frame would drop live mode to a crawl. The worker therefore:

- runs **faces every frame**;
- runs **plate OCR every `plate_every` frames** (default 5);
- reuses the last plate result on the frames in between, and clears it when a
  later scan finds no plate or the mode changes away from plates.

Change it in the RTSP tab (**Plate OCR: every Nth frame**) or with
`plate_every` in the API. `1` means every frame (slow), `10` means every ten
frames (about two seconds at 5 FPS before the scan itself is counted). The
event feed still de-duplicates the same face/plate for 15 seconds, so a parked
car does not flood the list.

Other live controls (mode buttons, sliders, CLAHE) are pushed to the running
session without reconnecting. `fps` is a ceiling: a plate frame that takes two
seconds sets the pace for that frame.

## HTTP API

| Endpoint | What it does |
| --- | --- |
| `GET /api/live/status` | `status` (`connecting`/`live`/`reconnecting`/`stopped`), `seq`, masked `source`, `stats` |
| `GET /api/live/frame?after=N` | latest JPEG + detections, or `{"unchanged": true, ...}` when `seq <= N` |
| `POST /api/live/start` | `{"url": "...", "mode": "both", "fps": 5, "plate_every": 5, ...}`; replaces a running session, returns `201` |
| `POST /api/live/config` | any of the tuning fields; applies to the running session, returns `409` when idle |
| `POST /api/live/stop` | stop the reader and worker |
| `GET /api/status` | model/gallery status |
| `POST /api/process/image` | run the pipeline on one base64 image |
| `POST /api/process/video` | run the pipeline on an uploaded or sample video |

## Tests

```bat
cd ai
.venv\Scripts\python -m unittest discover -s tests
```

`tests/test_live.py` covers URL validation and credential masking, tuning
clamps (including `plate_every`), event cooldown, and the `unchanged` snapshot
reply. `tests/test_face_recognition.py` and `tests/test_vehicle_lpr.py` cover
the models the live loop calls.

## Layout

| Path | What |
| --- | --- |
| `server.py` | HTTP server, live-session lifecycle, CLI |
| `live.py` | reader/worker threads, tuning, cooldown, snapshot API |
| `runner.py` | the real `VisionPipeline` used by live, uploads, webcam and video |
| `web/` | temporary UI (`index.html`, `app.js`, `style.css`) |
