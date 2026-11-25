# AI worker

Python side of the prototype. It asks go2rtc for frames and audio, runs the enabled plugins, and
sends their events to `POST /api/cctv/events`.

## Setup

```bash
cd ai
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

go2rtc must be running (http://localhost:1984). ffmpeg must be on PATH for H.264 frames and for audio.

## Running

```bash
.venv\Scripts\python -m cctv_ai.worker --list
```
```bash
.venv\Scripts\python -m cctv_ai.worker -v
```
```bash
.venv\Scripts\python -m cctv_ai.worker --plugins tampering,occupancy --cameras cam1
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--url` | `http://localhost:1984` | go2rtc API address |
| `--plugins` | every plugin with `enabled = True` | comma-separated slot names (stubs can be run this way too) |
| `--cameras` | every camera in `cctv.cameras` | comma-separated stream names |
| `--list` | | print every slot and whether it is enabled |
| `-v` | | debug logging |

## How it works

- One capture thread per camera continuously reads the go2rtc RTSP restream with a one-frame buffer;
  it overwrites the previous frame, so inference never builds a latency queue. HTTP JPEG is only a
  fallback for synthetic/test cameras.
- Each case has a separate inference lane. A slow case keeps at most one pending frame; new work replaces
  stale pending work so OCR or face inference cannot hold up sibling cases. The dashboard's Go stream
  remains a separate process from model inference.
- Event POSTs use a bounded asynchronous sender queue. A slow API, snapshot, or siren operation cannot
  block capture or inference; queue drops are logged.
- Cameras that `/api/cctv/status` reports offline are skipped (checked every 5 s).
- Audio is read from go2rtc's RTSP restream (`rtsp://localhost:8554/<camera>`) as 16 kHz mono
  int16, in 1 s chunks, so the camera keeps a single connection.
- If a plugin or the shared detector raises an exception it is logged and the other plugins keep running.
- Native OpenMP/BLAS threads are limited to one per worker process. INFO logs include capture rate and
  per-case timing. `GET /api/cctv/health` reports model and configuration readiness.

## Enabled use cases

The worker defaults to these seven requested use cases: `fall`, `baby_crying`, `face_recognition`,
`vehicle_lpr`, `tampering`, `occupancy`, and `power_when_absent`. `/api/cctv/health` reports missing
models and configuration as degraded/disabled. Face recognition needs YuNet/SFace plus an enrolled
gallery. Baby crying needs a supported camera audio track and YAMNet; the present harmonic fallback is
approximate and is explicitly marked degraded. Fall detection needs MoveNet for pose confirmation.
Configure appliance ROIs before expecting energy alerts. Intrusion and restricted-area remain dormant slots.

## Test bench

A temporary browser UI for trying the face and licence-plate models on an image, video,
webcam or live RTSP source:

```bat
cd ai
.venv\Scripts\python test_bench\server.py
```

Open http://localhost:8080, or run `scripts\test_bench.cmd` from the repository root. It uses
the same `VisionPipeline` as the worker; RTSP/go2rtc setup, the live API and the plate-OCR
cadence are documented in [test_bench/README.md](test_bench/README.md).

## Layout

| Path | What |
| --- | --- |
| `cctv_ai/plugin.py` | `Plugin` base class and `Event` (same fields as the Go event) |
| `cctv_ai/worker.py` | the worker loop and command line |
| `cctv_ai/client.py` | small go2rtc API client |
| `cctv_ai/audio.py` | reads audio through ffmpeg |
| `cctv_ai/loader.py` | finds plugins in `plugins/` |
| `plugins/example_motion/` | working reference plugin (motion detection) |
| `plugins/<slot>/` | one contract-compatible implementation per use case, each with a README |
| `tests/` | `python -m unittest discover -s tests` |

## Writing a plugin

```python
from cctv_ai.plugin import Plugin as Base


class Plugin(Base):
    use_case = "tampering"   # same as the folder name
    enabled = True
    needs = {"video"}        # or {"audio"}, or add "objects"
    interval = 1.0           # seconds between calls per camera

    def setup(self, cameras):
        self.model = ...     # load once

    def on_frame(self, camera, frame, ts):
        # frame: BGR numpy image
        if person_found:
            return [self.event(camera, "person", 0.9, data={"box": [x, y, w, h]})]
        return []

    def on_objects(self, camera, frame, ts, objects, ended):
        # objects are tracked TrackedObject values; use true_positive persons only.
        return []
```

See [CONTRIBUTING.md](../CONTRIBUTING.md) for branches and PRs.
