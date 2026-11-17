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
.venv\Scripts\python -m cctv_ai.worker --plugins example_motion,intrusion --cameras cam1
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--url` | `http://localhost:1984` | go2rtc API address |
| `--plugins` | every plugin with `enabled = True` | comma-separated slot names (stubs can be run this way too) |
| `--cameras` | every camera in `cctv.cameras` | comma-separated stream names |
| `--list` | | print every slot and whether it is enabled |
| `-v` | | debug logging |

## How it works

- One thread per camera for video and one for audio (only if some plugin needs it).
- Each frame is fetched **once** per camera and shared by all video plugins, at the fastest
  `interval` any plugin asks for. A plugin with a slower `interval` is called less often.
- Cameras that `/api/cctv/status` reports offline are skipped (checked every 5 s).
- Audio is read from go2rtc's RTSP restream (`rtsp://localhost:8554/<camera>`) as 16 kHz mono
  int16, in 1 s chunks, so the camera keeps a single connection.
- If a plugin raises an exception it is logged and the other plugins keep running.

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
| `plugins/<slot>/` | one stub per use case, each with a README |
| `tests/` | `python -m unittest discover -s tests` |

## Writing a plugin

```python
from cctv_ai.plugin import Plugin as Base


class Plugin(Base):
    use_case = "intrusion"   # same as the folder name
    enabled = True
    needs = {"video"}        # or {"audio"}
    interval = 1.0           # seconds between calls per camera

    def setup(self, cameras):
        self.model = ...     # load once

    def on_frame(self, camera, frame, ts):
        # frame: BGR numpy image
        if person_found:
            return [self.event(camera, "person", 0.9, data={"box": [x, y, w, h]})]
        return []
```

See [CONTRIBUTING.md](../CONTRIBUTING.md) for branches and PRs.
