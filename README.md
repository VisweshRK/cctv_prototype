# CCTV Prototype

A smart CCTV prototype for two Q-Tech Wi-Fi cameras with two-way audio, built on
[go2rtc](https://github.com/AlexxIT/go2rtc) and taking ideas from [Frigate](https://github.com/blakeblackshear/frigate).

- **go2rtc (Go)** connects to each camera once and serves live view (web UI, WebRTC, RTSP restream).
- **`internal/cctv` (Go, our module)** watches camera status, raises tampering/disconnect events, and
  stores all events behind a small HTTP API.
- **`ai/` (Python worker)** pulls frames and audio from go2rtc, runs one plugin per use case, and sends
  events back to go2rtc.

The original go2rtc README is kept as [README.go2rtc.md](README.go2rtc.md). Credits are in [NOTICE](NOTICE).

```
cameras ──RTSP──> go2rtc ──frames / RTSP restream──> ai worker (plugins) ──POST events──> go2rtc /api/cctv/events
                    └── internal/cctv: camera status + disconnect events
```

## Quick start (Windows)

You need [Go 1.24+](https://go.dev/dl/), Python 3.12, and **ffmpeg** on PATH. go2rtc uses ffmpeg to turn
H.264 frames into JPEGs, and the worker uses it to read audio:

```bash
winget install Gyan.FFmpeg
```

1. Camera addresses (kept out of git):
   ```bash
   copy .env.example .env
   ```
   Then put the real `CAM1_URL` / `CAM2_URL` in `.env`. **Never commit `.env`.**
2. Build and run go2rtc:
   ```bash
   go build -o go2rtc.exe .
   ```
   ```bash
   .\go2rtc.exe
   ```
   Open http://localhost:1984 for live view. Config is in [go2rtc.yaml](go2rtc.yaml).
3. Run the AI worker in a second terminal:
   ```bash
   cd ai
   py -3.12 -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   .venv\Scripts\python -m cctv_ai.worker --list
   .venv\Scripts\python -m cctv_ai.worker -v
   ```
   By default only the working `example_motion` plugin runs. See [ai/README.md](ai/README.md).

## API (added by `internal/cctv`)

| Endpoint | What it does |
| --- | --- |
| `GET /api/cctv/status` | Each camera: `online`, `bytes_recv`, `last_seen`, `since`, `error` |
| `GET /api/cctv/events?camera=&use_case=&limit=` | Recent events, newest first |
| `POST /api/cctv/events` | Add an event (JSON below). Returns `201` with the saved event, or `400` with `{"error": ...}` |

Event format (shared by Go and Python):

```json
{
  "camera": "cam1",
  "use_case": "intrusion",
  "label": "person",
  "confidence": 0.91,
  "time": "2026-10-18T21:30:00+05:30",
  "snapshot": "optional path or URL",
  "data": { "optional": "extra details" }
}
```

`camera`, `use_case` and `label` are required; `confidence` must be between 0 and 1. Events are also
appended to `data/events.jsonl` (git-ignored) and reloaded on restart.

## Use-case slots

Each use case has its own folder with a README that says what to build and which Frigate code to read.
Take a slot by working on a branch `usecase/<slot>` (see [CONTRIBUTING.md](CONTRIBUTING.md)).

| # | Use case | Slot | Input | Owner |
| --- | --- | --- | --- | --- |
| 1 | Intrusion | [ai/plugins/intrusion](ai/plugins/intrusion) | video | _open_ |
| 2 | Tampering / disconnect | [ai/plugins/tampering](ai/plugins/tampering) (disconnect is done in [internal/cctv/status.go](internal/cctv/status.go)) | video | _open_ |
| 3 | Face recognition | [ai/plugins/face_recognition](ai/plugins/face_recognition) | video | _open_ |
| 4 | Restricted area | [ai/plugins/restricted_area](ai/plugins/restricted_area) | video | _open_ |
| 5 | Fall detection | [ai/plugins/fall](ai/plugins/fall) | video | _open_ |
| 6 | Occupancy | [ai/plugins/occupancy](ai/plugins/occupancy) | video | _open_ |
| 7 | Electricity used when nobody is there | [ai/plugins/power_when_absent](ai/plugins/power_when_absent) | video | _open_ |
| 8 | Fire and smoke | [ai/plugins/fire](ai/plugins/fire) | video | _open_ |
| 9 | Vehicle + licence plate | [ai/plugins/vehicle_lpr](ai/plugins/vehicle_lpr) | video | _open_ |
| 10 | Baby crying | [ai/plugins/baby_crying](ai/plugins/baby_crying) | audio | _open_ |

Put your name in the Owner column in your first PR.

[docs/frigate-teardown.md](docs/frigate-teardown.md) explains how Frigate handles each use case and where
we can be more efficient than Frigate.

## Tests

```bash
go test ./internal/cctv/
```
```bash
cd ai
.venv\Scripts\python -m unittest discover -s tests
```

CI runs both on every push and pull request to `main` ([.github/workflows/cctv-ci.yml](.github/workflows/cctv-ci.yml)).

## Licence

MIT. go2rtc is MIT-licensed by Alexey Khit ([LICENSE](LICENSE)); see [NOTICE](NOTICE).
