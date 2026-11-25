# All-Use-Cases Integration Audit

Audit date: 2026-11-18
Working branch: `integration/all-usecases`
Base: `origin/main` at `bc73b65` (`Merge usecase/tampering into main`)
Audited HEAD: `2e6fbae` (`feat: complete all-usecases integration and verification`)

## Scope

The user-supplied integration brief specifies seven use cases: fall, baby-cry detection, face recognition, licence-plate recognition, camera tampering, zone occupancy, and energy saving when a configured zone is empty. It explicitly removes fire and smoke detection. Fire is therefore out of scope and must be removed from code, tests, UI, camera configuration, and documentation. The existing intrusion and restricted-area folders are separate unfinished slots and are not part of this seven-use-case list.

## Branch audit

`integration/all-usecases` is already a descendant of `origin/main`; it is 10 commits ahead and 0 behind. Continue on this branch. Do not recreate or reset it.

| Remote branch | Relative to `origin/main` | Audit finding |
| --- | ---: | --- |
| `origin/usecase/fall` | 1 ahead, 0 behind | Adds the shared YOLOX person detector, tracker, zone helpers, and MoveNet fall plugin. The MoveNet model is fetched separately; it was absent from this machine at audit time. |
| `origin/usecase/face_recognition` | 2 ahead, 0 behind | Adds YuNet/SFace recognition and gallery builder, the RapidOCR/rapidfuzz LPR plugin, and a separate FastAPI test bench with gallery UI. The standalone bench is not part of the main Go website. |
| `origin/usecase/occupancy` | 2 ahead, 0 behind | Adds occupancy and zones on the same object pipeline as fall, plus a tracker correction and small tampering change. |
| `origin/usecase/tampering` | 2 behind, 0 ahead | Already represented by `main`; no additional diff remains. |
| `origin/use-cases-fd,bc,es` | unrelated history; 2 source commits | A separate FastAPI/Vite architecture with no merge base. The baby-cry and energy-saving logic has already been selectively ported into main's Python plugin contract. Do not merge its app/API/database. Fire from that source is excluded. |
| `origin/sanjaym83786-hash-patch-1` | 1 ahead, 0 behind | Standalone FastAPI/Ultralytics intrusion server, not one of the seven requested slots and not compatible with main's plugin contract; do not import it. |

The integration branch already contains the compatible fall, occupancy, face, and LPR work. Re-merging the branches would duplicate shared-core files.

## Main architecture

- **Backend and live video:** Go 1.24+ application based on go2rtc. RTSP sources are configured in root `go2rtc.yaml` with secret URLs substituted from ignored `.env`. go2rtc serves web UI/API on 1984, RTSP restreams on 8554, and WebRTC on 8555. Its browser player uses the main static web UI's existing MSE/WebRTC stream component. The live stream path is separate from Python inference.
- **Use-case contract:** `ai/cctv_ai/plugin.py` defines `Plugin`, `Event`, `on_frame`, `on_audio`, and `on_objects`. `ai/cctv_ai/worker.py` discovers plugins, captures RTSP frames, runs the shared object detector/tracker where requested, and posts normalized events to Go. Python currently runs one serial inference loop per camera; the camera capture thread overwrites the latest frame while inference is busy. Event delivery uses a bounded queue and sender thread. This is a useful baseline, but there is no per-use-case process isolation or adaptive scheduler, and full-frame LPR can take multiple seconds on CPU.
- **Event API and storage:** `internal/cctv` exposes `GET/POST /api/cctv/events`, `GET /api/cctv/events/stream` (SSE), `POST /api/cctv/events/ack`, `GET /api/cctv/status`, and `GET /api/cctv/snapshot`. All use cases share the Go event schema and one append-only `data/events.jsonl` event store with an in-memory capped index. There is no SQLite database in main; preserving its established event store avoids introducing a second source of truth.
- **Frontend:** Main uses static HTML, CSS, and vanilla JavaScript in `www/`, served by Go. `www/cctv.html` and `www/cctv.js` provide the live camera dashboard, event queue, snapshots, KPI/timeline, basic occupancy/energy summaries, and demo events. They do not yet provide configuration, gallery enrollment, zone drawing, LPR watchlist, tamper calibration, video jobs, or system metrics. The separate `ai/test_bench` has a face/gallery studio and several inference controls, but must be folded into the main site rather than presented as another website.
- **Configuration:** Camera source credentials are in `.env`; stream names, use-case labels, and event/snapshot/siren settings are in root `go2rtc.yaml`. There is no unified runtime configuration API, and the `cctv.use_cases` arrays currently only describe static dashboard labels; they do not control the Python worker.
- **Tests:** Go package tests in `internal/cctv`; Python `unittest` in `ai/tests`; Node tests in `www/tests`. There is no browser E2E or load-test harness in the baseline.

## Use-case audit

| Use case | Current implementation and inputs | Output, dependencies, and current gap |
| --- | --- | --- |
| Fall detection | `ai/plugins/fall`; shared person tracks plus MoveNet pose. | Unified `fall` events; model downloader exists, but MoveNet asset is required. No main-UI sensitivity/configuration. |
| Baby-cry detection | `ai/plugins/baby_crying`; FFmpeg extracts 16 kHz mono audio from each camera RTSP restream; harmonic fallback and optional YAMNet. | Unified `baby_crying` events; current live camera restream emitted FFmpeg “no stream” errors in the worker, so camera audio was unavailable. Missing-audio state is not surfaced in the dashboard. |
| Face recognition | `ai/plugins/face_recognition`; YuNet and SFace ONNX models, 128-D embeddings, cosine matching, photo folders compiled to `data/faces.npz`. | Unified face events and snapshots. Main dashboard has no enrollment UI; enrollment is only in the standalone test bench. Model files were fetched locally during the previous turn; no identity gallery exists yet. |
| Licence-plate recognition | `ai/plugins/vehicle_lpr`; RapidOCR ONNX and rapidfuzz, currently full-frame OCR on a timer. | Unified plate events and optional known-plate labels. CPU OCR blocked the same per-camera inference loop for 5–40 seconds in the prior live smoke session; gate OCR behind vehicle/plate candidates and schedule it off the live capture path. |
| Camera tampering | `ai/plugins/tampering` plus Go camera-offline events; classical frame/reference checks and disconnect detection. | Unified tamper/disconnect events. Main UI lacks reference-frame capture and per-check sensitivity controls; camera-source RTSP timeouts were observed in the prior live smoke. |
| Zone occupancy | Shared YOLOX-Tiny ONNX person detector, tracker, polygon zones, occupancy counter. | Unified count/over-capacity events; main UI only shows a basic count summary, without a zone editor or over-capacity settings. |
| Energy saving | `ai/plugins/power_when_absent`; person tracks, absence hysteresis, configured appliance ROIs and brightness signals. | Unified energy events and estimated usage. ROI calibration and absence timing are not managed in the main UI; there is no simulated device-action toggle/log yet. |

## Baseline readiness observed on 2026-11-18

- Initial `ai/.venv` lacked `rapidfuzz` and `rapidocr-onnxruntime`; `worker --list` failed during plugin discovery. These packages were installed into the ignored venv from `ai/requirements.txt`.
- The face-model downloader successfully fetched YuNet and SFace with their repository SHA-256 checks. The gallery remains empty. MoveNet is absent. YAMNet is absent.
- `worker --list` shows the seven selected plugins enabled after dependency installation, but this is not proof of correct detection. It also lists `example_motion` and fire; intrusion and restricted-area are stubs.
- A real stream smoke confirmed both configured camera feeds returned frames and the worker posted events, but H.264 decode errors, camera RTSP timeouts, absent audio, and multi-second LPR calls were observed. A simultaneous gallery test-bench reader increased camera stream contention. The branch therefore does not yet meet the no-lag or seven-use-case acceptance criteria.
- Baseline Node suite: 34 tests passed. Go package tests/build were initially blocked by Go's module-stat cache permission outside the workspace; rerun with the approved elevated Go command. Python suite is in progress for the baseline record.

## Chosen integration direction

Keep main's Go/go2rtc backend, static vanilla-JS frontend, plugin/event contract, camera stream path, and JSONL event store. Add the operator-facing controls and gallery to `www/cctv.html`/`www/cctv.js`, expose them through same-origin Go routes, and leave all inference behind the Python worker. Give each camera a one-frame latest-value capture buffer; schedule each use case independently with bounded work so slow OCR cannot block another use case or live video. Preserve the independent go2rtc browser feed. Do not import the unrelated Vite/FastAPI platform.
