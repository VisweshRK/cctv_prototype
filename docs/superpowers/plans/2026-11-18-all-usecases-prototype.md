# One-Site All-Use-Cases Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the seven requested CCTV use cases inside the main Go/go2rtc website while keeping live camera video independent of inference.

**Architecture:** Preserve main's Go API, static website, RTSP/WebRTC/MSE stream path, Python `Plugin` contract and JSONL event store. Expose runtime settings and face-gallery controls on the main origin; schedule inference with latest-frame, bounded per-case work and keep heavyweight OCR away from the live path.

**Tech Stack:** Existing Go/go2rtc, Python 3.12 worker, OpenCV/ONNX Runtime/RapidOCR, vanilla HTML/CSS/JS, Go REST/SSE, Node and Python tests. CPU-only; no Vite/FastAPI rewrite.

**Spec:** `docs/INTEGRATION_DESIGN.md` and `docs/INTEGRATION_NOTES.md`; user-pasted brief from 2026-11-18.

## Global Constraints

- Exactly the seven user-listed use cases are the integrated acceptance target; remove fire/smoke code, UI, tests, config and documentation.
- Preserve main's stack and the branch's direct ancestry from `origin/main`.
- Keep `.env` credentials private; never return them in UI, logs, status or test output.
- Keep one latest frame per camera and bounded queues; a slow model must not queue stale frames or block browser playback.
- CPU-only model runtime; cap native model/OpenCV thread counts and document rates.
- Keep one unified Go event schema, REST/SSE path and existing event persistence.
- Model assets, private face photos, sample recordings, and generated data remain ignored/local.
- Run the full Go, Python, Node, API, browser, simulated-stream and 10-minute load checks requested by the brief; report measured results and limitations truthfully.

## Review Focus

- Camera disconnect/reconnect, bad H.264 packets, and camera URL credentials; test the reconnect/backoff path and ensure secrets never appear in response or logs.
- Missing/silent microphone tracks; test a no-audio state without blocking video or falsely reporting baby-cry detection as healthy.
- Empty/corrupt face gallery and low-quality photos; test validation, rebuild error handling, and model refresh after enrollment.
- Slow OCR/model inference and queue saturation; test that stale inference work is dropped and the independent live feed continues.
- Invalid zone/config updates and malformed events; test validation and keep prior valid settings active after rejected writes.

---

### Task 1: Record the baseline and remove fire/smoke

**Files:**
- Modify: `docs/INTEGRATION_NOTES.md`, `docs/INTEGRATION_REPORT.md`
- Delete: `ai/plugins/fire/`
- Modify: `ai/cctv_ai/worker.py`, `ai/tests/test_plugins.py`, `ai/tests/test_ported_plugins.py`, `www/cctv-logic.js`, `www/cctv.js`, `www/tests/cctv-logic.test.mjs`, `go2rtc.yaml`, `README.md`, `ai/README.md`

**Interfaces:** seven canonical use-case ids are shared by worker config, camera status, event filters, and the frontend.

- [ ] Add Python assertion `test_fire_is_not_discovered_or_enabled` and Node assertion `fire is absent from the use-case registry and sample events`.
- [ ] Run both tests and confirm they fail on current fire registration/fixtures.
- [ ] Remove fire plugin, model helper, docs, event simulation, siren default, use-case labels, and fire-only tests; update the seven-case list in README.
- [ ] Run targeted Python and Node tests; confirm fire is absent from `worker --list`, UI registry, and config.
- [ ] Record the audited branch counts and baseline suite results in `docs/INTEGRATION_REPORT.md`.

### Task 2: Pin CPU dependencies and make model readiness explicit

**Files:**
- Modify: `ai/requirements.txt`, `ai/plugins/fall/download_models.py`, `ai/plugins/face_recognition/download_models.py`, `ai/plugins/baby_crying/download_models.py`
- Create/modify: `scripts/download_models.ps1`, `scripts/download_models.sh`, model/readiness tests

**Interfaces:** `GET /api/cctv/health` reports each selected use case's `ready`, `degraded`, or `disabled` state plus a safe human-readable reason.

- [ ] Add readiness tests for missing MoveNet, YAMNet, face models/gallery, OCR runtime, and audio stream.
- [ ] Run them and confirm the current code cannot distinguish a configured slot from a ready detector.
- [ ] Pin the tested CPU-only direct dependency versions, verify downloader hashes, and make absent model state visible rather than silently healthy.
- [ ] Run dependency import/model-manifest checks on a fresh Python 3.12 environment.

### Task 3: Bound shared inference and protect real-time playback

**Files:**
- Modify: `ai/cctv_ai/worker.py`, `ai/cctv_ai/detector.py`, `ai/cctv_ai/tracker.py`, `ai/plugins/vehicle_lpr/`, `ai/cctv_ai/audio.py`
- Create: `ai/cctv_ai/scheduler.py`, `ai/tests/test_scheduler.py`, `ai/tests/test_worker_runtime.py`

**Interfaces:** scheduler accepts `(camera, frame_seq, frame, timestamp)` and offers at most one pending item per camera/use-case; metrics expose captured FPS, inference FPS/ms, queue depth, dropped stale work, CPU/RAM, and audio state.

- [ ] Add tests `test_slow_plugin_drops_old_frames`, `test_plugin_failure_does_not_stop_sibling_plugins`, `test_lpr_runs_only_for_vehicle_or_plate_candidate`, and `test_runtime_metrics_include_queue_and_drop_counts`.
- [ ] Run tests and observe stale-frame growth/serial OCR on the baseline.
- [ ] Implement per-case bounded latest-value scheduling and low-priority LPR; keep one shared person detector and tracker for fall/occupancy/energy.
- [ ] Configure OpenCV/ONNX/OMP thread ceilings before model load; isolate audio capture and model work from the Go stream path.
- [ ] Verify fresh-frame processing, bounded queue depth, and unchanged Go camera FPS under delayed inference.

### Task 4: Add persisted runtime configuration and health APIs

**Files:**
- Modify: `internal/cctv/cctv.go`, `internal/cctv/status.go`, `internal/cctv/cctv_test.go`, `go2rtc.yaml`
- Create: `internal/cctv/config.go`, `internal/cctv/config_test.go`, `data/ai-config.example.json`
- Modify: `ai/cctv_ai/worker.py`, `ai/cctv_ai/client.py`

**Interfaces:** add `GET/PUT /api/cctv/config`, `POST /api/cctv/config/test-camera`, and `GET /api/cctv/health`. Worker polls validated config and applies per-camera enablement/rates without restarting video; API never echoes credentials.

- [ ] Add API tests for per-camera use-case toggles, threshold/rate updates, invalid rollback, and credential masking.
- [ ] Run them and confirm the current API is read-only for runtime configuration.
- [ ] Implement atomic config persistence under `data/`, defaults from checked-in example, and health data from worker heartbeat.
- [ ] Verify a config update changes scheduler state without reconnecting the go2rtc stream.

### Task 5: Put face gallery and use-case controls in the main website

**Files:**
- Modify: `www/cctv.html`, `www/cctv.js`, `www/cctv-logic.js`, `www/tests/cctv-logic.test.mjs`
- Modify: `internal/cctv/cctv.go`, `internal/cctv/` API tests
- Extract/reuse: `ai/plugins/face_recognition/build_gallery.py`, gallery validation helpers

**Interfaces:** main-origin gallery endpoints `GET /api/cctv/faces`, `POST /api/cctv/faces`, `DELETE /api/cctv/faces/{id}`, and `POST /api/cctv/faces/rebuild`; UI updates include live worker readiness after rebuild.

- [ ] Add tests for safe identity names, accepted images, invalid/corrupt images, deletion, rebuild response, and same-origin main nav.
- [ ] Run them and confirm the gallery currently exists only at port 8080.
- [ ] Add Face Gallery, LPR/watchlist, Tampering, Zones/Occupancy, Energy Saving, Events, Settings, Video Jobs, and Health views under the main `www/` shell.
- [ ] Add per-camera enable controls, event filters/acknowledgement, zone polygon editing, occupancy capacity, tamper reference/sensitivity, appliance ROIs/absence delay/action toggle/log, RTSP diagnostics, and upload progress.
- [ ] Keep detection boxes and labels in a canvas overlay; never draw overlays into the streamed frame.
- [ ] Verify all seven use cases have an enabled/disabled state, configuration, healthy/degraded state, and event history in the main site.

### Task 6: Integrate audio, face identity refresh, and use-case settings

**Files:**
- Modify: `ai/plugins/baby_crying/`, `ai/plugins/face_recognition/`, `ai/plugins/vehicle_lpr/`, `ai/plugins/tampering/`, `ai/plugins/occupancy/`, `ai/plugins/power_when_absent/`, `ai/plugins/fall/`
- Create/modify: relevant plugin/API tests and checked-in sample manifests

**Interfaces:** plugin settings are loaded from the shared per-camera runtime configuration; each plugin publishes status and inference timing to worker health.

- [ ] Add tests for cry hysteresis/no-audio, face gallery reload, LPR cooldown/watchlist, tamper classes/reference update, zone over-capacity, vacancy timer/device action rearm, and fall state persistence.
- [ ] Run new tests and confirm missing settings/model/audio cases fail visibly.
- [ ] Implement only the missing behavior required by the seven use cases; remove unused parallel test-bench capture from product startup.
- [ ] Verify plugin events remain on the single event REST/SSE route and include snapshots/metadata where supported.

### Task 7: One-command startup and clean-clone documentation

**Files:**
- Create: `scripts/run.ps1`, `scripts/run.sh`, `.env.example` (if missing), `scripts/README.md`
- Modify: `README.md`, `ai/README.md`, `.gitignore`, `go2rtc.yaml`

- [ ] Add a startup smoke test covering dependency check, model manifest, and process health without needing camera secrets.
- [ ] Run it and confirm current setup requires hand-starting Go and Python separately.
- [ ] Start go2rtc and worker under one command, report child process failures, support Ctrl+C cleanup, and never log secret URLs.
- [ ] Verify fake sample inputs are test-only and do not replace a configured production RTSP URL.
- [ ] Verify clean-clone instructions on Windows and POSIX shell.

### Task 8: End-to-end and 10-minute CPU load verification

**Files:**
- Create: browser E2E fixtures/tests, `scripts/load_test.ps1` or `scripts/load_test.py`
- Create/modify: `docs/INTEGRATION_REPORT.md`

- [ ] Add and run unit tests for every plugin with the specified fall, cry, face, plate, tamper, zone, and vacancy fixtures.
- [ ] Add and run API tests for events, config, gallery enrollment, and credential masking.
- [ ] Add and run browser E2E against a simulated RTSP stream with video and audio, covering enroll/recognize, all seven alerts, settings, zones, upload, and disconnect/reconnect.
- [ ] Run the 10-minute seven-case + upload load test on CPU; record source/live FPS, added latency under events, per-case FPS/ms, queue depth/drops, CPU/RAM, and memory growth.
- [ ] Fix failures and rerun; report unavailable hardware, missing reference data, and any unmet acceptance criterion explicitly.

## Execution order

Implement tasks sequentially because configuration, health, scheduler and UI contracts are shared. Each behavior change follows red/green/refactor. Re-run the complete Go, Python, and Node suites after each cross-cutting task, and complete the browser/load checks last.
