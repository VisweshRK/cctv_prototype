# CCTV all-use-cases integration report

Date: 2026-11-17  
Branch: `integration/all-usecases`  
Base: `origin/main` at `bc73b65`

## Outcome

The branch preserves main's Go/API/frontend structure and the `cctv_ai.plugin.Plugin` / `Event` contract. It integrates the compatible use-case branches in the requested order, ports the three applicable implementations from the unrelated platform branch, updates the dashboard logic and tests, and changes the worker to prefer fresh frames and asynchronous event delivery.

## Merged and ported work

1. `origin/usecase/face_recognition` was merged as commit `8e19a40`. This added the face-recognition and vehicle-LPR implementations, their model helpers/tests, and the required Python dependencies. The test bench remains a separate development tool and is not part of the production ports.
2. `origin/usecase/occupancy` was merged as `e6c4fd4`. Its corrected tracker helpers were retained, including `union`, IoU, and containment handling. The shared contract and tests were combined without retaining broad ignore rules that hid test assets.
3. `origin/usecase/fall` was merged as `38a2ec7`. Occupancy's tracker implementation was retained, while fall's detector/tracker exception isolation was kept in the worker.
4. `origin/sanjaym83786-hash-patch-1` was reviewed but not wired. Its intrusion file is a standalone FastAPI/Ultralytics RTSP application, not a `Plugin` subclass and not compatible with the main event, zone, or worker contract.
5. `origin/use-cases-fd,bc,es` was not merged. It has unrelated history and a separate FastAPI/Vite/`StandardAIEvent` architecture. Fire, baby crying, and power-when-absent were ported selectively into main's plugin contract as commit `d2e1155`; its API, database, auth, frontend, binaries, hardcoded paths, and duplicate platform implementations were not imported.
6. `tampering` was already represented by main and required no additional merge.

## Ported use cases

- **Fire:** HSV flame/smoke signals, temporal persistence, and cooking-zone awareness; emits normalized frontend fields such as `area`, `detail`, and `box` where available.
- **Baby crying:** 16 kHz audio processing with RMS/harmonic fallback and optional YAMNet ONNX support; maintains episode start/end state rather than flooding one event per audio chunk.
- **Power when absent:** tracked-object absence hysteresis, appliance ROIs, frame-based appliance signals, wattage/reference estimates, and re-arming; emits `watts`, `minutes`, and/or `kwh` for the dashboard energy panel.

No model weights, secrets, `.env` files, or binaries were added.

## Conflict and contract decisions

- Main's directory layout and plugin discovery mechanism were preserved.
- The occupancy tracker fix won the shared-file conflict.
- Detector/tracker failures remain isolated so a broken model does not terminate other use cases.
- The canonical intrusion and restricted-area slots remain the main-contract implementations; the standalone intrusion server was not adopted.
- Frontend event fields continue to use normalized `data.box`, `data.area`/`data.zone`, `data.detail`, occupancy `data.count`, and energy `data.kwh` or `data.watts` plus `data.minutes`.

## Frontend changes

`www/cctv-logic.js` now exercises all ten use cases in its simulation cycle, including fall, baby crying, power-when-absent, and occupancy. Simulation confidence is clamped to `[0, 1]`, missing boxes are handled safely, and camera selection falls back to an available camera when no camera is configured for a use case. Existing popup, acknowledgement, siren, snapshot, incident grouping, timeline, occupancy, and energy helpers remain covered. The dashboard continues to update existing streams rather than recreating `<video-stream>` elements.

## Worker performance changes

The previous synchronous HTTP-frame / synchronous event-POST path was changed as follows:

- Each camera has a daemon RTSP capture thread with a one-frame OpenCV buffer. It overwrites stale frames instead of allowing a queue to grow.
- The HTTP JPEG endpoint remains a fallback when no newer RTSP frame is available, which keeps fake/test-camera operation possible.
- Shared object detection and tracking still run once per selected frame for all object plugins.
- Event delivery uses a bounded non-blocking queue and a sender thread, so slow HTTP, snapshot, or siren handling does not block capture/inference. Queue overflow is logged and drops are explicit.
- Plugin and detector/tracker durations are logged at DEBUG level; capture FPS is logged periodically at INFO; final queue/drop counts are logged on shutdown.
- Models are loaded in plugin setup where supported, and plugin/detector exceptions are isolated.

A controlled before/after camera benchmark was not available in this sandbox: there were no real RTSP cameras and no representative hardware/model workload. Therefore no numeric latency improvement is claimed. The implementation provides the intended latest-frame/drop mechanism and runtime metrics needed for deployment measurement.

## Verification performed

Commands actually run on this branch:

```bash
cd ai && python -m unittest discover -s tests
```

Result: **134 tests passed, 7 skipped**. Skips are model-asset-dependent tests; the output also showed expected missing optional MoveNet/face model notices.

```bash
cd ai && python -m unittest tests.test_ported_plugins -v
```

Result: **3 tests passed** covering fire persistence/frontend fields, baby-crying episode lifecycle, and power-when-absent re-arming.

```bash
node --test 'www/tests/*.test.mjs'
```

Result: **34 tests passed**. Node emitted only the existing module-type warning for the vanilla frontend test setup.

```bash
go test ./internal/cctv/...
```

Result: **passed**.

```bash
go build ./...
```

Result: **passed earlier during integration**.

```bash
go vet ./...
```

Result: **not clean because of pre-existing go2rtc-tree vet findings outside `internal/cctv`**, including legacy `ReadByte`/`WriteByte` signatures, non-constant `fmt.Errorf` strings, undefined symbols in existing tests, unkeyed third-party structs, and unreachable code. This command was run and its failures were not caused by the CCTV integration changes.

A full hardware/browser E2E run was not possible because this sandbox had no live cameras, production stream service, optional model weights, or browser device. The unit/integration coverage uses fake frames/audio and contract-level event propagation; deployment still needs a two-camera RTSP smoke test.

## Known limitations and follow-up

- Face recognition and vehicle LPR remain silent until their model/gallery assets are downloaded and configured.
- Baby crying can use YAMNet when `ai/models/yamnet.onnx` exists; otherwise its documented harmonic fallback is used.
- Power-when-absent needs appliance ROIs calibrated for each camera.
- Optional model availability and camera/audio quality affect detection accuracy.
- `go vet ./...` still requires cleanup in unrelated upstream go2rtc packages before it can be a repository-wide gate.
- A real before/after latency measurement and browser E2E test remain deployment tasks requiring cameras and a running browser-capable environment.

## Exact full-system startup commands

From the repository root, create/configure `.env` from `.env.example`, install the AI requirements, and ensure FFmpeg is on `PATH`.

Terminal 1:

```bash
go build -o go2rtc .
./go2rtc
```

Terminal 2:

```bash
cd ai
python -m cctv_ai.worker --list
python -m cctv_ai.worker -v --cameras cam1,cam2
```

Open `http://localhost:1984/` for the dashboard. Use `python -m cctv_ai.worker --plugins <comma-separated-slots> --cameras cam1` to isolate a test run. Model download helpers write only to ignored model directories; never commit weights, camera URLs, credentials, or generated recordings.
