# Integration Notes

Date: 2026-11-17
Branch: `integration/all-usecases`
Base: `origin/main` at `bc73b65` (`Merge usecase/tampering into main`)

## Repository and branch verification

- The working tree was clean before work began.
- `integration/all-usecases` was created from `origin/main`.
- The remote branches named in the integration request are present:
  - `usecase/face_recognition`
  - `usecase/fall`
  - `usecase/occupancy`
  - `usecase/tampering`
  - `sanjaym83786-hash-patch-1`
  - `use-cases-fd,bc,es`
- `usecase/face_recognition`, `usecase/fall`, and `usecase/occupancy` have merge base `bc73b65`, the current main tip.
- `sanjaym83786-hash-patch-1` also has merge base `bc73b65` and adds only `ai/plugins/intrusion/intrusion.py`.
- `usecase/tampering` is already represented in main; its branch has a different older base and no remaining diff against main.
- `use-cases-fd,bc,es` has no merge base with main. It is an unrelated platform tree and will not be merged.

## Branch findings

### Face recognition branch

The branch adds the face-recognition and vehicle-LPR implementations, model-download scripts, tests, sample test images, a test bench, and additional dependencies. It also updates the shared plugin contract tests. The test bench is a standalone development tool and must remain isolated from the production worker/dashboard ports. The branch's requirements add `onnxruntime`, `rapidocr-onnxruntime`, `rapidfuzz`, and `PyYAML` in addition to the base NumPy/OpenCV/requests set.

### Fall and occupancy branches

Both branches add the same shared object pipeline: `detector.py`, `tracker.py`, `zones.py`, `objects.py`, `draw_zones.py`, and `testing.py`, plus the `objects` plugin need and `on_objects` hook in the shared contract/worker. Their shared files are materially identical except for the explicitly requested conflict details:

- Keep occupancy's `tracker.py`, which contains the `union`, `iou`, and `containment` helpers missing from fall's version.
- Keep fall's worker error isolation around detector/tracker processing.
- Review occupancy's three-line tampering change during the merge rather than accepting it blindly.
- Combine duplicate zone/plugin tests and keep the union of `.gitignore` entries without broad patterns that hide legitimate test assets.

The current main branch still has placeholder slots for fall and occupancy, so these branches are required for their implementations.

### Intrusion patch

`ai/plugins/intrusion/intrusion.py` is not a main-contract plugin. It is a standalone FastAPI/Ultralytics application with its own RTSP capture loop, HTTP endpoints, global event deque, environment configuration, and no `cctv_ai.plugin.Plugin` subclass. It imports `dotenv`, `fastapi`, and `ultralytics`, draws its own stream, and does not use main's zones/arming/event contract. It is therefore not safe to wire into `ai/plugins/intrusion/__init__.py` without a substantial rewrite. The integration will preserve the main intrusion slot and document this patch as reviewed but not adopted unless a contract-compatible implementation can be produced without importing the standalone server architecture.

### Flattened use-cases branch

`use-cases-fd,bc,es` is a separate FastAPI/Vite platform. Its handoff claims all ten use cases and reports tests for a different `StandardAIEvent`/`BasePlugin` architecture, but its fire, baby-crying, and power-when-absent plugins import the flattened branch's `ai.cctv_ai`, `packages.event_schemas`, and tracker/zone types. The code also contains hardcoded Windows paths and model probing conventions that are incompatible with main.

The three relevant implementations are useful source material only:

- Fire: HSV flame mask, smoke coverage, cooking-zone awareness, and persistence state.
- Baby crying: 16 kHz mono audio, RMS/harmonic fallback, optional YAMNet ONNX, and episode state.
- Power when absent: appliance ROIs, absence hysteresis, frame-based appliance signals, wattage reference estimates, and re-arming.

They will be ported into main's `Plugin`/`Event` contract, with event data shaped for `www/cctv-logic.js`. The flattened platform's API, web app, infrastructure, binaries, auth/rules/database, and duplicate use-case implementations will not be imported.

## Main contract observations

- Plugin discovery imports `ai/plugins/<slot>/__init__.py` and requires a class named `Plugin` subclassing `cctv_ai.plugin.Plugin`.
- Main currently has all ten requested slot folders plus `example_motion`; only `tampering` and `example_motion` are enabled by default.
- The current worker fetches frames synchronously through the HTTP JPEG endpoint, runs all due plugins serially, and sends events synchronously. This is the suspected source of live-camera contention and will be measured before and after the performance changes.
- The event contract is `camera`, `use_case`, `label`, `confidence`, optional `time`, `snapshot`, and `data`.
- The dashboard already has entries for all ten requested use cases and existing helpers for `box`, `area`/`zone`, `detail`, occupancy counts, and energy waste. Ported event fields must match these helpers.

## Planned merge/port order

1. Merge face recognition.
2. Merge occupancy, then fall, resolving the shared-core differences above.
3. Review the intrusion patch and only adopt a contract-compatible implementation.
4. Port fire, baby crying, and power when absent without merging the unrelated-history branch.
5. Update frontend logic/tests, then measure and fix worker/Go/browser latency.
6. Run the requested automated and end-to-end checks, recording only commands actually executed and their results in `docs/INTEGRATION_REPORT.md`.
