# All-Use-Cases Prototype Design

Date: 2026-11-18
Scope: the seven use cases in the user brief: fall, baby crying, face recognition, LPR, camera tampering, zone occupancy, and energy saving.

## Intent and acceptance

Provide one local CCTV website that an operator can use to see both live cameras, enable and tune all seven use cases, enroll people, configure zones and appliance regions, review/acknowledge alerts, and see camera/model/audio/CPU health. Live video must continue independently of inference, event handling, or video analysis. The integrated branch is expected to run on CPU only from one documented command and be verifiable with unit/API/browser tests plus a 10-minute load measurement.

Fire and smoke is explicitly removed from this scope. Existing intrusion and restricted-area stubs are not among the seven requested use cases and will not be shown as ready features in the integrated UI.

## Approaches considered

1. **Keep main's stack and integrate into it (chosen).** Use go2rtc/Go for the server, RTSP and browser streaming; preserve its JSONL event store, REST/SSE schema and static vanilla-JS website. Keep the Python worker for CV/audio plugins, but expose configuration and gallery operations through same-origin Go routes and give every use case a bounded independent scheduler. This retains the architecture and stream path already exercised by the cameras.
2. **Replace the main app with the separate FastAPI/Vite platform.** This would import unrelated history, replace Go event/storage routes, and fork the camera stack; the integration brief explicitly says keep main's structure, so this is rejected.
3. **Leave the face test bench as a linked second app or iframe.** This is the smallest change but violates the one-website requirement and encourages duplicate camera connections; use its tested gallery operations as implementation reference and merge the required page into `www/` instead.

## Architecture

- **Video:** go2rtc remains the sole camera connection for browser playback. One capture reader per camera publishes only the latest decoded frame; capture never waits for inference. Browser detection overlays use event metadata and canvas rather than changing the stream pixels.
- **Inference:** one shared CPU person detector/tracker/zone pipeline serves fall, occupancy, and energy-saving. Face matching is performed only for new/changed face tracks. OCR starts only after a candidate vehicle/plate signal. Tampering uses a low-rate reference comparison. Baby crying uses a separate audio reader and rolling 1–2 second PCM windows. Each scheduler has a bounded queue and drops stale work. OMP/OpenCV/ONNX thread counts are capped; metrics report frame rate, inference duration, queue depth, drops, process CPU/RAM, and audio availability.
- **Events and persistence:** every plugin emits the existing Go/Python event shape to the existing event endpoint; event history, SSE, snapshots, acknowledgements, and CSV export use the one main event store. Configuration is persisted separately from event history and is read by the worker at runtime.
- **Website:** retain the main dashboard shell. Add first-party pages/sections for Face Gallery, LPR/watchlist, Tampering, Zones/Occupancy, Energy Saving, Events, Settings, Video Jobs, and Health. The test bench's gallery and live diagnostics are references; the final page and endpoints share the main website origin.
- **Startup:** a root PowerShell runner for Windows plus a POSIX shell runner/build path will set up the pinned Python environment, build Go, download checksum-verified model assets into ignored directories, then start the Go server and worker. `.env.example` remains credential-free; sample media backs automated tests only.

## Goals

- Seven selected use cases can run concurrently from a real RTSP camera with available video/audio tracks.
- One UI owns all configuration and operator flows; no dead controls or demo events in production mode.
- No unbounded frame/event/inference backlog; slow inference cannot block go2rtc or the browser live feed.
- CPU-only model setup, dependency versions, camera/microphone status, missing models, and unavailable detection paths are visible.
- Tests exercise plugins, API/config/gallery flows, browser operator paths, disconnect/reconnect, and a reproducible load run.

## Non-goals and constraints

- Do not import the separate FastAPI/Vite platform or replace go2rtc.
- Do not commit camera credentials, private face photos, model binaries, or generated recordings.
- Use local face images only for identities enrolled by the operator. The application stores local folders and compiled embeddings; no cloud recognition service is introduced.
- Network/model/hardware failures must surface as degraded status and not freeze video or mislabel a use case as healthy.
- A test can prove pipeline behavior and timing on its captured machine; it cannot guarantee accuracy for unseen camera placement, lighting, microphones, or CPU models.

## Decisions left to implementation

- Keep `data/events.jsonl` as the existing single event store unless the audit reveals a main-supported database; don't create a second event database.
- Adapt the existing face gallery builder rather than keep its separate application server.
- Use the already integrated OpenCV YuNet/SFace, RapidOCR, YOLOX person detector, and MoveNet sources. Verify licensing and checksums before adding any additional weights.
- For real-time performance, prioritize camera decode and dashboard playback, then shared person tracks, then face/tamper/energy/baby; throttle OCR and low-priority analytics before dropping live frames.
