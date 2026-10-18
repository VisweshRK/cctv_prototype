# Plan: Viswesh (pipeline, shared detector, tampering, help P2)

You own the shared code: `ai/cctv_ai/`, `internal/cctv/`, `go2rtc.yaml`, camera setup. Your work
is what gives the whole team Frigate's speed and accuracy, because every video plugin sits on top
of it. The engine you are copying is described in [../how-frigate-works.md](../how-frigate-works.md)
sections 2–9.

## What you deliver

1. A **frame reader** that gives each camera's frames at 5 fps with no JPEG round trip.
2. A **motion gate** and **region crops** (Frigate's `ImprovedMotionDetector` and region logic).
3. One **shared YOLOX detector** for both cameras.
4. A **tracker** that turns detections into `TrackedObject`s with median scores and stationary flags.
5. The **`on_objects` plugin hook** from [README.md](README.md#shared-contract-code-against-this-now).
6. **Metrics** so everyone can see fps, dropped frames and per-plugin time.
7. The **tampering** plugin (covered, blurred, moved).
8. **Model helpers** for P2 (download scripts, one place to create ONNX sessions).
9. **Camera hookup** for the two Q-Tech cameras.

## Tech stack

Python 3.12, `numpy`, `opencv-python`, `onnxruntime` (CPU first, then `onnxruntime-openvino` on an
Intel laptop or `onnxruntime-directml` on any Windows GPU), `norfair` (BSD-3), `pyyaml`, ffmpeg.
Model: **YOLOX-Tiny** (416×416, Apache 2.0) from the YOLOX GitHub releases; YOLOX-S (640) if the
laptop has a GPU.

## Architecture

```
ai/cctv_ai/
  frames.py      FrameReader: one ffmpeg per camera, raw BGR at 5 fps, keeps only the newest frame
  motion.py      MotionDetector: port of Frigate's ImprovedMotionDetector
  regions.py     motion boxes + tracked boxes -> square regions >= model size
  detector.py    Detector: YOLOX ONNX, batch of regions in, list of {label, box, score} out
  models.py      onnx_session(path): picks providers, sets threads; download + SHA-256 check
  tracker.py     Norfair tracker with Frigate's distance and Kalman settings -> TrackedObject
  objects.py     TrackedObject dataclass (the shared contract)
  pipeline.py    per camera: frame -> motion -> regions -> detect -> filter -> NMS -> track
  metrics.py     counters + timings, logged every 30 s
  worker.py      runs pipeline per camera, calls plugins (on_frame / on_objects / on_audio)
```

Threads: one frame-reader thread and one pipeline thread per camera, one detector shared by both
(guarded by a lock, or a small queue). Heavy plugins (face, plate) run on their own thread with a
queue of size 1: if they are busy, the newest objects replace the waiting ones (drop, don't queue).

## Step by step

### Step 1: frame reader (fixes speed first)

Today `client.frame()` calls `/api/frame.jpeg`: go2rtc encodes a JPEG, sends it over HTTP, we decode
it. That is slow and stuck near 1 fps.

1. In `frames.py`, start ffmpeg on go2rtc's restream (the camera keeps a single connection):
   ```
   ffmpeg -hide_banner -loglevel error -rtsp_transport tcp -i rtsp://localhost:8554/cam1
          -an -vf fps=5,scale=1280:-2 -pix_fmt bgr24 -f rawvideo -
   ```
   Read exactly `width × height × 3` bytes per frame from stdout.
2. A reader thread keeps only the **newest** frame (a one-slot buffer). If the pipeline was busy,
   the old frame is overwritten and a `dropped` counter goes up (Frigate's `DroppedFrameTracker`).
3. Restart ffmpeg with a back-off (1 s, 2 s, 5 s, max 30 s) when it exits; use the existing
   `Online` check so an offline camera doesn't spin.
4. If the Q-Tech camera has a sub-stream (often 640×360), use it for detection and keep the main
   stream for live view, like Frigate's `detect` vs `record` roles. Add `cam1_sub` in `go2rtc.yaml`.
5. Keep `/api/frame.jpeg` as a fallback (`--frames jpeg`).
6. Test: log the real fps and latency (time the frame was read vs now) for 5 minutes on a fake
   camera.

### Step 2: motion gate

Port `frigate/motion/improved_motion.py` to `motion.py` (credit Frigate's MIT copyright line if you
copy code, rule 11):

1. Grey frame, resize to height 100.
2. Contrast stretch with percentiles 4 / 96 averaged over 50 frames.
3. Mask, Gaussian blur (sigma 1), `absdiff` with the running average (`accumulateWeighted`, alpha
   0.01; 0.2 while calibrating), threshold 30, dilate, contours with area > 10.
4. If more than 80% of the frame changed, recalibrate and return no motion (lights on, IR switch).
   Expose a `scene_changed` flag: tampering uses it.
5. Scale the boxes back to full-frame pixels.
6. Unit test with two synthetic frames (a white square appears → one box in the right place).

### Step 3: regions

1. Inputs: motion boxes, and boxes of tracked objects that are not stationary.
2. Cluster boxes that are close (Frigate's `get_cluster_candidates`), make one square per cluster:
   side = longest edge × 1.35, at least the model size (416 for YOLOX-Tiny), clipped to the frame.
3. Startup and every 30 s with no motion at all: one full-frame pass (letterboxed to 416) so
   objects already in the scene are found. This replaces Frigate's region grid for now; the grid can
   come later if small far objects are missed.
4. Stationary objects: skip them unless motion overlaps them or every 10th frame (Frigate's
   `stationary.interval`); reuse their last box.

### Step 4: shared detector

1. `models.py`: `onnx_session(path)` creates an `InferenceSession` with the best available provider
   and `intra_op_num_threads` = half the CPU cores. A `download_models.py` fetches each model from its
   release URL into `ai/models/` (git-ignored) and checks its SHA-256.
2. `detector.py` for YOLOX ONNX:
   - Pre-process: resize each region to 416×416 keeping the aspect ratio, pad with 114, BGR, float32,
     no normalisation, NCHW. Stack regions into one batch if the model allows a dynamic batch;
     otherwise loop.
   - Post-process like Frigate's `post_process_yolox`: grid/stride decode, score = objectness ×
     class, `cv2.dnn.NMSBoxes` at score 0.4 / NMS 0.4.
   - Map boxes back to full-frame pixels. Keep only COCO labels we use: person, car, motorcycle,
     bus, truck, bicycle (plus anything a plugin asks for).
3. Clean-up in `pipeline.py` (Frigate's `reduce_detections` and `is_object_filtered`):
   - Filters per label: `min_score` 0.5, `min_area`, `max_area`, width/height ratio limits, and an
     optional mask tested at the box's bottom centre.
   - Boxes touching a region edge (but not the frame edge): cap the score at 0.6.
   - Per-label NMS over all regions: IoU 0.4, car 0.6.
4. Measure: inference ms per region and regions per frame. Target: both cameras at 5 fps under
   ~50% CPU on the laptop. If not, drop to 3 fps or try OpenVINO.

### Step 5: tracker and `TrackedObject`

1. `tracker.py`: one Norfair `Tracker` per camera and per label group, with Frigate's settings:
   - Distance: bottom-centre displacement normalised by box width/height, plus width and height
     ratio changes (copy `frigate/track/norfair_tracker.py`'s `distance`, with credit).
   - Person: Kalman R=3.4, Q=0.03, distance threshold 2.5. Vehicles: R=2.5, Q=0.05, threshold 3.75.
   - `initialization_delay = max(fps/2, 2)`; `hit_counter_max = fps × 5`.
2. For each track keep: last 10 scores (0.0 when missed), `median_score`, `true_positive` (median ≥
   0.7, sticky), `motionless_count` (IoU with the previous box > 0.9 → +1, else reset), `stationary`
   (motionless_count ≥ 10 at 5 fps = 2 s; tune), the last 25 boxes, first/last seen.
3. Report `ended` objects when Norfair drops a track.
4. Best snapshot per object (Frigate's `is_better_thumbnail`): not on the frame edge, then score
   +5%, then area +10%. Save JPEGs to `data/snapshots/` (git-ignored) so plugins can put the path in
   `Event.snapshot`.
5. Unit tests: a box moving 10 px per frame keeps one id; a single-frame box never becomes an
   object; a box missing for 3 frames keeps its id.

### Step 6: plugin hook and worker

1. Add `TrackedObject` (`objects.py`) and `on_objects(...)` (`plugin.py`) exactly as in the
   contract. Plugins with `needs = {"objects"}` get called once per processed frame.
2. Keep `on_frame` for plugins that want only pixels (tampering, fire).
3. Time every plugin call; log a warning if a plugin averages over 30 ms. Run plugins marked
   `heavy = True` on their own thread with a one-slot queue.
4. Tests: a fake pipeline that yields hand-made objects, so P1–P4 can run their tests without a
   model. Ship it as `cctv_ai/testing.py` (`make_object(label, box, ...)`, `fake_track(...)`).

### Step 7: metrics

Log every 30 s per camera: camera fps, frames processed, frames dropped, motion %, regions per
frame, detector ms, objects tracked, ms per plugin. Later, serve them from go2rtc
(`/api/cctv/metrics`) for a small dashboard.

### Step 8: tampering plugin (`ai/plugins/tampering`)

Disconnect is already done in Go. This is the image side. Run at 1 check per second.

1. Small grey frame (160×90). Learn a **baseline** over the first 60 s and keep a rolling median of
   normal values per camera.
2. **Covered**: mean brightness < 15 or standard deviation < 8 (one flat colour: hand, tape, cloth),
   while the baseline was normal.
3. **Blurred / sprayed**: `cv2.Laplacian(...).var()` below 30% of the baseline median.
4. **Moved / turned away**: keep a reference frame. Compare with ORB features + `findHomography`;
   few matches (or a large shift) = moved. The motion detector's `scene_changed` flag is a quick
   trigger for this check.
5. **Day/night switch is not tampering**: Q-Tech cameras switch to IR (grey image). If the frame
   becomes grey-scale (saturation ≈ 0) at the same moment, re-learn the baseline instead of alerting.
6. **Persistence**: the condition must hold for 3 checks in a row (3 s). One event per incident with
   label `covered`, `blurred` or `moved`; when it clears, send `restored` once.
7. Test images: a black frame, a heavy Gaussian blur of a sample frame, a shifted crop.

### Step 9: camera hookup (when the camera details arrive)

1. Find the RTSP URLs (Q-Tech often uses ONVIF; try go2rtc's ONVIF discovery at
   http://localhost:1984). Put them only in `.env` (`CAM1_URL`, `CAM2_URL`).
2. Check the main and sub stream resolutions and audio codec. Two-way audio goes through go2rtc's
   backchannel.
3. Record 1–2 minute clips per use case for the team (kept in `samples/`, never in git).

### Step 10: help P2 with models

- Put YuNet, SFace and the RapidOCR models in `download_models.py`.
- Make sure face and plate plugins run as `heavy = True` and only on true-positive objects.
- Pair on the per-track voting code (it is the same pattern for faces and plates).

## Done when

- Two cameras at 5 fps, < 1 s from event to `/api/cctv/events`, dropped frames < 5%.
- No detector runs when the scene is still (check the metrics).
- A person walking across the frame gives one track id from start to end.
- Tampering raises `covered`, `blurred` and `moved` on test clips, and nothing on a day/night switch.
