# How Frigate works (and why it is fast and accurate)

Read from Frigate's source at commit `add310d` (Oct 2026). Paths are relative to the Frigate repo
(https://github.com/blakeblackshear/frigate). Numbers are Frigate's defaults unless stated.

[frigate-teardown.md](frigate-teardown.md) maps Frigate to our 10 use cases. This page explains the
engine underneath: the per-frame loop that gives Frigate its accuracy and its fast response. Our
shared pipeline ([plans/viswesh.md](plans/viswesh.md)) copies this loop.

## 1. The big picture

```
camera ──RTSP──> go2rtc ──restream──> ffmpeg (1 per camera, decode to raw YUV, detect fps = 5)
                                          │  frames go into shared memory, only the slot name is queued
                                          ▼
                              per-camera process: motion ─> regions ─> detector ─> filters ─> NMS ─> tracker
                                                                          ▲
                                     one detector process for all cameras (queue of region tensors)
                                          │
                              tracked objects ─> zones / loitering / stationary ─> events + snapshots
                                          │
                              post-processors on tracked objects only: face, licence plate, classification
audio: ffmpeg (16 kHz mono) ─> RMS gate ─> YAMNet ─> audio events
```

Each camera connects once (to go2rtc). Everything after that works on decoded frames in memory.

## 2. Frame capture: decode once, never wait

`frigate/video/ffmpeg.py`, `frigate/video/detect.py`

- One ffmpeg process per camera reads go2rtc's restream and outputs **raw YUV420** frames at the
  detect size and detect fps (default **5 fps**). No JPEG encode/decode anywhere in the hot path.
- Frames are written into a ring of shared-memory slots. Only the slot name goes on a queue.
- If the processing queue is full, the frame is **dropped and counted** (`DroppedFrameTracker`)
  instead of queued. Latency stays bounded: Frigate always works on a recent frame, never a backlog.

**Why it matters for us:** our worker currently asks go2rtc for `/api/frame.jpeg` each tick. That is
an encode, an HTTP call and a decode per frame, and it caps us at about 1 fps. Fall detection and
"enters the zone then leaves" need about 5 fps.

## 3. Motion first: the detector only looks where something changed

`frigate/motion/improved_motion.py` (`ImprovedMotionDetector`)

1. Take the Y (grey) plane and resize it to a height of **100 px** (tiny, so this is very cheap).
2. Stretch contrast using the 4th and 96th percentiles, averaged over the last 50 frames.
3. Apply the motion mask, then a Gaussian blur (sigma 1).
4. `absdiff` against a running average background (`accumulateWeighted`, alpha **0.01**; **0.2**
   while calibrating).
5. Threshold at **30**, dilate, find contours, keep contours with area > **10**.
6. If more than **80%** of the frame changed (`lightning_threshold` 0.8: lights switched on, IR
   mode switch, camera moved), it does not report motion. It **recalibrates** the background
   instead. Calibration ends when motion is under 5% of the frame with 4 or fewer boxes.
7. The background is only updated after motion has continued for 10 frames, so a person standing
   still does not melt into the background immediately.

Output: a list of motion boxes in full-frame coordinates. No motion, no detection.

## 4. Regions: crop small squares, run the model at its native size

`frigate/video/detect.py` (`process_frames`), `frigate/util/object.py`

- For each frame, Frigate builds **regions**: square crops that cover
  - the boxes of objects already being tracked (so they keep being followed), and
  - motion boxes not already covered by a tracked object.
- Nearby boxes are clustered (`get_cluster_candidates`, `get_cluster_region`) so one region covers
  several close objects. The square is the longest edge × **1.35**, and **never smaller than the
  model size** (320 for a 320 model).
- `get_cluster_region_from_grid` uses a learned per-camera **region grid** (how big objects usually
  are at each place in the image; refreshed daily) to pick a good region size for new motion.
- At startup it scans the 8 most popular grid regions so objects that are already in the scene are
  found without waiting for motion.
- Each region is cropped from the frame and resized to the model input (`create_tensor_input`).

**Why this is both faster and more accurate:**
- Faster: the detector runs on a few 320×320 crops instead of a full frame, and on nothing when the
  scene is still.
- More accurate: a person 60 px tall in a 1920×1080 frame becomes about 2 px tall if you shrink the
  whole frame to 320. In a 320 crop around them, they stay 60 px or are even upscaled. Small and far
  objects are where most detectors fail; regions fix that.

## 5. Stationary objects are not re-detected

`frigate/video/detect.py`, `frigate/track/stationary_classifier.py`

- Each tracked object has a `motionless_count` (frames where its box barely moved, by IoU).
- Once `motionless_count >= stationary.threshold` and no motion box overlaps it, the object is
  **stationary**: Frigate stops running the detector on it and reuses its last detection. It is
  re-checked every `stationary.interval` frames.
- The stationary classifier compares IoU against thresholds per label (for example cars use
  `active_check_iou` 0.75) and can compare a 96×96 crop with normalised cross-correlation
  (keep stationary at NCC ≥ 0.90, active at < 0.85).

Result: a parked car or a person sitting at a desk costs almost nothing, but is **still tracked**
(it still counts for occupancy, it does not vanish and come back as a "new" object).

## 6. The detector: one process, many cameras

`frigate/object_detection/base.py`, `frigate/detectors/plugins/onnx.py`, `frigate/util/model.py`

- One detector process serves all cameras. Cameras put region tensors in shared memory and send
  a connection id on a queue; the detector writes up to **20** detections back into the camera's
  output buffer (`[class, score, ymin, xmin, ymax, xmax]`, normalised).
- Supported model types include YOLOX, D-FINE, RF-DETR, YOLO-NAS and SSD, on CPU, OpenVINO, CUDA,
  ROCm, Coral and more.
- YOLOX post-processing (`post_process_yolox`): decode grid/stride outputs, score = objectness ×
  class score, `cv2.dnn.NMSBoxes` at score **0.4** / NMS **0.4**.

## 7. Cleaning detections

`frigate/video/detect.py` (`reduce_detections`), `frigate/util/object.py` (`is_object_filtered`)

1. Map each region's boxes back to full-frame coordinates.
2. **Filters** per label: `min_area`, `max_area`, `min_score` (0.5), `min_ratio`/`max_ratio`
   (width/height), and a per-object mask tested at the **bottom centre** of the box.
3. **Per-label NMS** across regions: IoU **0.4** by default, **0.6** for cars (parked cars overlap).
4. A box that touches a region edge (the object was cut off) gets its confidence capped at **0.6**,
   so a full view of the same object wins.
5. Overlapping detections are consolidated; "attribute" labels (face, licence plate) are attached
   to the best parent object (person, car).

## 8. Tracking: the object keeps its identity

`frigate/track/norfair_tracker.py` (Norfair library, BSD-3)

- A Kalman filter per object (`OptimizedKalmanFilterFactory`) predicts where it will be next frame.
  Example settings: person R=3.4, Q=0.03, distance threshold 2.5; car R=2.5, Q=0.05, threshold 3.75.
- Custom distance: movement of the **bottom-centre** normalised by box width/height, plus change in
  width and height ratios. One more config uses a colour histogram for re-identification.
- `initialization_delay = max(fps/2, 2)`: a new object needs a few hits in a row before it exists.
  One-frame false positives never become tracked objects.
- `hit_counter_max = fps × 5`: an object can be missed for up to ~5 s (occlusion, a bad frame) and
  keep its id.

## 9. Accuracy: median score and "true positive"

`frigate/track/tracked_object.py`

- Each object keeps the last **10** scores; frames where it was not detected add **0.0**.
- `computed_score` = **median** of that history.
- The object is a **false positive until the median reaches `threshold` 0.7** (with each frame's
  `min_score` 0.5). Once true, it stays true for the life of the track.
- Nothing is alerted until the object is a true positive.

This is the single biggest reason Frigate has so few false alarms: a shadow that scores 0.75 for
one frame and 0.3 for the next never becomes an event; a real person who scores 0.65–0.85 does.

## 10. Zones, loitering and events

`frigate/track/tracked_object.py`, `frigate/config/camera/zone.py`

- A zone is a polygon. An object is "in" the zone when its **bottom centre** (the feet) is inside,
  tested with `cv2.pointPolygonTest`. Feet, not box centre, so a person standing in front of a door
  is not "in" the room behind it.
- **Inertia** (default 3): the object must be inside for 3 frames in a row before it counts as
  entered. Kills flicker at the zone border.
- **Loitering time** (seconds × fps = frames): alert only if it stays that long.
- Per-zone object filters and speed zones exist.
- **Events are per tracked object**, not per frame. One person walking through = one event, with
  start time, end time, zones visited and a best snapshot.
- **Best snapshot** (`is_better_thumbnail` in `frigate/util/image.py`): prefer one not cut off at
  the frame edge, then a score at least 5% better, then an area at least 10% larger.
- Severity: alert vs detection, based on labels and `required_zones`.

## 11. Post-processors: expensive models only on tracked objects

### Faces (`frigate/data_processing/real_time/face.py`, `.../common/face/recognizer.py`)
- Runs only for `person` objects. YuNet (`cv2.FaceDetectorYN`, threshold 0.7) on the **person
  crop**, not the whole frame. Faces smaller than `min_area` **750 px** are skipped.
- Face is aligned to a template with 5 landmarks (`estimateAffinePartial2D`, LMEDS) before
  embedding. Embedders: FaceNet (small) or ArcFace (large).
- Each known person has a **class-mean embedding** built from their photos with outlier removal
  (drop photos far from the mean by cosine, then a 15% trimmed mean per dimension).
- Cosine similarity to each mean is mapped to a confidence with a sigmoid; blurry faces lose up to
  0.06 (Laplacian variance < 120).
- Score ≤ `unknown_score` (0.8) = "unknown".
- **Voting per tracked person**: every attempt (name, score, face area) is stored. The weighted
  average uses weight = min(area, 4000) × (score − unknown_score) × 10. A name needs `min_faces`
  votes and no tie; it is accepted at `recognition_threshold` **0.9**.
- Limits: at most **12** attempts per person, or **6** once recognised. Cost drops to zero once a
  person is known.

### Licence plates (`frigate/data_processing/common/license_plate/mixin.py`)
- Runs only on car/motorcycle objects; skips stationary vehicles that were already read.
- Plate detection, then PaddleOCR-style text detection (DB) → orientation classifier → recognition
  (ONNX). Two-line plates are merged by grouping boxes with similar height and vertical overlap.
- Plate `min_area` 1000, `detection_threshold` 0.7, `recognition_threshold` **0.9** average
  character confidence, `min_plate_length` 4, optional `format` regex.
- **Many reads per vehicle**: the last `fps × 5` reads are kept, clustered by Jaro-Winkler
  similarity ≥ **0.85**; the biggest cluster wins, and its most confident read is the plate.
- `known_plates` matched with an edit distance of `match_distance` 1.

### Audio (`frigate/events/audio.py`)
- ffmpeg gives 16 kHz mono int16 audio in **0.975 s** chunks (15 600 samples, YAMNet's input).
- **RMS gate**: the model only runs if RMS ≥ `min_volume` **500** (int16 scale). Quiet rooms cost
  nothing.
- YAMNet (TFLite, 521 AudioSet classes). Top 20 labels above 0.5; each label also has a per-label
  `threshold` (default **0.8**) and must be in the camera's `listen` list.
- An audio event stays open until the label has not been heard for `max_not_heard` (30 s): one
  crying episode = one event.

## 12. Why it is fast, in one list

1. Decode once per camera, raw frames in shared memory, no JPEG in the loop.
2. Drop frames when behind, never build a backlog.
3. Motion gate on a 100-px-high grey image.
4. Detector runs on small region crops, and only where there is motion or a moving tracked object.
5. Stationary objects are not re-detected.
6. One shared detector process with hardware acceleration.
7. Face/plate/classification models run only on confirmed tracked objects, with attempt limits.
8. Audio model only runs when the sound is loud enough.

## 13. Why it is accurate, in one list

1. Regions keep small objects big enough for the model.
2. Filters (area, ratio, masks) remove impossible boxes.
3. Per-label NMS and edge-cut confidence cap.
4. Kalman tracker with an initialization delay: one-frame ghosts never become objects.
5. Median-of-10 score ≥ 0.7 before anything is a true positive.
6. Zones use the feet point, with inertia and loitering time.
7. Faces and plates are decided by voting over many frames, not one.
