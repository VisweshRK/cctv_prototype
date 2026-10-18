# Plan: P2 Krithick (face recognition, vehicle + licence plate)

Both use cases follow the same Frigate pattern: **find the object with the shared detector, run your
model only inside that object's box, and decide by voting over many frames.** Viswesh helps with
model downloads, ONNX sessions and the voting code.

Read first: [../how-frigate-works.md](../how-frigate-works.md) sections 8, 9 and 11.

## Tech stack

| Part | Model / library | Licence |
| --- | --- | --- |
| Face detection | OpenCV Zoo **YuNet** (`cv2.FaceDetectorYN`), 5 landmarks | MIT |
| Face embedding | OpenCV Zoo **SFace** (`cv2.FaceRecognizerSF`), 128 numbers per face | Apache 2.0 |
| Plate text detection + OCR | **RapidOCR** (`rapidocr-onnxruntime`: PaddleOCR det/cls/rec in ONNX) | Apache 2.0 |
| String matching | `rapidfuzz` (Jaro-Winkler, Levenshtein) | MIT |
| Runtime | `opencv-python`, `onnxruntime`, `numpy` | Apache / MIT |

Avoid InsightFace's pretrained ArcFace weights (non-commercial licence), Ultralytics models (AGPL)
and YOLOv9 plate detectors (GPL).

## 1. Face recognition (`ai/plugins/face_recognition`)

### Architecture
```
on_objects: person, true_positive, not yet decided
   ─> crop the person box (top half) ─> YuNet (threshold 0.7) ─> face ≥ 750 px?
   ─> align with 5 landmarks (SFace alignCrop) ─> SFace embedding
   ─> cosine vs each known person's mean embedding ─> blur penalty ─> (name, score, face area)
   ─> per-track history ─> weighted vote ─> one event per person track
```

### Steps
1. **Gallery** (on your laptop only, never in git, and only people who agreed): `faces/<name>/*.jpg`,
   5–20 photos each, different angles and light. Better: photos taken by our own cameras.
2. **Build each person's mean embedding** (Frigate's `build_class_mean`):
   - Embed every photo (YuNet → align → SFace).
   - Drop photos whose cosine similarity to the person's mean is far below the others (blurry or
     wrong face), then take a 15% trimmed mean per dimension (`scipy.stats.trim_mean`, or sort and
     cut by hand).
   - Save the means to `data/faces.npz` (git-ignored). Rebuild when photos change.
3. **Per frame** in `on_objects()` (`heavy = True`, so it runs on its own thread):
   - Only `person` objects with `true_positive` and no decision yet.
   - Crop the person box; run YuNet on the crop (faster and more accurate than the whole frame).
   - Skip faces under 750 px area (too small to recognise).
   - Embed, compare with every mean by cosine. SFace's suggested match threshold is 0.363; map the
     similarity to a 0–1 confidence (Frigate uses a sigmoid) and treat anything below your
     "unknown" level as unknown.
   - Blur penalty: `cv2.Laplacian(face).var()` < 120 → −0.06, < 160 → −0.04, < 200 → −0.02,
     < 250 → −0.01.
4. **Voting per track** (Frigate's `weighted_average`):
   - Store (name, score, face_area) for each attempt on this track id.
   - weight = min(area, 4000) × (score − unknown_score) × 10. The best name needs at least
     `min_faces` votes and no tie with another name.
   - Accept when the weighted average ≥ `recognition_threshold` (start at 0.8; Frigate uses 0.9 with
     its own score scale; tune on our cameras).
   - Stop after 12 attempts, or 6 once recognised.
5. **Events** (one per person track): `label: "known_person"` with `data: {"name": ...}`, or
   `label: "unknown_person"` after 12 attempts with no match. Attach the best face snapshot.
6. Tests: the voting function with fake attempt lists; the embedding pipeline on 2–3 public test
   faces (for example OpenCV's sample images), not team photos.

## 2. Vehicle + licence plate (`ai/plugins/vehicle_lpr`)

### Architecture
```
on_objects: car / motorcycle / bus / truck, true_positive
   ─> skip if stationary and already read ─> crop vehicle box ─> CLAHE contrast boost (night)
   ─> text detection (RapidOCR det) ─> keep plate-shaped boxes ≥ 1000 px ─> merge 2-line plates
   ─> orientation + OCR (rec) ─> clean text ─> avg char confidence ≥ 0.9? ─> Indian format check
   ─> per-track reads (last fps × 5) ─> cluster by Jaro-Winkler ≥ 0.85 ─> best plate
   ─> one event per vehicle track
```

### Steps
1. **OCR on still photos first**: collect 30–50 photos of Indian plates (your own photos of parked
   cars are fine; blur faces). Run RapidOCR on the full photo, then on a crop of just the car. Note
   what fails (night, angle, two-line bike plates).
2. **Plate-shaped boxes**: from the text boxes, keep the ones with plate proportions (car plates
   roughly 2:1 to 5:1 wide; bike plates 1:1 to 2:1 with two lines) in the lower part of the vehicle.
   Merge boxes that are stacked with similar height and overlapping x (Frigate merges two-line plates
   this way). A dedicated plate detector (a small YOLOX trained on an open plate dataset) can come
   later if text detection is too slow or finds too many signs.
3. **Clean the text**: uppercase, remove spaces and symbols. Check the Indian format:
   - normal: `^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$` (TN09AB1234)
   - Bharat series: `^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$`
   Fix common OCR swaps by position: in the digit positions O→0, I→1, S→5, B→8; in the letter
   positions 0→O, 1→I, 5→S, 8→B. Minimum length 4 (Frigate's `min_plate_length`).
4. **Many reads per vehicle** (Frigate's `_get_cluster_rep`):
   - Keep the last `fps × 5` reads per vehicle track: (plate, avg confidence, area).
   - Group reads whose average Jaro-Winkler similarity ≥ 0.85; the biggest group wins; its most
     confident read is the plate.
5. **When to read**: only `true_positive` vehicles; skip stationary vehicles once a plate was
   accepted; run at most every second frame per vehicle (`heavy = True`).
6. **Known plates** (optional): `data/known_plates.yaml` (git-ignored), e.g.
   `{"Principal": ["TN09AB1234"]}`; match with Levenshtein distance ≤ 1.
7. **Events** (one per vehicle track, sent when the plate is confirmed by ≥ 3 agreeing reads, or when
   the track ends): `label: "vehicle"` with
   `data: {"plate": "TN09AB1234", "vehicle": "car", "reads": 7, "known_as": null}`.
8. Tests: format check and clustering with fake read lists; OCR on 3 small plate crops you made.

## Done when

- Faces: a known person is named correctly within ~2 s of facing the camera; strangers come out as
  `unknown_person`; one event per visit, not per frame.
- Plates: correct plate for most daytime cars at the gate on test clips, one event per vehicle;
  results (right / wrong / missed) written in the folder README.
