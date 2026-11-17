# Slot: Face recognition

**Status:** built and enabled. The worker runs it automatically; it stays silent if the
models or `data/faces.npz` are missing. Thresholds are tuned on synthetic footage, not every
camera. Owner: Krithick (plan [docs/plans/p2-krithick.md](../../docs/plans/p2-krithick.md)).

Recognise known people (family/staff) and flag unknown faces. Models:
[OpenCV Zoo **YuNet**](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (MIT)
for detection with 5 landmarks and
[OpenCV Zoo **SFace**](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
(Apache 2.0) for the 128-number embeddings. Voting over many frames follows Frigate's
`weighted_average`.

## Events

`use_case: "face_recognition"`, one event per person track (per "visit" in demo mode):

| label | confidence | data |
| --- | --- | --- |
| `known_person` | weighted vote score 0..1 | `{"name": "krithick"}` |
| `unknown_person` | best attempt score | - |

A face snapshot JPEG is written to `data/snapshots/face_recognition/` and its path goes in
`Event.snapshot`.

## Files

| File | What |
| --- | --- |
| `voting.py` | per-track weighted vote (ported from Frigate `face.py`, MIT) |
| `gallery.py` | mean embedding per person: outlier drop + 15% trimmed mean |
| `engine.py` | YuNet detection, SFace embedding, cosine match, blur penalty |
| `build_gallery.py` | CLI: `faces/<name>/*.jpg` -> `data/faces.npz` |
| `download_models.py` | CLI: fetch the two ONNX models into `ai/models/` |
| `faces/<name>/*.jpg` | your photo gallery - **git-ignored, laptop only, only people who agreed** |

## One-time setup

```bash
cd ai
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m plugins.face_recognition.download_models   # ~39 MB, SHA-256 checked
mkdir plugins\face_recognition\faces\<name>                        # 5-20 photos each
.venv\Scripts\python -m plugins.face_recognition.build_gallery     # writes data\faces.npz
```

## Running

```bash
.venv\Scripts\python -m cctv_ai.worker --plugins face_recognition -v
```

Events show up at http://localhost:1984/api/cctv/events?use_case=face_recognition

## How it works

1. **Demo mode (today):** the worker only has `on_frame`, so YuNet runs on the whole frame.
   Every face gets its own pseudo-track ("visit"), matched across frames by box overlap
   (IoU > 0.3), so two people in frame never mix their votes. A visit ends 10 s after the
   face disappears. One known event per name per 2 minutes.
2. **Real mode (when the shared tracker lands):** set `needs = {"objects"}` (one line) and
   `on_objects()` takes over - person objects only, `true_positive` only, crop the top half of the
   person box, faces under 750 px are skipped, one vote history per tracked object.
3. **Quality gate first:** an attempt is only voted on when the aligned 112x112 crop is sharp
   enough (Laplacian variance >= 70) and near-frontal (nose within 0.35 eye-distances of the
   eye-line midpoint). Blurry or profile faces keep the visit alive but never burn an attempt -
   this is what stops motion-blurred CCTV frames from producing wrong names.
4. Every attempt stores (name, score, face area). Weight = `min(area, 4000) * (score - 0.8) * 10`,
   so big confident faces count most and sub-unknown attempts drag the average down. Accept at
   `recognition_threshold` 0.8 after at least 3 agreeing votes with no count tie; strangers are
   reported after 12 attempts (or when the track ends). Cosine similarity maps to a confidence with a
   sigmoid centred on SFace's 0.363 match threshold; the blur penalty (up to 0.06) is measured on
   the aligned face crop, never the whole frame.

## Tuning notes

| Knob | Default | Meaning |
| --- | --- | --- |
| `recognition_threshold` | 0.8 | Frigate uses 0.9 on its own score scale; raise it if we get wrong names |
| `min_face_area` | 750 px | raise if far-away faces give false names |
| `unknown_score` | 0.8 | below this an attempt counts as "unknown" |
| `visit_timeout` | 10 s | demo mode only |
| `MIN_SHARPNESS` (engine) | 70 | Laplacian variance gate on the aligned crop |
| `MAX_POSE_OFFSET` (engine) | 0.35 | eye-distance gate for yaw; raise to accept more profile faces |

## Test bench

The web bench (`ai/test_bench`) shows per-face similarity, confidence, blur penalty and quality-gate
status live; its video analysis votes per face track (IoU-matched), exactly like the plugin. Run the
end-to-end acceptance check any time with:

```bash
.venv/Scripts/python -m tests.integration_check   # builds a synthetic clip, asserts the events
```

## Results

(write hits / misses / false alarms from test clips here)

| Clip | Expected | Got |
| --- | --- | --- |
| synthetic (tests/integration_check.py) | krithick2 known, stranger unknown, plate side n/a | known 0.99, 1 unknown, PASS |
