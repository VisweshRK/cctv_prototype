# Slot: Fall detection

**Status:** not built yet. Take it by opening a branch `usecase/fall` (see `CONTRIBUTING.md`).

## Goal
Detect a person falling and staying on the floor (elderly care).

## Events to send
`use_case: "fall"`, labels: `fall`

## What to copy from Frigate
- No fall detection in Frigate. Reuse its idea of running the heavy model only on `person` boxes from the tracker.

## Suggested approach
1. Pose model on each `person` box from the shared detector, e.g. MoveNet or RTMPose in ONNX (both Apache 2.0).
   Not YOLO-pose from Ultralytics: it is AGPL.
2. Fall = body angle goes from upright to horizontal quickly, then the person stays down for a few seconds.
3. Keep a short per-person history (last ~2 s of keypoints) to tell a fall from lying down slowly.

## Where your code goes
- `__init__.py`: fill in `setup()` (load models once) and `on_frame()`.
- Put helper code in this folder. Model files (`*.onnx`, `*.pt`, `*.tflite`) are git-ignored, so add a
  `download_models.py` or link in this README instead of committing them.
- Add a test in `ai/tests/test_fall.py` with a sample image/sound (small files only).
- Set `enabled = True` when it works.

## Try it
```bash
cd ai
python -m cctv_ai.worker --plugins fall -v
```
Events show up at http://localhost:1984/api/cctv/events?use_case=fall
