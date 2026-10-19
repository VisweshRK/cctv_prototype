# Slot: Fire and smoke

**Status:** not built yet. Take it by opening a branch `usecase/fire` (see `CONTRIBUTING.md`).

## Goal
Detect flames or smoke early.

## Events to send
`use_case: "fire"`, labels: `fire`, `smoke`

## What to copy from Frigate
- No fire model in the free Frigate models. Reuse the motion-first pattern from `frigate/motion/improved_motion.py` so the model runs less.

## Suggested approach
1. A fire/smoke model in ONNX, e.g. YOLOX trained on an open fire/smoke dataset (check the dataset's licence).
   Most ready-made fire models online are trained with Ultralytics and are AGPL: don't use those.
2. Require the detection in several frames in a row: flames flicker, a red shirt does not.
3. Low thresholds are OK here: a false fire alarm is cheaper than a missed one.

## Where your code goes
- `__init__.py`: fill in `setup()` (load models once) and `on_frame()`.
- Put helper code in this folder. Model files (`*.onnx`, `*.pt`, `*.tflite`) are git-ignored, so add a
  `download_models.py` or link in this README instead of committing them.
- Add a test in `ai/tests/test_fire.py` with a sample image/sound (small files only).
- Set `enabled = True` when it works.

## Try it
```bash
cd ai
python -m cctv_ai.worker --plugins fire -v
```
Events show up at http://localhost:1984/api/cctv/events?use_case=fire
