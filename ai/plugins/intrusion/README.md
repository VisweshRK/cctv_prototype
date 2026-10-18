# Slot: Intrusion detection

**Status:** not built yet. Take it by opening a branch `usecase/intrusion` (see `CONTRIBUTING.md`).

## Goal
Raise an event when a person appears on a camera during armed hours (e.g. night, or when everyone is away).

## Events to send
`use_case: "intrusion"`, labels: `person` (optionally `person_armed_hours` in `data`)

## What to copy from Frigate
- `frigate/motion/improved_motion.py`: cheap motion check first, so the person detector only runs on frames that changed.
- `frigate/video/detect.py`: how detections are run on motion regions.
- `frigate/track/tracked_object.py`: tracking one person across frames so one intruder = one event, not 30.

## Suggested approach
1. Start from `example_motion` (already detects change) and only run the detector when motion is found.
2. Person detector: a small YOLO model exported to ONNX, run with `onnxruntime` (add it to `requirements.txt` in your PR).
3. Add an armed schedule (start/end time) as a class attribute first; move to config later.
4. Send one event per new person, not one per frame (keep a short cooldown per camera).

## Where your code goes
- `__init__.py`: fill in `setup()` (load models once) and `on_frame()`.
- Put helper code in this folder. Model files (`*.onnx`, `*.pt`, `*.tflite`) are git-ignored, so add a
  `download_models.py` or link in this README instead of committing them.
- Add a test in `ai/tests/test_intrusion.py` with a sample image/sound (small files only).
- Set `enabled = True` when it works.

## Try it
```bash
cd ai
python -m cctv_ai.worker --plugins intrusion -v
```
Events show up at http://localhost:1984/api/cctv/events?use_case=intrusion
