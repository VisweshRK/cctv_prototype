# Slot: Occupancy counting

**Status:** not built yet. Take it by opening a branch `usecase/occupancy` (see `CONTRIBUTING.md`).

## Goal
Count how many people are in each camera's view (or zone) and report when the count changes.

## Events to send
`use_case: "occupancy"`, labels: `occupancy` (count in `data.count`)

## What to copy from Frigate
- `frigate/camera/activity_manager.py`: keeps per-camera and per-zone object counts and only publishes when they change.

## Suggested approach
1. Reuse the person detector from `intrusion`.
2. Smooth the count (e.g. the most common value over the last 5 checks) so it doesn't flicker.
3. Send an event only when the smoothed count changes. `power_when_absent` uses these events to know if a room is empty.

## Where your code goes
- `__init__.py`: fill in `setup()` (load models once) and `on_frame()`.
- Put helper code in this folder. Model files (`*.onnx`, `*.pt`, `*.tflite`) are git-ignored, so add a
  `download_models.py` or link in this README instead of committing them.
- Add a test in `ai/tests/test_occupancy.py` with a sample image/sound (small files only).
- Set `enabled = True` when it works.

## Try it
```bash
cd ai
python -m cctv_ai.worker --plugins occupancy -v
```
Events show up at http://localhost:1984/api/cctv/events?use_case=occupancy
