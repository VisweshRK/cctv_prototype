# Slot: Electricity used when nobody is there

**Status:** not built yet. Take it by opening a branch `usecase/power_when_absent` (see `CONTRIBUTING.md`).

## Goal
Warn when lights/fans/appliances are on while the room has been empty for a while.

## Events to send
`use_case: "power_when_absent"`, labels: `lights_on_empty_room`, `power_on_empty_room`

## What to copy from Frigate
- No equivalent in Frigate. Build on our own `occupancy` events.

## Suggested approach
1. Room empty: read recent `occupancy` events from go2rtc (`GET /api/cctv/events?use_case=occupancy`) or share the detector.
2. Lights on: average brightness of the frame clearly above the camera's night baseline (cameras switch to IR/black-and-white when the room is dark).
3. Later, add a smart-plug / energy meter reading in `data` for real power numbers.
4. Event only after the room has been empty with lights on for N minutes.

## Where your code goes
- `__init__.py`: fill in `setup()` (load models once) and `on_frame()`.
- Put helper code in this folder. Model files (`*.onnx`, `*.pt`, `*.tflite`) are git-ignored, so add a
  `download_models.py` or link in this README instead of committing them.
- Add a test in `ai/tests/test_power_when_absent.py` with a sample image/sound (small files only).
- Set `enabled = True` when it works.

## Try it
```bash
cd ai
python -m cctv_ai.worker --plugins power_when_absent -v
```
Events show up at http://localhost:1984/api/cctv/events?use_case=power_when_absent
