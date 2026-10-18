# Slot: Camera tampering (image side)

**Status:** not built yet. Take it by opening a branch `usecase/tampering` (see `CONTRIBUTING.md`).

## Goal
Detect a camera that is covered, blurred, sprayed or turned away. **Disconnect is already done in Go** (`internal/cctv/status.go` sends `camera_offline` / `camera_online` events with `use_case: tampering`).

## Events to send
`use_case: "tampering"`, labels: `covered`, `blurred`, `moved`

## What to copy from Frigate
- `frigate/motion/improved_motion.py`: its *calibrating* logic treats a sudden change of most of the image as 'scene changed' instead of motion. The same signal is a good 'camera moved' detector.

## Suggested approach
1. Covered: average brightness very low, or very low variance (one flat colour).
2. Blurred: variance of `cv2.Laplacian` drops far below this camera's normal value.
3. Moved: compare against a reference frame saved at setup (ORB features or a histogram); a big lasting difference = moved.
4. Only raise an event when the condition lasts a few checks in a row, to avoid false alarms from headlights.

## Where your code goes
- `__init__.py`: fill in `setup()` (load models once) and `on_frame()`.
- Put helper code in this folder. Model files (`*.onnx`, `*.pt`, `*.tflite`) are git-ignored, so add a
  `download_models.py` or link in this README instead of committing them.
- Add a test in `ai/tests/test_tampering.py` with a sample image/sound (small files only).
- Set `enabled = True` when it works.

## Try it
```bash
cd ai
python -m cctv_ai.worker --plugins tampering -v
```
Events show up at http://localhost:1984/api/cctv/events?use_case=tampering
