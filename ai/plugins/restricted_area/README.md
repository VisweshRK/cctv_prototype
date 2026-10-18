# Slot: Restricted area

**Status:** not built yet. Take it by opening a branch `usecase/restricted_area` (see `CONTRIBUTING.md`).

## Goal
Raise an event when a person enters, or stays too long in, a marked zone (polygon) of the image.

## Events to send
`use_case: "restricted_area"`, labels: `zone_entered`, `loitering` (zone name in `data.zone`)

## What to copy from Frigate
- `frigate/config/camera/zone.py`: zone polygon, `inertia` (frames inside before it counts) and `loitering_time`.
- `frigate/track/tracked_object.py`: checks the *bottom centre* of the person box against the zone, not the whole box.

## Suggested approach
1. Zones as a dict `{camera: {zone_name: [(x, y), ...]}}` in relative 0..1 coordinates.
2. Reuse the person detector from `intrusion` (share code, don't copy it).
3. `cv2.pointPolygonTest` on the bottom-centre point of each person box.

## Where your code goes
- `__init__.py`: fill in `setup()` (load models once) and `on_frame()`.
- Put helper code in this folder. Model files (`*.onnx`, `*.pt`, `*.tflite`) are git-ignored, so add a
  `download_models.py` or link in this README instead of committing them.
- Add a test in `ai/tests/test_restricted_area.py` with a sample image/sound (small files only).
- Set `enabled = True` when it works.

## Try it
```bash
cd ai
python -m cctv_ai.worker --plugins restricted_area -v
```
Events show up at http://localhost:1984/api/cctv/events?use_case=restricted_area
