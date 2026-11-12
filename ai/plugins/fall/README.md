# Slot: Fall detection

**Status:** Completed (`enabled = True`).
**Owner:** Kishore

## Goal
Detect a person falling and remaining on the floor, distinguishing falls from normal actions like sitting down quickly or bending down, and suppressing alerts when lying intentionally on a bed or sofa.

## Architecture: Two-Tier Cascade
1. **Tier 1 — Cheap Signal Screening (Every Frame @ 5 FPS, 0 ms model overhead)**:
   - Evaluates `TrackedObject.history`.
   - Checks aspect ratio: $w / h > 1.2$ (horizontal) vs $< 0.7$ (upright).
   - Checks downward velocity: top of bounding box drops by $> 40\%$ of standing height within $\le 1.0\text{ s}$.
   - Suspicious tracks are escalated to Tier 2. Normal upright tracks bypass pose inference completely, saving CPU.
2. **Tier 2 — Pose Estimation (Only on Suspicious Person Crops)**:
   - Uses Apache 2.0 MoveNet Lightning ONNX / RTMPose via `onnxruntime`.
   - **No AGPL YOLO-pose.**
   - Calculates torso angle from vertical: $\theta = \arctan2(\Delta x, \Delta y)$. An angle $> 60^\circ$ indicates a horizontal torso.
   - Evaluates head height relative to hips.
3. **State Machine & Persistence**:
   - `fallen` requires persistent down posture for $\ge 3.0\text{ s}$.
   - **Recovery**: If the person stands back up within 3.0 seconds (e.g. bending or quick recovery), the timer aborts and no event is emitted.
   - **Bed / Sofa Furniture Exclusion**: Checks `ai/zones.yaml` for zones with `exclude: true`. If a person's feet/box are in an exclude zone, fall alerts are suppressed.
   - **One Event per Episode**: Latches so repeated frames do not spam alerts. Re-arms only after the person stands upright for $\ge 3.0\text{ s}$ or the track ends.

## Events Emitted
- `label: "fall"`
- `data: {"object_id": "cam1-17", "seconds_down": 3.6, "torso_angle": 78.4, "confidence": 0.88}`
- `snapshot`: frame snapshot reference if enabled.

## Models
MoveNet SinglePose Lightning ONNX (~9 MB, Apache 2.0).
Download script:
```bash
python ai/plugins/fall/download_models.py
```

## Testing
```bash
cd ai
python -m unittest tests/test_fall.py -v
```

## Running
```bash
cd ai
python -m cctv_ai.worker --plugins fall -v
```
Events show up at http://localhost:1984/api/cctv/events?use_case=fall
