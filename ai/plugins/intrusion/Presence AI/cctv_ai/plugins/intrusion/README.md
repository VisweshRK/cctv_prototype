# Intrusion Detection Plugin

> **Status:** implemented, pending integration testing.  
> **Branch:** `usecase/intrusion`

Raises an event when a **person** appears on a camera during **armed hours**
(e.g. night-time, or when everyone is away).

## Architecture

```
Frame
  │
  ▼
┌──────────────────┐   no motion    ┌──────────┐
│  Motion Detector │ ──────────────▶│  SKIP     │
│  (MOG2, cheap)   │                └──────────┘
└────────┬─────────┘
         │ motion regions
         ▼
┌──────────────────┐   no person    ┌──────────┐
│  Shared YOLOX    │ ──────────────▶│  SKIP     │
│  (detect regions)│                └──────────┘
└────────┬─────────┘
         │ person boxes
         ▼
┌──────────────────┐   existing     ┌──────────┐
│  IoU Tracker     │ ──────────────▶│  SKIP     │
│  (new vs seen)   │                └──────────┘
└────────┬─────────┘
         │ NEW person
         ▼
┌──────────────────┐
│  Event:          │
│  intrusion/person│
│  + armed_hours   │
└──────────────────┘
```

### Key ideas (from Frigate)

| Frigate file | What we took | Our file |
|---|---|---|
| `frigate/motion/improved_motion.py` | Cheap motion check first | [`motion.py`](motion.py) |
| `frigate/video/detect.py` | Run detector only on motion regions | [`__init__.py`](__init__.py) |
| `frigate/track/norfair_tracker.py` | Track one person = one event | [`tracker.py`](tracker.py) |

## Events emitted

```json
{
  "use_case": "intrusion",
  "camera": "cam_front_door",
  "label": "person",
  "confidence": 0.87,
  "data": { "person_armed_hours": true }
}
```

## Armed schedule

Set as class attributes in `__init__.py` (move to config later):

```python
armed_start = time(22, 0)   # 10 PM
armed_end   = time(6, 0)    #  6 AM
```

If `start > end`, the window wraps past midnight.

## Files

| File | Purpose |
|---|---|
| [`__init__.py`](__init__.py) | Plugin entry point: `setup()` + `on_frame()` |
| [`motion.py`](motion.py) | MOG2 background-subtraction motion detector |
| [`tracker.py`](tracker.py) | Simple IoU-based person tracker |
| [`config.yaml`](config.yaml) | (Optional) future config for zones/schedule |

## Model files

Model files (`*.onnx`, `*.pt`, `*.tflite`) are **git-ignored**.  
This plugin uses the **shared YOLOX detector** provided by `cctv_ai` — no
extra models to download.

## Running

```bash
cd ai
python -m cctv_ai.worker --plugins intrusion -v
```

Events show up at: `http://localhost:1984/api/cctv/events?use_case=intrusion`

## Testing

```bash
cd ai
python -m pytest tests/test_intrusion.py -v
```

## What the tests cover

- **Motion detector** – identical frames → no motion; changed region → motion boxes
- **IoU tracker** – new tracks returned; matched tracks suppressed; disappeared tracks pruned
- **Armed schedule** – night wrap, daytime range, boundary conditions
- **Plugin integration** – not armed → no event; no motion → no event; person detected → event; same person → no duplicate
