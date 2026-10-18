# Slot: Vehicle and licence plate recognition

**Status:** not built yet. Take it by opening a branch `usecase/vehicle_lpr` (see `CONTRIBUTING.md`).

## Goal
Detect vehicles at the gate and read their number plates; flag unknown plates.

## Events to send
`use_case: "vehicle_lpr"`, labels: `vehicle`, `plate` (text in `data.plate`, known/unknown in `data.known`)

## What to copy from Frigate
- `frigate/data_processing/real_time/license_plate.py`: runs plate reading only on `car` objects.
- `frigate/data_processing/common/license_plate/mixin.py`: plate detection, OCR, and merging repeated reads of the same car into one result.

## Suggested approach
1. Vehicle + plate detector (YOLO, ONNX), then OCR on the plate crop (e.g. PaddleOCR models in ONNX).
2. Indian plate format check (e.g. `TN 01 AB 1234`) to fix common OCR mix-ups (0/O, 1/I, 8/B).
3. Send one event per vehicle visit, with the best plate read.

## Where your code goes
- `__init__.py`: fill in `setup()` (load models once) and `on_frame()`.
- Put helper code in this folder. Model files (`*.onnx`, `*.pt`, `*.tflite`) are git-ignored, so add a
  `download_models.py` or link in this README instead of committing them.
- Add a test in `ai/tests/test_vehicle_lpr.py` with a sample image/sound (small files only).
- Set `enabled = True` when it works.

## Try it
```bash
cd ai
python -m cctv_ai.worker --plugins vehicle_lpr -v
```
Events show up at http://localhost:1984/api/cctv/events?use_case=vehicle_lpr
