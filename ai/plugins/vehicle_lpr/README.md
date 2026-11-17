# Slot: Vehicle and licence plate recognition

**Status:** built and enabled. The worker runs it automatically; it stays silent if RapidOCR
is missing. Thresholds are tuned on synthetic footage, not every camera. Owner: Krithick
(plan [docs/plans/p2-krithick.md](../../docs/plans/p2-krithick.md)).

Detect vehicles at the gate and read their number plates. OCR:
[RapidOCR](https://github.com/RapidAI/RapidOCR) (`rapidocr-onnxruntime`, Apache 2.0 - PaddleOCR
det/cls/rec ONNX models bundled in the pip package, nothing to download). Read clustering:
[rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) (MIT). Follows Frigate's LPR mixin:
plate-shaped text boxes, two-line merge, Indian format check, many reads per vehicle.

## Events

`use_case: "vehicle_lpr"`, one event per vehicle track, label `vehicle`:

```json
{"plate": "TN09AB1234", "vehicle": "car", "reads": 7, "known_as": null}
```

`reads` is the number of agreeing OCR reads behind the plate; `known_as` is a name from
`data/known_plates.yaml` when the plate matches within Levenshtein distance 1, else `null`.

## Files

| File | What |
| --- | --- |
| `plates.py` | cleaning, Indian format check, position-based OCR fixes, plate-shaped boxes, two-line merge |
| `clustering.py` | Jaro-Winkler clustering of repeated reads (ported from Frigate `mixin.py`, MIT) |
| `ocr.py` | RapidOCR wrapper + CLAHE night boost |
| `data/known_plates.yaml` | optional, git-ignored: `{"Principal": ["TN09AB1234"]}` |

## Running

```bash
cd ai
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m cctv_ai.worker --plugins vehicle_lpr -v
```

Events show up at http://localhost:1984/api/cctv/events?use_case=vehicle_lpr

## How it works

1. **Demo mode (today):** the worker only has `on_frame`, so RapidOCR runs on the whole frame at
   most once per second per camera; only plate-shaped text boxes (area >= 1000 px, width:height
   1.8..10 - OCR boxes hug the glyphs so they run wider than the plate itself, or stacked
   two-line bike plates) become reads.
2. **Real mode (when the shared tracker lands):** set `needs = {"objects"}` (one line) and
   `on_objects()` takes over - car/motorcycle/bus/truck objects only, `true_positive` only, OCR on
   the vehicle crop (upscaled up to 2x when small, CLAHE when dark), reads kept per track id
   (last 25), event when >= 3 reads agree or the track ends; stationary vehicles are skipped
   once read (and the skip is cleared when the track ends).
3. A read survives cleaning (A-Z0-9), the Indian format check
   (`^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$` or Bharat `^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$`), position
   based OCR fixes (O->0, I->1, S->5, B->8 in digit slots and the reverse in letter slots) and an
   0.85 average confidence gate. Two-line plates are always assembled top-to-bottom (OCR returns
   lines by confidence, not position). Reads with Jaro-Winkler similarity >= 0.85 cluster; the
   biggest cluster wins and its most confident read is the plate.

## Tuning notes

| Knob | Default | Meaning |
| --- | --- | --- |
| `min_read_confidence` | 0.80 | Frigate uses 0.9, but RapidOCR's average character confidence often sits at 0.8-0.9 even on clean plates; the real precision guard is agreeing reads + the format check |
| `cluster_threshold` | 0.85 | raise if different plates merge into one cluster |
| `confirm_reads` | 3 | reads that must agree before an event |
| `min_plate_area` | 1000 px | text boxes below this are signs, not plates |
| `prepare_crop` min_height / max_zoom | 200 px / 2.0x | upscale of small vehicle crops before OCR (real mode) |

## Results

(write hits / misses / false alarms from test clips here)

| Clip | Expected plate | Got |
| --- | --- | --- |
| synthetic (tests/integration_check.py) | TN09AB1234 | TN09AB1234, 3 reads conf 0.89, PASS |
