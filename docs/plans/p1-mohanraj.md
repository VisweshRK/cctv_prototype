# Plan: P1 Mohanraj (fire and smoke, electricity when nobody is there, baby crying)

Three use cases, one theme: **"something is on that shouldn't be"** (a fire, the lights, a crying
baby nobody is attending to). Start with baby crying: it needs only audio, so you can finish it
before the shared detector exists.

Read first: [../how-frigate-works.md](../how-frigate-works.md) sections 3 (motion), 9 (median score)
and 11 (audio).

## Tech stack

| Use case | Model | Licence | Libraries |
| --- | --- | --- | --- |
| Baby crying | YAMNet (521 sound classes), converted to ONNX | Apache 2.0 | `onnxruntime`, `numpy`, `tf2onnx` (one-off conversion) |
| Fire and smoke | YOLOX-Nano or YOLOX-Tiny trained on a fire/smoke dataset | Apache 2.0 (model code); check the dataset's licence | YOLOX repo for training (Colab/Kaggle GPU), `onnxruntime` to run |
| Electricity | No model: tracked people + image brightness rules | — | `opencv-python`, the zone helper |

No Ultralytics fire models (AGPL), see rule 12 in the team guide.

## 1. Baby crying (`ai/plugins/baby_crying`)

### How Frigate does it
ffmpeg gives 16 kHz mono audio in 0.975 s chunks. If RMS < `min_volume` (500), skip. Otherwise
YAMNet scores 521 classes; a label counts if its score ≥ 0.8. The event stays open until the label
hasn't been heard for 30 s, so one crying episode = one event.

### Architecture
```
on_audio(1 s int16 @ 16 kHz) ─> RMS gate ─> YAMNet (15 600 samples) ─> "Baby cry, infant cry" score
       ─> sliding window (last 3 chunks) ─> state machine: quiet -> crying (event) -> quiet after 30 s
```

### Steps
1. Get YAMNet's TFLite model and `yamnet_class_map.csv` (Kaggle Models / TF Hub, Apache 2.0).
   Convert once on your laptop: `python -m tf2onnx.convert --tflite yamnet.tflite --output yamnet.onnx`.
   Add the download + convert steps to a `download_models.py` in your folder. Don't commit the model.
2. In `setup()`: load the ONNX session and the class map. Find the indices of
   `Baby cry, infant cry` (main), and `Crying, sobbing` and `Whimper` (supporting).
3. In `on_audio()`:
   - `rms = sqrt(mean(samples.astype(float32) ** 2))`. If `rms < 500`, treat as silence (tune:
     200 = sensitive, 1000 = only loud).
   - Take the last 15 600 samples, divide by 32768 → float32 in [-1, 1], run YAMNet.
4. Smoothing (better than Frigate's single-chunk trigger): keep the last 3 scores. Start an
   incident when at least 2 of the last 3 are ≥ 0.6 for baby cry (tune 0.5–0.8).
5. State machine per camera: `quiet` → `crying` (send one event, `label: "crying"`,
   `confidence` = best score, `data: {"seconds": ..., "peak_rms": ...}`) → back to `quiet` after 30 s
   without crying. While crying, send nothing else.
6. Tests: a few 5 s WAV clips (baby crying, TV talking, silence, a dog barking) kept on your laptop;
   in git, test the state machine with fake score sequences.
7. Accuracy check: try ESC-50 `crying_baby` clips (research licence: use for testing only, don't
   commit) and a recording near the real camera; camera microphones sound very different from clean
   datasets.

### Later
- Fire alarm sounds: YAMNet also has `Smoke detector, smoke alarm` and `Fire alarm`. When the fire
  plugin wants them, move the YAMNet code to `ai/cctv_ai/audio_classifier.py` (shared-code PR,
  Viswesh reviews) so YAMNet runs once per chunk for both plugins.

## 2. Fire and smoke (`ai/plugins/fire`)

Frigate only does fire by sound. Visual fire is our addition.

### Architecture
```
on_frame (2 per second) ─> fire/smoke detector (YOLOX-Nano, 416) ─> per-region tracking by IoU
      ─> median score over the last 10 checks ─> persists ≥ 3 s ─> one event per incident
```
Fire flickers and smoke drifts, so single frames are unreliable; the median and persistence rules
(Frigate section 9) remove most false alarms (sunsets, red clothes, a lamp).

### Steps
1. Dataset: an open fire/smoke detection dataset such as D-Fire (check its licence and write it in
   the folder README). Add 200–500 of your own **negative** frames from our cameras (lamps, red
   objects, sunlight, a TV) as images with no boxes.
2. Train YOLOX-Nano or YOLOX-Tiny with the official YOLOX repo (Apache 2.0) on Colab or Kaggle
   GPU: two classes, `fire` and `smoke`, input 416. Export to ONNX with YOLOX's `export_onnx.py`.
3. Post-process the same way as the shared detector (grid decode + NMS). Copy Viswesh's YOLOX
   post-processing function once it lands; until then, YOLOX's `demo/ONNXRuntime` example shows it.
4. In `on_frame()` at `interval = 0.5`:
   - Run the model on the full frame (letterboxed). Fire is usually large; small distant fire can use
     region crops later.
   - Match boxes to the previous checks by IoU > 0.3 to follow the same fire.
   - Keep the last 10 scores per fire (0 when missing), take the median.
   - Raise `fire` or `smoke` when the median ≥ 0.6 and it has lasted ≥ 3 s.
5. One event per incident; re-arm after 60 s with no fire. Put the box and a snapshot in the event.
6. Cheap pre-check (optional, saves CPU): skip the model when the frame has no fire-coloured pixels
   (HSV hue 0–35, high saturation and value) and no smoke-grey change since the last check.
7. Test on fire videos (candles, a lighter, a fire video played on a phone in front of the camera),
   and on negatives (red shirt, sunset, orange lamp). Record hits and false alarms in the README.

## 3. Electricity used when nobody is there (`ai/plugins/power_when_absent`)

Frigate doesn't have this. Rule: **power is on AND no person for N minutes → one event.**

### Architecture
```
on_objects ─> people in the room (tracked persons, including stationary ones)
on_frame   ─> "power on?" signals: lights on, screen on, fan moving (per-camera ROIs)
      ─> timer: empty AND power on for 10 min ─> event "power_wasted" ─> re-arm when someone enters
```

### Steps
1. Use the shared tracker, not raw detections: a person sitting still is still tracked
   (`stationary`), so the room isn't "empty" just because they stopped moving. Count only
   `true_positive` persons. Use the zone helper if only part of the frame is the room.
2. "Lights on" without extra hardware:
   - Mark one or two ROIs in `ai/zones.yaml` on the light fixture or a wall it lights.
   - At setup, record the average brightness with lights off and on (a small calibration script).
     Lights are on when the ROI brightness is closer to the "on" value.
   - **Night mode:** when the Q-Tech camera switches to IR, the image turns grey (saturation ≈ 0).
     A grey IR image means the room lights are off.
3. "Screen on": a bright rectangle ROI on a TV/monitor. "Fan on": constant motion inside a fan ROI
   (use frame differences inside the ROI only).
4. Timer per camera: empty and any power signal on for `empty_minutes` (start with 10) → one event
   `label: "power_wasted"`, `data: {"lights": true, "screen": false, "empty_minutes": 12}`.
   Re-arm when a person is seen again or everything turns off.
5. Hysteresis: "empty" only after 60 s with no person, so someone leaving for a moment doesn't
   start the timer.
6. Until the shared tracker lands, test with hand-made object lists (`cctv_ai/testing.py`, see
   Viswesh's plan) and a clip where you switch the light on and leave.
7. Later (optional): read a smart plug (Shelly or Tasmota local HTTP API) for real power usage.

## Done when

- Baby crying: one event per crying episode on test recordings, none for TV talking or silence.
- Fire: catches the candle/lighter/phone-video tests within 3–5 s, no alarm on red objects or lamps.
- Electricity: event after N empty minutes with lights on; no event while someone sits still.
