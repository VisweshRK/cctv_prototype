# Slot: Baby crying

**Status:** not built yet. Take it by opening a branch `usecase/baby_crying` (see `CONTRIBUTING.md`).

## Goal
Detect a baby crying from the camera microphone.

## Events to send
`use_case: "baby_crying"`, labels: `baby_crying`

## What to copy from Frigate
- `frigate/events/audio.py`: reads only the audio track with ffmpeg (`-vn`) and runs the YAMNet sound classifier on ~1 s chunks; class `crying` is in `audio-labelmap.txt`.
- Unlike Frigate we reuse go2rtc's single camera connection (via its RTSP restream) instead of opening a second one to the camera.

## Suggested approach
1. YAMNet (TFLite or ONNX) on the 16 kHz mono chunks the worker already gives you in `on_audio`.
2. Skip quiet chunks first (RMS below a threshold) so the model runs rarely.
3. Event when 'Baby cry' scores high in several chunks within ~5 s.
4. Needs ffmpeg on PATH (the worker reads audio through it).

## Where your code goes
- `__init__.py`: fill in `setup()` (load models once) and `on_audio()`.
- Put helper code in this folder. Model files (`*.onnx`, `*.pt`, `*.tflite`) are git-ignored, so add a
  `download_models.py` or link in this README instead of committing them.
- Add a test in `ai/tests/test_baby_crying.py` with a sample image/sound (small files only).
- Set `enabled = True` when it works.

## Try it
```bash
cd ai
python -m cctv_ai.worker --plugins baby_crying -v
```
Events show up at http://localhost:1984/api/cctv/events?use_case=baby_crying
