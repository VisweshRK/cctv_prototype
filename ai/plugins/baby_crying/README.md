# Baby crying

This use case reads 16 kHz mono audio from the camera's go2rtc RTSP restream and scores one-second
chunks with YAMNet. It uses the AudioSet `Baby cry, infant cry` class, smooths five consecutive scores,
then opens, continues, and closes a crying episode. It does not analyze video.

## Setup

Download the checksum-pinned, Apache-2.0 YAMNet ONNX file into the ignored `ai/models/` directory:

```powershell
cd ai
.venv\Scripts\python.exe -m plugins.baby_crying.download_models
```

The configured camera must expose a microphone audio track. The dual-lens `.env` configuration preserves
the source audio on cam1 with `audio=copy`; cam2 remains video-only so the worker does not process the
same microphone twice. Restart the Go2RTC server and AI worker after changing `.env`.

Check `GET /api/cctv/health`: the case is ready only when the model loads and audio arrives. If the camera
has no audio track, health reports it disabled. Do not treat camera speaker/two-way-audio support as proof
that its microphone is being sent in the RTSP stream.

## Events

The use case emits `crying`, `crying_continues`, and `crying_ended` events through the normal CCTV event
API. This is a general pretrained sound classifier, not a model trained specifically on this camera or
room; validate its alerts against representative audio before relying on them.

## Local checks

```powershell
cd ai
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_ported_plugins.py"
.venv\Scripts\python.exe -m plugins.baby_crying.download_models
```
