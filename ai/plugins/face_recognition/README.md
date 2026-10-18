# Slot: Face recognition

**Status:** not built yet. Take it by opening a branch `usecase/face_recognition` (see `CONTRIBUTING.md`).

## Goal
Recognise known people (family/staff) and flag unknown faces.

## Events to send
`use_case: "face_recognition"`, labels: `known_face` (name in `data.name`), `unknown_face`

## What to copy from Frigate
- `frigate/data_processing/real_time/face.py`: runs face recognition only on tracked `person` objects.
- `frigate/data_processing/common/face/detector.py` and `recognizer.py`: detection + embedding + matching.

## Suggested approach
1. Detect faces with OpenCV's `cv2.FaceDetectorYN` (YuNet) and get embeddings with `cv2.FaceRecognizerSF` (SFace): both ship with `opencv-python`, only the model files need downloading.
2. Keep a small folder of known faces (`ai/plugins/face_recognition/faces/<name>/*.jpg`, git-ignored for privacy) and compare embeddings with cosine similarity.
3. Put the face crop in `snapshot` when you save one.

## Where your code goes
- `__init__.py`: fill in `setup()` (load models once) and `on_frame()`.
- Put helper code in this folder. Model files (`*.onnx`, `*.pt`, `*.tflite`) are git-ignored, so add a
  `download_models.py` or link in this README instead of committing them.
- Add a test in `ai/tests/test_face_recognition.py` with a sample image/sound (small files only).
- Set `enabled = True` when it works.

## Try it
```bash
cd ai
python -m cctv_ai.worker --plugins face_recognition -v
```
Events show up at http://localhost:1984/api/cctv/events?use_case=face_recognition
