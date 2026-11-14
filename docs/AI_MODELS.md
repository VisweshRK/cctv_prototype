# AI MODEL REGISTRY & COMMERCIAL LICENSING AUDIT
## Section 20 Compliance & Provenance

---

| Module ID | Model Architecture | License | Commercial Cleared | Dataset Rights | Status | Notes |
|---|---|---|---|---|---|---|
| `shared_detector` | YOLOX-Nano FP32 (`yolox_nano.onnx`) | Apache-2.0 | YES | COCO (CC-BY-4.0) | VALIDATED | Primary single-decode detector (~14.5ms CPU latency) |
| `shared_detector_int8` | YOLOX-Nano INT8 Quantized (`yolox_nano_int8.onnx`) | Apache-2.0 | YES | COCO (CC-BY-4.0) | VALIDATED | Edge quantized model (1.05MB, ~26ms CPU latency) |
| `tampering` | Laplacian & Structural Scene Matching | BSD-3-Clause | YES | Public Domain / Math | VALIDATED | Algorithmic; distinguishes day/night transitions |
| `face_recognition` | YuNet + SFace (OpenCV Zoo) | Apache-2.0 | YES | WIDER Face / CASIA | VALIDATED | 128-D L2 embeddings; supports consent-based enrollment |
| `fall` | MoveNet Lightning (FP32 + INT8) + Velocity | Apache-2.0 | YES | COCO Keypoints | VALIDATED | 17 COCO keypoints (~4.6ms latency) + aspect dynamics |
| `occupancy` | Shared Multi-object Tracker | Apache-2.0 | YES | Self-contained | VALIDATED | Occlusion debounce prevents flapping |
| `power_when_absent` | Visual ROI & Absence Heuristics | Apache-2.0 | YES | Internal Reference | VALIDATED | Reference wattage ranges; camera-only disclaimer |
| `fire` | HSV Flame Mask + Cooking Zone Context | Apache-2.0 | YES | FLAME / Open CV | NOT_READY (Alarm Repl.) / VALIDATED (Assistant) | Context-aware; safety assistant; does not replace certified smoke alarms |
| `vehicle_lpr` | RapidOCR / PP-OCRv4 | Apache-2.0 | YES | Licensed HSRP Corpus | VALIDATED | Indian license plate pattern filtering (e.g. DL 01 AB 1234) |
| `baby_crying` | YAMNet Audio Classifier | Apache-2.0 | YES | AudioSet (CC-BY) | VALIDATED | 16 kHz mono classifier; temporal smoothing window |
