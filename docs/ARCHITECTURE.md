# PRESENCE B2C — ATOMIC ARCHITECTURE SPECIFICATION
## Complete AI CCTV Platform (10 Use Cases + Dynamic Customer Rules + Monitoring Website)

---

## 1. Executive Summary

Presence B2C is an integrated AI-powered CCTV platform designed for consumer and household safety monitoring. It connects Wi-Fi IP cameras (via RTSP/ONVIF or synthetic test streams) and executes 10 specialized computer vision and audio machine learning use cases.

Instead of 10 redundant streams, the platform employs a **single shared frame reader** and a **single shared object detector** per camera stream, feeding tracked entities to downstream safety modules.

---

## 2. Integrated Data Pipeline

```text
                           PRESENCE B2C
                                 |
                           Wi-Fi IP Camera
                                 |
                        RTSP / ONVIF / Synthetic
                                 |
                               go2rtc
                                 |
                 +---------------+---------------+
                 |                               |
           Video Pipeline                  Audio Pipeline
                 |                               |
        CameraFrameReader               AudioStreamReader
         (Single Decode)                  (16 kHz Mono)
                 |                               |
        +--------+--------+                    YAMNet
        |        |        |                      |
      Shared   Fire AI  Tampering          Baby Crying
     Detector (flame / (sharpness /
      (YOLOX)  smoke)   displacement)
        |
     Shared
     Tracker
     (Norfair)
        |
  Tracked Person / Vehicle Entities
        |
  +-----+-----------------+----------------+
  |                       |                |
  +-- Intrusion           +-- Occupancy    +-- Vehicle LPR
  +-- Restricted Area     +-- Electricity  +-- Face Recognition
  +-- Fall / Person-Down      Absence
                          |
                   All AI Events
                          |
                   Event Normalizer
                          |
                 Dynamic Rule Engine
                          |
                 FastAPI + Database
                          |
             +------------+------------+
             |            |            |
       React Website  WebSockets   Push Alerts
```

---

## 3. The 10 AI Prototype Use Cases

| # | Module ID | Name | Inference Modality | Output Events |
|---|---|---|---|---|
| 1 | `intrusion` | Intrusion Detection / Trespassing | Shared Person Tracking + Polygons | `PROTECTED_ZONE_ENTERED`, `PROTECTED_ZONE_EXITED` |
| 2 | `tampering` | Camera Tampering / Health | Laplacian Edge Var + Frame Diff | `CAMERA_COVERED`, `CAMERA_BLURRED`, `CAMERA_MOVED`, `CAMERA_STREAM_FROZEN`, `CAMERA_RESTORED` |
| 3 | `face_recognition` | Household Facial Recognition | Consent Biometrics + Cosine Sim | `ENROLLED_PERSON_RECOGNIZED`, `PERSON_NOT_RECOGNIZED` |
| 4 | `restricted_area` | Restricted Area Breach | Polygons + Loitering Timer | `RESTRICTED_ZONE_ENTERED`, `RESTRICTED_ZONE_LOITERING`, `RESTRICTED_ZONE_EXITED` |
| 5 | `fall` | Fall / Person-Down Assistance | Aspect Ratio Dynamics + Motionless | `POSSIBLE_FALL`, `PERSON_DOWN`, `PERSON_VISIBILITY_LOST` |
| 6 | `occupancy` | Room Occupancy Analytics | Headcount + Occlusion Debounce | `OCCUPIED`, `PROBABLY_EMPTY`, `OCCUPANCY_COUNT_CHANGED` |
| 7 | `power_when_absent` | Electricity During Room Absence | Occupancy + Appliance ROI Heuristics | `LIGHT_POSSIBLY_LEFT_ON`, `FAN_POSSIBLY_LEFT_ON`, `POTENTIAL_ENERGY_WASTE` |
| 8 | `fire` | Fire & Smoke Detection | HSV Flame Mask + Cooking Zone Context | `POSSIBLE_DANGEROUS_FIRE`, `EXPECTED_COOKING_FLAME`, `SMOKE_CANDIDATE` |
| 9 | `vehicle_lpr` | Vehicle Detection + License Plate | Vehicle Tracking + Indian HSRP OCR | `VEHICLE_DETECTED`, `PLATE_READ`, `REGISTERED_VEHICLE`, `UNRECOGNIZED_PLATE` |
| 10 | `baby_crying` | Baby Crying Monitor | 16 kHz Mono Audio RMS + FFT Harmonics | `BABY_CRY_STARTED`, `BABY_CRY_CONTINUES`, `BABY_CRY_ENDED` |

---

## 4. Shared Contracts

### 4.1. Shared Tracked Object (`TrackedObject`)
```python
@dataclass
class TrackedObject:
    id: str
    label: str
    box: tuple[int, int, int, int]
    score: float
    median_score: float
    true_positive: bool
    stationary: bool
    motionless_count: int
    first_seen: datetime
    last_seen: datetime
    history: list[tuple[int, int, int, int]]
```

### 4.2. Plugin Life Cycle Hooks
- `setup(config)`
- `on_frame(camera, frame, timestamp)`
- `on_audio(camera, samples, timestamp)`
- `on_objects(camera, frame, timestamp, objects, ended)`
- `shutdown()`

---

## 5. Dynamic Rule Engine Architecture

Customers configure flexible no-code rules with typed allowlisted operators (`EQ`, `NEQ`, `CONTAINS`, `GT`, `LT`, etc.).
Custom notifications support variable interpolation:
Example: `"Welcome Home! Child {metadata.person_name} came from school at {event.time}!"`
Rules include time-of-day schedules, cooldowns, and recipient groups.
Arbitrary user code execution is prohibited by design.
