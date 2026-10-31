# Implementation plans

One plan per person. Read [../how-frigate-works.md](../how-frigate-works.md) first: every plan reuses
the same Frigate ideas (motion first, region crops, tracker, median score, zones, voting).

| Plan | Who | Use cases |
| --- | --- | --- |
| [viswesh.md](viswesh.md) | Viswesh | Video pipeline, shared detector + tracker, camera tampering, model help for P2 |
| [p1-mohanraj.md](p1-mohanraj.md) | P1 Mohanraj | Fire and smoke, electricity used when nobody is there, baby crying |
| [p2-krithick.md](p2-krithick.md) | P2 Krithick | Face recognition, vehicle + licence plate |
| [p3-p4-sanjay-kishore.md](p3-p4-sanjay-kishore.md) | P3 Sanjay + P4 Kishore | Intrusion, restricted area, occupancy, fall |

## The architecture everyone builds on

```
go2rtc ──restream──> frame reader (ffmpeg, 5 fps, raw BGR) ──> motion ──> regions ──> shared detector (YOLOX)
                                                                                         │
                                                       filters + per-label NMS ──> tracker (Norfair, Kalman)
                                                                                         │
                                                      TrackedObject list (id, label, box, median score, ...)
                                                                                         │
      ┌──────────────┬──────────────┬───────────────┬───────────────┬──────────────────┬─┴─────────────┐
   intrusion   restricted_area   occupancy        fall       power_when_absent   face_recognition  vehicle_lpr
   (zones)        (zones)        (counts)        (pose)       (counts + lights)   (YuNet + SFace)   (OCR)

   raw frame ──> tampering, fire          audio (16 kHz) ──> baby_crying (YAMNet)
```

Rules that make this work:

- **One detector for everyone.** Plugins never run their own person/car detector.
- **Plugins get tracked objects, not raw boxes.** A tracked object has a stable `id`, so "one event
  per object" is easy and one-frame ghosts never reach your code.
- **Only act on `true_positive` objects** (median of the last 10 scores ≥ 0.7, like Frigate).
- **Heavy models only on objects, with attempt limits** (face, plate, pose), never on every frame.

## Shared contract (code against this now)

Until Viswesh's pipeline lands, write your logic against these shapes and test it with hand-made
objects. The names below are the contract; Viswesh will not change them without telling everyone.

```python
# ai/cctv_ai/objects.py (Viswesh)
@dataclass
class TrackedObject:
    id: str                       # "cam1-17": same id while the tracker follows this object
    label: str                    # "person", "car", "motorcycle", "bus", "truck", ...
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels of the frame you are given
    score: float                  # this frame's score (0 if predicted, not detected)
    median_score: float           # median of the last 10 scores (missed frames count as 0)
    true_positive: bool           # median_score has reached 0.7 at least once
    stationary: bool              # has not moved for a while (still tracked, not re-detected)
    motionless_count: int         # frames in a row without moving
    first_seen: datetime
    last_seen: datetime
    history: list[tuple[int, int, int, int]]  # last ~25 boxes (about 5 s at 5 fps), oldest first

# ai/cctv_ai/plugin.py: new hook (on_frame and on_audio stay as they are)
class Plugin:
    needs = {"objects"}           # ask for tracked objects
    def on_objects(self, camera: str, frame: np.ndarray, ts: datetime,
                   objects: list[TrackedObject], ended: list[TrackedObject]) -> list[Event]:
        """objects = tracked now; ended = objects the tracker lost since the last call."""
        return []
```

Zones live in `ai/zones.yaml` (committed: it holds no secrets). Points are fractions of the frame
width and height, so they still work if the resolution changes:

```yaml
cam1:
  door:
    points: [[0.10, 0.55], [0.45, 0.55], [0.45, 1.0], [0.10, 1.0]]
    objects: [person]
    inertia: 3            # frames inside before "entered" (Frigate default)
    loitering_time: 0     # seconds; 0 = alert on entry
```

The zone helper `ai/cctv_ai/zones.py` is written by P3 + P4 as a small shared-code PR reviewed by
Viswesh, because intrusion, restricted area, occupancy and power_when_absent all use it.

### Event fields the dashboard uses

The dashboard (`www/cctv.html`) reads these optional keys from an event's `data`. All are optional;
send them when you have them.

| Field | What it is | Example |
| --- | --- | --- |
| `data.area` | A human name for the place. Shown as the popup's area (falls back to `data.zone`, then "Whole view"). | `"Server room door"` |
| `data.box` | `[x1, y1, x2, y2]` as **0..1 fractions** of the frame, drawn on the snapshot. Leave it out if the image already has boxes drawn. | `[0.31, 0.22, 0.58, 0.94]` |
| `data.detail` | The one-line popup subtitle. | `"1 person in zone for 6 s"` |

`data.box` is in fractions, but `TrackedObject.box` above is in **pixels**: divide x by the frame
width and y by the frame height before you send it.

`POST /api/cctv/events` also takes an optional top-level `image`: a JPEG up to about 2.9 MB
(base64 inflates it ~4/3 within the 4 MB body limit). It becomes the event's snapshot (served at
`api/cctv/snapshot?id=<event id>&t=<event time>`) and is never stored in the event itself. Without
it the server grabs a frame from the camera. In Python, pass the frame: `Go2rtc.send_event(event, image=frame)`.

`data` may be at most **64 KiB** of JSON; a bigger one is rejected with `400`. The server owns
`id`, `siren`, `acknowledged_at` and `snapshot`: values sent for them are ignored.

```json
{"camera": "cam1", "use_case": "intrusion", "label": "person", "confidence": 0.91,
 "data": {"area": "Server room door", "box": [0.31, 0.22, 0.58, 0.94], "detail": "1 person in zone for 6 s"},
 "image": "/9j/4AAQSkZJRg..."}
```

**What pops up** (`USE_CASES` and `describe()` in `www/cctv-logic.js`): every use case opens the
centre popup, except these, which go only to the alert feed:

- `occupancy` (any label);
- `face_recognition` with label `known_face` (only unknown faces pop up);
- `vehicle_lpr` without `data.plate` (a vehicle pops up only with a plate, as "Vehicle · <plate>");
- `tampering` with label `camera_online` ("Camera back online") or `restored` ("Camera view restored").

**What rings the camera siren** (`internal/cctv/cctv.go` defaults, `siren.go`): `intrusion`,
`restricted_area` and `fire`. The siren plays on the two-way-audio speaker of **the camera that saw
it** only, until a guard acknowledges every such alert on that camera (or for at most 3 minutes).
The list can be changed with `cctv.siren.use_cases` in the go2rtc config.

Full design: [../superpowers/specs/2026-10-19-demo-ui-design.md](../superpowers/specs/2026-10-19-demo-ui-design.md).

## Order of work

| Step | Viswesh | P1 | P2 | P3 + P4 |
| --- | --- | --- | --- | --- |
| 1 | Frame reader at 5 fps, metrics | Baby crying (needs only audio, starts now) | Face gallery + SFace matching on photos | `zones.py` + zone drawing tool, hand-made-object tests |
| 2 | Motion + regions + YOLOX detector | Fire model choice + dataset + training | Plate OCR on still photos of plates | Intrusion + restricted area logic |
| 3 | Tracker + `TrackedObject` + `on_objects` hook | Lights-on detector for power_when_absent | Per-track voting (faces, plates) | Occupancy counts, fall pose features |
| 4 | Tampering plugin; camera hookup | Plug into tracked objects | Plug into tracked objects | Plug into tracked objects, tune on real clips |
| 5 | Performance pass with all plugins on | Tune thresholds on clips | Tune thresholds on clips | Tune thresholds on clips |

Step 1 for everyone except Viswesh needs no detector. Nobody waits.

## Shared tech stack

| Need | Use | Licence |
| --- | --- | --- |
| Model runtime | `onnxruntime` (CPU; try the OpenVINO or DirectML build later) | MIT |
| Image code | `opencv-python` | Apache 2.0 |
| Object detector | YOLOX-Tiny / YOLOX-S ONNX from Megvii | Apache 2.0 |
| Tracker | `norfair` (what Frigate uses) | BSD-3 |
| Config | `pyyaml` | MIT |
| Faces | OpenCV Zoo YuNet + SFace | MIT / Apache 2.0 |
| Plates OCR | RapidOCR (PaddleOCR models in ONNX) | Apache 2.0 |
| Pose (fall) | RTMPose via `rtmlib`, or MoveNet | Apache 2.0 |
| Audio | YAMNet converted to ONNX | Apache 2.0 |
| String matching | `rapidfuzz` (Jaro-Winkler, Levenshtein) | MIT |

Each person adds their packages to `ai/requirements.txt` in their own PR, with a minimum version.

## What "done" means for every use case

On top of the checklist in [../TEAM_GUIDE.md](../TEAM_GUIDE.md) section 6:

- **Accuracy:** test on at least 3 clips that show the incident and 3 that don't. Write the
  results (hits, misses, false alarms) in your folder README.
- **Speed:** your plugin's time per call is logged by the worker. Keep it under 30 ms on average
  (heavy models: under 100 ms, and only on objects).
- **Fast response:** the event reaches `/api/cctv/events` within 2 s of the incident starting
  (intrusion and fire: within 1 s once the pipeline runs at 5 fps).
