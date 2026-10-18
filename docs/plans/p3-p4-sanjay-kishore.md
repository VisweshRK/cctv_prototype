# Plan: P3 Sanjay + P4 Kishore (intrusion, restricted area, occupancy, fall)

All four use cases are **people logic on top of the shared tracker**: you don't train a person
detector, you turn tracked people into decisions. Three of them share one piece of code, the zone
helper, so build that first and together.

Suggested split (swap if you like):
- **Sanjay**: zone helper + zone drawing tool, intrusion, restricted area.
- **Kishore**: occupancy, fall detection.

Read first: [../how-frigate-works.md](../how-frigate-works.md) sections 8, 9 and 10.

## Tech stack

| Part | Use | Licence |
| --- | --- | --- |
| People | the shared `TrackedObject` list (`on_objects`) | ours |
| Zones | `cv2.pointPolygonTest`, `pyyaml` for `ai/zones.yaml` | Apache / MIT |
| Fall pose | **RTMPose** via `rtmlib` (ONNX, runs on our person boxes) or **MoveNet** Lightning | Apache 2.0 |
| Runtime | `opencv-python`, `onnxruntime`, `numpy` | Apache / MIT |

No Ultralytics YOLO-pose (AGPL).

## 0. Zone helper (both of you, first)

A small shared-code PR (`ai/cctv_ai/zones.py`, reviewed by Viswesh) because intrusion, restricted
area, occupancy and power_when_absent all need it.

1. Load `ai/zones.yaml` (format in [README.md](README.md#shared-contract-code-against-this-now)):
   points are fractions of width/height, converted to pixels for each frame size.
2. `feet(box)` = bottom centre `((x1 + x2) / 2, y2)`. An object is inside a zone when its feet are
   inside the polygon (`cv2.pointPolygonTest(...) >= 0`). Feet, not centre: a person in front of a
   door is not inside the room behind it.
3. `ZoneTracker` per camera keeps, for each object id and zone:
   - `inside_count`: frames in a row inside. **Entered** when it reaches `inertia` (3). After
     entering, the object stays "in" until it is outside for `inertia` frames in a row (no flicker).
   - `inside_since`: time of entry, for loitering.
   - `objects` filter: only the labels listed for that zone.
4. Method: `update(objects, ts) -> list of (object, zone, change)` with change `entered`, `left` or
   `loitering`, and `count(zone)` for occupancy.
5. A zone drawing tool, `python -m cctv_ai.draw_zones cam1`: grab one frame, click points with the
   mouse (`cv2.setMouseCallback`), press Enter, print the YAML block with fractions.
6. Tests: hand-made objects walking in and out, standing on the border (no flicker), a box whose
   centre is inside but feet are outside (not inside).

## 1. Intrusion (`ai/plugins/intrusion`)

**"A person appears where nobody should be, during armed hours."**

1. Config in the plugin (or `zones.yaml`): which zones are intrusion zones, armed hours
   (e.g. 22:00–06:00), and `min_seconds` (0 or 1).
2. In `on_objects()`: for `true_positive` persons only, update the zone tracker. On `entered` during
   armed hours, send one event per object id: `label: "person"`,
   `data: {"zone": "door", "object_id": "cam1-17"}`, with the snapshot.
3. Don't send again for the same object id. A new id (a new person, or the same person after the
   tracker lost them for > 5 s) can alert again; add a 30 s per-zone cooldown to stop repeats.
4. Speed target: event within 1 s of the person entering. That means alert on `entered`, don't wait
   for the track to end (Frigate's "alert" behaviour).
5. Test clips: someone walks through the zone (1 event), someone walks past just outside it
   (0 events), a cat or a shadow (0 events: not a true-positive person).

## 2. Restricted area (`ai/plugins/restricted_area`)

**Like intrusion, but the area is always off-limits, and loitering matters.**

1. Zones with `loitering_time` (e.g. 5 s). Events:
   - `entered`: someone stepped into the restricted zone (optional, per zone).
   - `loitering`: inside for longer than `loitering_time` (Frigate counts frames:
     `loitering_time × fps`).
2. One event per object id per change. Put `seconds_inside` in `data`.
3. Later, with P2: skip people that face recognition named as allowed (the face plugin's result will
   be available on the event list; agree the exact way with P2 and Viswesh first).
4. Tests: walks in and out within 2 s (entered only), stands inside 10 s (entered + loitering).

## 3. Occupancy (`ai/plugins/occupancy`)

**How many people are in each zone (or the whole room) right now.**

1. Count `true_positive` persons per zone, **including stationary ones** (someone sitting at a desk
   is still in the room). The shared tracker keeps them; Frigate does the same.
2. Debounce: a new count is reported only after it is stable for 3 s, so a person walking past a
   pillar doesn't make the count jump 2 → 1 → 2.
3. Events:
   - `count_changed` when the stable count changes, `data: {"zone": "room", "count": 3}`.
   - `over_capacity` once when the count goes above a limit from the config, and again only after
     it has gone back under.
4. Keep the current counts in memory and send a summary every 5 minutes (`label: "summary"`) for
   graphs later. Don't send a count every frame.
5. P1's power_when_absent needs "is anyone here?". It uses the same tracker and zone helper
   directly, so it doesn't depend on your plugin's events.
6. Tests: 0 → 1 → 2 → 1 → 0 with hand-made objects; a 1-frame dropout doesn't change the count.

## 4. Fall detection (`ai/plugins/fall`)

Frigate doesn't do this. Approach: **pose on the person box + rules over time**.

### Architecture
```
on_objects: true_positive persons (5 fps from the shared pipeline)
   ─> box shape history (cheap, every frame) ─> suspicious? (box got much wider than tall, or dropped fast)
   ─> pose model on that person's crop (RTMPose / MoveNet) ─> torso angle, head height
   ─> "fallen" for ≥ 3 s and not getting up ─> one event per person track
```

### Steps
1. **Cheap signals first** (no model), from `TrackedObject.history`:
   - Aspect ratio: width / height > 1.2 (lying) vs < 0.6 (standing).
   - Fast drop: the top of the box moves down by more than ~40% of the standing height within 1 s.
2. **Pose only when suspicious** (keeps CPU low): run RTMPose (`rtmlib`, lightweight model) or
   MoveNet Lightning on the person crop. Features:
   - Torso angle: line from mid-shoulders to mid-hips; > 60° from vertical = lying.
   - Head below hips, or head near the floor line of the box.
3. **Decision**: `fallen` = (lying pose or lying box) for ≥ 3 s **after** a fast drop, and the
   person is stationary on the ground. Sleeping on a bed or sofa: mark those areas as an exclude zone
   in `zones.yaml` and ignore people whose feet are inside it.
4. One event per person track: `label: "fall"`, `confidence` = pose confidence,
   `data: {"seconds_down": 4.2, "torso_angle": 78}`, with the snapshot. If they get up within 3 s,
   no event.
5. **Plan B** if pose is unreliable on our cameras: a small "standing / sitting / lying" classifier
   on person crops (Frigate's custom object classification idea), trained on our own clips.
6. Test clips: act out falls (on a mattress), sitting down fast, bending to pick something up,
   lying on a bed. Record hits and false alarms in the folder README. Fall datasets such as UR Fall
   are useful for testing; check their licence and don't commit them.

## Done when

- Zone helper merged with tests; zones drawn for both cameras.
- Intrusion and restricted area: one event per person, within 1 s, nothing for people outside the
  zone.
- Occupancy: correct stable counts on a clip with 0–3 people, including someone sitting still.
- Fall: catches acted falls, no events for sitting down or bending, results written in the README.
