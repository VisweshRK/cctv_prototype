# Camera tampering (image side)

**Status:** working (`enabled = True`). Owner: Viswesh.

Detects a camera that is **covered**, **blurred / sprayed** or **moved / turned away**.
**Disconnect is done in Go** (`internal/cctv/status.go` sends `camera_offline` / `camera_online` events with
`use_case: tampering`).

No model and no extra packages: only OpenCV and NumPy, about 13 ms per check on a 1920×2160 frame.

## Events

| Label | When | `data` |
| --- | --- | --- |
| `covered` | Image went dark, or one flat colour (hand, tape, cloth) | `detail`, `brightness`, `contrast` |
| `blurred` | Sharpness fell below 30% of this camera's normal (sprayed, smeared, out of focus) | `detail`, `sharpness`, `normal_sharpness` |
| `moved` | Few features still match the reference view (turned away), or the view's corners moved > 10% of the width (pan, tilt, rotation, zoom) | `detail`, `features`, `inliers`, `shift` |
| `restored` | The view is normal again after one of the above. Sent once. The dashboard shows it in the feed without a popup. | `detail`, `was`, `seconds` |

Every event also has `data.area = "Whole view"`. **One event per incident:** after `covered`, nothing more
is sent until `restored`.

## How it works

1 check per second (`interval = 1.0`). Each frame is shrunk to 320×180 grey (ORB needs more pixels than
160×90).

1. **Learn** what this camera normally looks like for the first 60 checks (60 s). No alerts while learning.
   The baseline is the median of the last 300 normal checks (5 min), so it follows slow light changes.
   Cameras burn a clock and a name onto the picture, and that text stays sharp and bright when the lens is
   covered or smeared. So the measures ignore a few percent of the picture:
   - brightness = **median** grey level;
   - contrast = **spread** of grey levels, `(90th − 10th percentile) / 2.56` (about the standard deviation);
   - sharpness = Laplacian variance per tile (8×6 tiles), the **75th-percentile tile**, divided by
     contrast². The clock's tiles are among the sharpest and are skipped; a mostly blank view still has a
     quarter of its tiles with real detail.
2. **Covered:** brightness < 15, or contrast < 8, while this camera's normal view is not itself dark
   (normal brightness ≥ 30) or plain (normal contrast ≥ 12). A camera that always looks dark never alarms.
3. **Blurred:** sharpness below 30% of normal. Dividing by the contrast means a darker room or lower
   contrast does not look like blur; only lost detail does.
4. **Moved:** ORB features of the current view are matched against a reference view (saved after learning,
   refreshed every 10 minutes of normal view) and `findHomography` (RANSAC) is fitted. Fewer than
   `max(12, 15% of the reference features)` matching = turned away. Otherwise the four corners are mapped
   through the homography: if they moved more than 10% of the width on average, the camera was moved.
   Corners rather than the centre, because a rotation leaves the centre where it was.
5. **Day/night is not tampering:** when the camera switches to IR the picture loses its colour (mean HSV
   saturation drops below 8; back to colour above 16). The plugin then learns a new normal for 10 checks
   instead of alerting. A covered lens is colourless too, so a dark or flat frame never triggers the
   re-learn.
6. **Persistence:** a problem must last 3 checks in a row (3 s) before an event, and the view must be
   normal for 3 checks before `restored`. Headlights or a hand passing by don't count.

The motion detector's `scene_changed` flag (plan step 2) is not built yet, so the moved check runs on every
check. It is cheap enough (a few ms).

## Settings (class attributes in `__init__.py`)

| Setting | Default | Meaning |
| --- | --- | --- |
| `learn_checks` | 60 | Checks to learn the normal view at start |
| `relearn_checks` | 10 | Checks to learn again after a colour ↔ IR switch |
| `persist` | 3 | Checks in a row before an event, and before `restored` |
| `dark` / `flat` | 15 / 8 | Brightness / contrast below this = covered |
| `plain` | 12 | A normal view with less contrast than this is not checked for flat colour or blur |
| `blur_ratio` | 0.3 | Sharpness below this share of normal = blurred |
| `min_features` | 30 | A reference with fewer ORB features (e.g. a blank wall) is not checked for moved |
| `min_inliers` / `inlier_ratio` | 12 / 0.15 | Fewer matching features than both = turned away |
| `max_shift` | 0.1 | Corners moved more than this share of the width = moved |
| `colour_on` / `colour_off` | 16 / 8 | Mean saturation for colour / IR mode |
| `reference_every` | 600 | Normal checks between reference refreshes |

## Results

Tests: `ai/tests/test_tampering.py` (synthetic scene, no footage needed): 17 tests, one of them with a
clock burned onto the picture.

| Clip | Expected | Got |
| --- | --- | --- |
| Synthetic scene, 30 normal checks, a small shake, a 40% darker room | nothing | nothing |
| Synthetic: black, flat colour | `covered` | `covered` |
| Synthetic: Gaussian blur σ=6 | `blurred` | `blurred` (sharpness 1% of normal) |
| Synthetic: shifted 25% sideways / upside down / another scene | `moved` | `moved` |
| Synthetic: colour → grey-scale (IR) and back | nothing | nothing |
| `samples/test.mp4`: normal / covered / flat colour / blur σ=5 / shifted / upside down / IR | nothing / `covered` / `covered` / `blurred` / `moved` / `moved` / nothing | same |
| Live worker on cam2 (looped clip), 100 s | nothing | nothing |
| `video_test.py` dual-lens video (below), 2 loops: hand over cam2, spray on cam1, cam2 turned, tape on cam1 | `covered` / `blurred` / `moved` / `covered`, each then `restored` | same, 2 s after each, on the right lens |
| Same video: someone walks past cam1 (2 s), both lenses switch to IR (30 s) | nothing | nothing |
| Same video live through go2rtc (split into cam1 / cam2) + worker + dashboard, one loop | as above | same: 8 events, each 4–5 s after the incident, popups on the right tile; IR switch only re-learns |

Still to do: try it on the real cameras (cover the lens, smear it, turn the camera, wait for dusk) and tune
the numbers above if needed.

Known limits:
- If the camera stays moved, the incident stays open (no new events). Restart the worker to accept the new
  view as normal.
- A room going fully dark without IR looks like `covered`. The Q-Tech cameras switch to IR, so this
  should only happen if IR takes more than 3 s.

Numbers on that video (normal → tampered): contrast 27–49 → 4.7 (hand) / 0.4 (tape); sharpness 0.25–0.41
→ 0.006 (spray; lowest normal, during the walk-past, 0.13).

## Try it
```bash
cd ai
python -m cctv_ai.worker --plugins tampering -v
```
Events show up at http://localhost:1984/api/cctv/events?use_case=tampering

## Test on a video (no camera needed)
`video_test.py` draws a fake dual-lens camera (two views stacked, 1920×2160, 10 fps, 5 minutes) with a
burned-in clock, people walking past, and scripted tampering (`SCRIPT` in the file), then runs the plugin on
each lens once per second of video and checks the events.
```bash
cd ai
python -m plugins.tampering.video_test make  ../samples/dual.mp4
python -m plugins.tampering.video_test check ../samples/dual.mp4
```
`check` exits 1 if an event is missing, late (> 5 s), or extra. Any other clip:
`check clip.mp4 --no-split --no-expect` just prints what fires.

To run the video through the whole system instead, set `DUAL_URL=ffmpeg:samples/dual.mp4#input=loop#video=copy`
and the dual-lens `CAM1_URL` / `CAM2_URL` lines from `.env.example`, start go2rtc and the worker with
`--cameras cam1,cam2`.
