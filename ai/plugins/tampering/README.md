# Camera tampering (image side)

**Status:** working (`enabled = True`). Owner: Viswesh.

Detects a camera that is **covered**, **blurred / sprayed**, **moved / turned away**, or **being shaken**
(handled, taken off its mount). People moving in front of it, even right at the lens, are not tampering.
**Disconnect is done in Go** (`internal/cctv/status.go` sends `camera_offline` / `camera_online` events with
`use_case: tampering`).

No model and no extra packages: only OpenCV and NumPy, about 13 ms per check on a 1920×2160 frame.

## Events

| Label | When | `data` |
| --- | --- | --- |
| `covered` | Image went dark, or one flat colour (hand, tape, cloth) | `detail`, `brightness`, `contrast` |
| `blurred` | Sharpness fell below 30% of this camera's normal (sprayed, smeared, out of focus), and fewer than a quarter of the detailed tiles are still sharp | `detail`, `sharpness`, `normal_sharpness`, `sharp_tiles`, `detailed_tiles` |
| `moved` | Few features still match the reference view (turned away), or the view's corners moved > 20% of the width (pan, tilt, rotation, zoom) | `detail`, `features`, `inliers`, `shift`, `anchored` |
| `shaking` | The whole picture jumped more than 2% of the width since the last check, and the background is no longer where it was (someone is handling or removing the camera) | `detail`, `jump`, `strength`, `anchored` |
| `restored` | The view is normal again after one of the above, or the camera stayed moved for 2 minutes and its new position was accepted (`accepted: true`). Sent once. The dashboard shows it in the feed without a popup. | `detail`, `was`, `seconds`, `accepted` |

Every event also has `data.area = "Whole view"`. **One event per incident:** after `covered`, nothing more
is sent until `restored`, except a **worse** problem: turned and then covered sends `moved`, then
`covered` (covered > blurred > moved > shaking). Shaken and then left pointing elsewhere sends `shaking`,
then `moved`. A lesser problem during an incident stays quiet.

## How it works

2 checks per second (`interval = 0.5`). Each frame is shrunk to 320×180 grey (ORB needs more pixels than
160×90).

1. **Learn** what this camera normally looks like for the first 120 checks (60 s). No alerts while learning.
   The baseline is the median of the last 600 normal checks (5 min), so it follows slow light changes.
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
   refreshed every 10 minutes of normal view). The reference is the per-pixel **median** of up to 30 views
   spread over time (every 4th learning check, then every 40th normal check), so people who were walking
   about while it learned drop out of it and only the room is left and `findHomography` (RANSAC) is fitted. Fewer than 12
   matching = turned away. Otherwise the four corners are mapped through the homography: if they moved more
   than 20% of the width on average, the camera was moved. With fewer than 15% of the reference features
   matching, the fit is rough (people close to the lens hide most of the view for a moment), so that
   `moved` must last 3 s instead of 1 s.
   Corners rather than the centre, because a rotation leaves the centre where it was.
   **Background anchors** come first: matches whose feature is still within 4 px (of 320) of where it was,
   outside the clock and name bands (top and bottom tile row). If at least 20 of them (and 10% of the
   reference features) are spread over 30% of the view, the camera has not moved, whatever is in front of
   it. People close to the lens hide part of the background but can't move it; a moved camera moves all of
   it. On the real cameras, frames with a head right at the lens still had 159–277 anchors.
   **Edge check** as a second vote, for plain rooms where ORB finds few corners (a wall and a cupboard
   gave only 7 anchors): phase correlation of the Sobel edge strength of the reference and the current
   view, clock and name bands left out (they stay put when the camera turns, and would pin the answer at
   0). Shift ≤ 1% of the width with strength ≥ 0.05 = not moved. Either vote saying "in place" is
   enough. On 24 live frames per camera with people about: in place 83–100% of the time (one reference
   frame instead of the median: 77%); the other room's frames: 0%.
4a. **Blur needs most of the view soft:** per tile, Laplacian variance / the tile's own variance, compared
   with that tile's normal. A person's plain shirt or a hand near the lens softens part of the view; a
   smeared lens softens all of it. `blurred` only if fewer than 25% of the detailed tiles (normal variance
   ≥ 64, clock bands left out) keep half their normal sharpness. Real frames with people: 16–26 of 26 tiles
   kept; a light smear (σ 0.7): 0.
4b. **Shaking:** phase correlation (`cv2.phaseCorrelate`, Hanning window) between this check's picture and
   the last one gives how far the whole picture moved. People moving leave the background still, so the
   peak stays at ~0 (≤ 0.5% on the real frames); a shaken camera moves everything together (≥ 4.7%).
   Jump ≥ 2% of the width, strength ≥ 0.1, and the background not anchored = `shaking`.
4c. **Wait while the picture is changing:** if more than 10% of the picture changed since the last check
   (80×45 thumbnails, > 20 grey levels after taking out an overall brightness change), a `blurred` or
   `moved` result is held: no event, and it doesn't count towards the 2 checks either way. Someone moving
   right at the lens changes 17–25% of the picture; a turned camera at rest or a smeared lens, ~0. So a
   real `moved` or `blurred` costs at most one extra check.
5. **Day/night is not tampering:** when the camera switches to IR the picture loses its colour (mean HSV
   saturation drops below 8; back to colour above 16). The plugin then learns a new normal for 20 checks
   instead of alerting. A covered lens is colourless too, so a dark or flat frame never triggers the
   re-learn.
6. **Persistence:** a problem must last 2 checks in a row (about 1 s) before an event, and the view must be
   normal for 2 checks before `restored`. A flash or a hand passing in under a second doesn't count.
7. **New position:** if the camera stays moved for 240 checks (2 min), `restored` is sent with
   `accepted: true` and the plugin learns the new view (20 checks). Tampering is caught again from there,
   and the old view now counts as moved.

The motion detector's `scene_changed` flag (plan step 2) is not built yet, so the moved check runs on every
check. It is cheap enough (a few ms).

## Settings (class attributes in `__init__.py`)

| Setting | Default | Meaning |
| --- | --- | --- |
| `interval` | 0.5 | Seconds between checks |
| `learn_checks` | 120 | Checks to learn the normal view at start |
| `relearn_checks` | 20 | Checks to learn again after a colour ↔ IR switch or a new position |
| `persist` | 2 | Checks in a row before an event, and before `restored` |
| `unsure_persist` | 6 | Checks in a row before a borderline `moved` (few features match, see step 4) |
| `accept_checks` | 240 | Checks moved before the new position is accepted as normal |
| `dark` / `flat` | 15 / 8 | Brightness / contrast below this = covered |
| `plain` | 12 | A normal view with less contrast than this is not checked for flat colour or blur |
| `blur_ratio` | 0.3 | Sharpness below this share of normal = blurred |
| `min_features` | 30 | A reference with fewer ORB features (e.g. a blank wall) is not checked for moved |
| `min_inliers` | 12 | Fewer matching features than this = turned away |
| `inlier_ratio` | 0.15 | Fewer than this share of the reference features matching = borderline `moved` |
| `max_shift` | 0.2 | Corners moved more than this share of the width = moved |
| `anchor_px` | 4 | A feature within this many pixels (of 320) of where it was is still in place |
| `min_anchored` / `anchored_ratio` / `min_spread` | 20 / 0.1 / 0.3 | Features in place, share of the reference, and how far apart they must be for "not moved" |
| `textured` | 64 | Tiles with less grey-level variance than this (plain wall) are not used for blur |
| `keep_ratio` / `min_kept` | 0.5 / 0.25 | A tile at this share of its normal sharpness is still sharp; this share of tiles still sharp = not blurred |
| `settle` | 0.1 | More than this share of the picture changed since the last check = wait |
| `shake_shift` / `shake_response` | 0.02 / 0.1 | Whole-picture jump (share of the width) and phase-correlation strength for `shaking` |
| `still_shift` / `still_response` | 0.01 / 0.05 | Edge check: offset from the reference (share of the width) and strength below / above which the camera has not moved |
| `colour_on` / `colour_off` | 16 / 8 | Mean saturation for colour / IR mode |
| `reference_every` | 1200 | Normal checks between reference refreshes |

## Results

Tests: `ai/tests/test_tampering.py` (synthetic scene, no footage needed): 24 tests, one of them with a
clock burned onto the picture.

| Clip | Expected | Got |
| --- | --- | --- |
| Synthetic scene, 30 normal checks, a small shake, a 40% darker room | nothing | nothing |
| Synthetic: black, flat colour | `covered` | `covered` |
| Synthetic: Gaussian blur σ=6 | `blurred` | `blurred` (sharpness 1% of normal) |
| Synthetic: shifted 25% sideways / upside down / another scene | `moved` | `moved` |
| Synthetic: colour → grey-scale (IR) and back | nothing | nothing |
| Synthetic: a person close to the lens walking fast back and forth, then sitting still | nothing | nothing |
| Synthetic: shaken back and forth (±30 px with motion blur), then left pointing elsewhere / put back / a single knock | `shaking` + `moved` / `shaking` + `restored` / nothing | same |
| `samples/test.mp4`: normal / covered / flat colour / blur σ=5 / shifted / upside down / IR | nothing / `covered` / `covered` / `blurred` / `moved` / `moved` / nothing | same |
| Live worker on cam2 (looped clip), 100 s | nothing | nothing |
| Synthetic: turned then covered / covered then smeared or turned / left turned for 2 min | `moved` + `covered` / `covered` only / `moved` + `restored` (accepted), then caught again | same |
| `video_test.py` dual-lens video (below), 2 loops: hand over cam2, spray on cam1, cam2 turned, tape on cam1 | `covered` / `blurred` / `moved` / `covered`, each then `restored` | same, 0.5 s after each, on the right lens (2 s with the old 1 check/s, persist 3) |
| Same video: someone walks past cam1 (2 s), both lenses switch to IR (30 s) | nothing | nothing |
| Same video live through go2rtc (split into cam1 / cam2) + worker + dashboard, one loop (old settings) | as above | same: 8 events, each 4–5 s after the incident, popups on the right tile; IR switch only re-learns |
| Live frames from both lenses (24 each, people about), replayed through the plugin: people moving, 72 checks | nothing | nothing |
| Same, the other lens's picture (same room, ~5% to the side) / panned 19% | nothing (under the 20% limit) | nothing; the 19% pan once gave `moved` + `restored` (borderline) |
| Real XMEye dual-lens camera, old settings | | turn caught on both lenses within 1 s of each other; 3–4 s delay; a head or feet close to the lens gave false `moved`, and a turned camera that stayed turned hid later tests (both fixed by the changes above) |

Still to do: try it on the real cameras (cover the lens, smear it, turn the camera, wait for dusk) and tune
the numbers above if needed.

Known limits:
- Someone standing still right at the lens for more than a second and hiding nearly all of the background
  can still give `moved` or `blurred`. Mount the camera high, looking at fixed things (walls, doors,
  furniture).
- A camera unscrewed slowly and smoothly (no jump between checks) gives `moved` once it stops, not
  `shaking`.
- cam2 burns yellow person boxes into its picture, and now and then sends a corrupt frame. Neither has
  given an alert, but they are the first thing to look at if cam2 alerts and cam1 doesn't.
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
