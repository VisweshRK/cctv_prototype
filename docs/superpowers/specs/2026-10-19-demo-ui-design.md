# Demo UI: live view + instant alerts (design)

Date: 2026-10-19 · Owner: Viswesh · Status: draft for approval

## Why

We build the frontend before the AI use cases so we can **demo the prototype to reviewers** early,
and so every teammate's use case has a place to show up the moment it sends its first event. The
reviewer must see, the instant something happens: **a popup with the snapshot, the timestamp, the
camera, the area and the confidence score**. It has to look like a professional security product,
not a student page (the first mockup was rejected as "poor in design").

The approved look is the mockup [2026-10-19-demo-ui-mockup.html](2026-10-19-demo-ui-mockup.html)
(open it in a browser and click "Simulate alert"), using the **centre popup** variant.

## Goals

1. One dashboard page served by our go2rtc binary: two live cameras, an alert feed, and a
   centre popup for every new alert.
2. **Instant push:** a new event reaches the open dashboard with no polling (target: under 1 s from
   `POST /api/cctv/events` to popup on screen, on the same LAN).
3. **Snapshots:** every alert shows a still image of the moment, served by the backend.
4. **Camera siren:** burglary (`intrusion`), breach (`restricted_area`) and `fire` alerts ring a
   siren on **the speaker of the camera that saw it** (two-way audio), until someone presses
   Acknowledge.
5. Works today with the fake cameras (`samples/test.mp4` looped through go2rtc via `.env`), and
   with the real Q-Tech cameras later by changing only `CAM1_URL` / `CAM2_URL` in `.env`.

## Non-goals (not in this version)

- Logins, users, permissions.
- Event search, history pages, video clips/recordings.
- Mobile layout (desktop/laptop browser, ≥ 1280 px wide, is enough for the demo).
- Any sound in the browser (decided: no dashboard siren).
- Drawing zones in the UI (zones stay in `ai/zones.yaml`).

## Anti-goals (would make this a failure even if it "works")

- The popup arrives late or needs a page refresh: the whole point is "instant".
- It looks like a plain admin page. Keep the mockup's quality: dark theme, real typography,
  severity colours, smooth popup animation, detection-box overlays on snapshots.
- A camera siren that can't be stopped, or keeps ringing all night when nobody is watching.
- Fake numbers shown to reviewers as if they were real (see "Dashboard numbers" below).
- Breaking go2rtc's own web UI (`index.html`, `stream.html`, ...) or its APIs.

## Constraints

- Lives in this repo (go2rtc fork). Backend changes only in our module `internal/cctv`; the page in
  `www/` so it is embedded in the binary by `www/static.go` (`//go:embed *.html`, `*.js`).
- No build step, no npm framework: plain HTML/CSS/JS like the rest of `www/`. Reuse go2rtc's
  `video-stream.js` / `video-rtc.js` player for live video.
- Only fonts/assets that work offline: the demo may run without internet, so no Google Fonts at
  runtime (use system fonts, or font files embedded in `www/`; licence must be MIT/Apache/OFL).
- Licence rule of the project: MIT/Apache/BSD only.
- Secrets stay in `.env`; nothing camera-related is hard-coded. No faces, videos or models in git;
  snapshots go to the git-ignored `data/` folder.

## What was decided (by Viswesh)

| Topic | Decision |
| --- | --- |
| Audience | Reviewers watching a demo: live view + alerts |
| Popup | Centre of the screen, over a dimmed background, with the snapshot, time, camera, area, confidence |
| Alert sound | Only for burglary (`intrusion`), breach area (`restricted_area`) and `fire` |
| Where it rings | Only on the camera that saw it, through its two-way-audio speaker. **No** sound in the dashboard |
| How long | Rings until someone presses Acknowledge |
| What pops up | Every use case, including vehicle plates; occupancy, known faces and "camera back online" go only in the feed |
| Camera input | Use whatever go2rtc streams as `cam1` / `cam2` (test video now, Q-Tech later) |

## How it works

### Page (backend gap 1: there was no UI page)

- URL: `http://localhost:1984/cctv.html` (file `www/cctv.html`, plus `www/cctv.js` / `www/cctv.css`
  if splitting helps). Add a link to it from go2rtc's `index.html` header so it is easy to find.
- Layout as in the mockup: left icon rail, top bar (title, cameras online, use cases armed, clock),
  two 16:9 live tiles, summary numbers, 12-hour timeline, detection-pipeline strip, and the
  "Live alerts" feed on the right. Rail icons other than "Live" can be inert in this version.
- Live tiles: go2rtc's `<video-stream>` element with `src=api/ws?src=cam1`. Overlay on each tile:
  LIVE badge, camera name and place, the use cases active on it, and an offline state (from
  `/api/cctv/status`) when the camera is down.
- On load: `GET /api/cctv/events?limit=50` fills the feed, then the page opens the live push
  stream. On reconnect it re-fetches the list (so nothing is missed while disconnected).

### Popup (one alert at a time)

- Which events pop up: see the "Popup" column in the table below.
- Shows: use-case icon + title, severity badge, a one-line detail, **snapshot** (16:9, timestamp
  burnt in the corner), **Time**, **Camera** (id + place name), **Area**, **Confidence** (% + bar),
  and "Siren: ringing on cam1 / camera speaker not available / off" when the use case has a siren.
- Buttons: **Acknowledge** (closes it, stops the siren, marks the event acknowledged on the
  server) and **Open live view** (closes the popup and enlarges that camera's tile).
  Clicking outside or ✕ also closes it, but **only Acknowledge stops the siren**.
- The tile of the camera that fired gets the red glow while its alert is open.
- If more alerts arrive while one is open: the newest is shown, with a "+N more" counter;
  acknowledging shows the next unacknowledged one (newest first).
- *(My call)* **One incident, one popup.** While an alert for the same camera + use case is still
  unacknowledged, new events for that pair join it instead of queuing a new popup: the popup shows
  "×N detections", switches to the newest snapshot, time and confidence, and Acknowledge
  acknowledges all of them. So a person loitering in a zone gives one popup, not twenty.
- *(My call)* **Page load / reconnect:** only events that arrive while the page is open pop up.
  Older unacknowledged events appear in the feed with an "Unacknowledged" badge (click to open and
  acknowledge), except a camera whose siren is still ringing: its alert pops up on load so it can
  be silenced.
- Clicking an item in the feed re-opens its popup.
- **Detection boxes** *(my call)*: a new optional contract field `data.box = [x1, y1, x2, y2]` as
  0..1 fractions of the frame (like `zones.yaml`). The page draws it as an overlay on the
  snapshot, with the label and confidence. If the worker sends an `image` with boxes already
  drawn, it leaves `data.box` out. "Simulate alert" sends a box, so the demo shows one. A second
  optional field, `data.detail`, is the popup subtitle (e.g. "1 person in zone for 6 s"); if it
  is missing, the page builds one from `label` and `data`. Both go into `docs/plans/README.md`.

### Severity, titles and siren per use case

| `use_case` | Severity | Title shown | Popup | Siren on camera |
| --- | --- | --- | --- | --- |
| `intrusion` | critical | Burglary alarm · intrusion | yes | yes |
| `restricted_area` | critical | Breach · restricted area | yes | yes |
| `fire` | critical | Fire / smoke detected | yes | yes |
| `fall` | high | Person fell | yes | no |
| `tampering` (incl. `camera_offline`) | high | Camera tampering / offline | yes | no |
| `tampering` label `camera_online` | info | Camera back online | feed only | no |
| `baby_crying` | medium | Baby crying | yes | no |
| `face_recognition`, unknown person | medium | Unknown person | yes | no |
| `face_recognition`, known person | info | Known: {name} | feed only | no |
| `power_when_absent` | medium | Power on, nobody here | yes | no |
| `occupancy` | info | Occupancy changed | feed only | no |
| `vehicle_lpr` | info | Vehicle · {plate} | yes | no |

**Popup rule (decided by Viswesh):** everything pops up except occupancy changes, known faces and
"camera back online", which only go in the feed. Vehicle plates do pop up.
Unknown `use_case` values still show, as "info" with the raw use case as the title, feed only. The
table lives in one place in the JS (and the siren list in one place in Go) so it is easy to change.

### Area and camera names

- **Area** = `data.area` if the event has it, else `data.zone`, else "Whole view". Plugins should
  send a human name in `data.area` (e.g. "Server room door"); this is added to the shared contract
  in `docs/plans/README.md`.
- **Camera place names** ("Front corridor", "Main gate") come from a new optional config field,
  `cctv.names: {cam1: "Front corridor", cam2: "Main gate"}` in `go2rtc.yaml`, returned by
  `/api/cctv/status`. Missing name → show the camera id only.

### Backend gap 2: instant push (Server-Sent Events)

- `GET /api/cctv/events/stream` → `Content-Type: text/event-stream`, one long-lived connection per
  open dashboard.
- Messages:
  - `event: alert` + the event JSON, for every new event (from POST or from Go's own events such
    as `camera_offline`).
  - `event: update` + the event JSON, when an event changes (snapshot ready, siren state changed,
    acknowledged).
  - a comment line (`: ping`) every 15 s to keep the connection open through proxies.
- The event store gets a small fan-out: subscribers are channels; a slow or closed client is
  dropped instead of blocking `add()`.
- go2rtc's API server sets no write timeout, so long-lived responses are fine; check that our
  handler flushes after each message (`http.Flusher`).

### Backend gap 3: snapshots

- `GET /api/cctv/snapshot?id=<event id>` → `image/jpeg` from `data/snapshots/<id>.jpg`. Only
  numeric ids, so no path tricks. 404 if missing.
- **Where the image comes from:**
  1. If the POST carries an image (new optional field `image`: base64 JPEG, e.g. from the Python
     worker with the detection boxes already drawn), save that.
  2. Otherwise the server grabs the current frame of that camera from go2rtc itself (the same
     keyframe-to-JPEG path as go2rtc's `GET /api/frame.jpeg?src=cam1`) and saves it.
- The event is pushed **immediately**, without waiting for the image; when the file is saved, the
  server sets `snapshot` to `api/cctv/snapshot?id=<id>` and sends `event: update`. The popup shows a
  short loading shimmer in the snapshot area until then.
- POST body limit goes from 1 MB to 4 MB to fit the image.
- Old snapshot files beyond `max_events` are deleted so `data/` doesn't grow forever.

### Acknowledge

- `POST /api/cctv/events/ack {"id": 17}` → sets `acknowledged_at` on the event (new field), stops
  that camera's siren if no other unacknowledged siren alert remains for it, and pushes
  `event: update` so every open dashboard closes that popup.
- Acknowledged state is written to the events file too (appended update line, or rewrite; builder's
  choice) so a restart doesn't bring back old popups.

### Camera siren (two-way audio)

- On a new `intrusion` / `restricted_area` / `fire` event, the Go module asks go2rtc to play a
  looping siren **to that camera's stream** (the same mechanism as go2rtc's
  `POST /api/streams?dst=cam1&src=...`, i.e. `streams.Get(cam).Play(src)` with an ffmpeg file
  source looped and encoded to the camera's audio codec).
- It stops when the event is acknowledged (`Play("")` stops go2rtc's internal producer).
- **Safety cut-off:** the camera siren stops by itself after `siren_max` (default 3 min) even if
  nobody acknowledges; the popup stays until acknowledged.
- *(My call)* **One siren per camera:** the 3-minute timer starts with the first unacknowledged
  siren event on that camera. More siren events while it rings don't restart it. After the cut-off
  that camera stays quiet until its siren alerts are all acknowledged; only after that can a new
  siren event ring it again. A person loitering all night gives at most 3 minutes of siren.
  Siren state per event: `ringing`, `stopped` (acknowledged or cut off), `unavailable`.
- If the camera has no speaker or go2rtc can't reach it (true for the looping test video today),
  `Play` fails: log it once, set the event's siren state to `unavailable`, and the popup says
  "camera speaker not available". Nothing else breaks.
- The siren sound is a small WAV generated once with ffmpeg (two-tone, ~2 s, loopable) and
  committed under `www/` or `internal/cctv/` (it's generated by us, so no licence issue).
- Config: `cctv.siren: {use_cases: [intrusion, restricted_area, fire], max: 3m, file: ...}`.

### Simulate alert (demo before the AI exists)

- The "▶ Simulate alert" button POSTs a realistic event to the **real** `/api/cctv/events`, cycling
  through use cases and both cameras, so the demo exercises the true path: POST → store → push →
  popup → snapshot grab → siren.
- It is visible only when the page is opened with `?demo=1`, so the real dashboard doesn't have a
  fake-alert button.

### Dashboard numbers

- Real values only: cameras online (from status), alerts today, unacknowledged count, last 12 hours
  timeline (from events), per-use-case counts. "Alerts" means events that pop up; the counts
  don't include feed-only events.
- *(My call)* **Use cases per camera:** a static list in config,
  `cctv.use_cases: {cam1: [intrusion, restricted_area, ...], cam2: [...]}`, returned by
  `/api/cctv/status`. The top bar and the tiles label it **"configured"**, not "armed", because Go
  can't yet tell whether the AI worker is actually running. A worker heartbeat that turns this into
  "armed" comes later, together with the pipeline numbers.
- The **detection-pipeline strip** and "average alert time" need numbers the AI worker doesn't send
  yet. Until it does, show them as "waiting for AI worker" (greyed), not invented numbers. The
  worker can later POST its counters to a metrics endpoint (out of scope here; the strip is ready
  for it).

## Left to the builder

- Exact file split (`cctv.html` alone vs. `.html` + `.js` + `.css`), CSS details, icon drawing.
- How the event store notifies subscribers (channels, callback list) and how acknowledged state is
  persisted.
- How the frame is grabbed inside Go (call go2rtc's keyframe code directly vs. an internal HTTP
  call to `/api/frame.jpeg`).
- The exact ffmpeg source string for the looping siren and its codec choice (PCMA/PCMU/AAC depends
  on the camera's backchannel).
- Tests: Go tests for the SSE fan-out, ack, snapshot endpoint (id validation, 404), siren
  start/stop/timeout with a fake player; a manual demo check list for the page.
