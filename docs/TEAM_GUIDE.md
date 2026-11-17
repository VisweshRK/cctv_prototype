# Team guide: start here

Everyone on the team reads this once. It covers what we're building, how to set up your laptop, and
the rules that keep us from breaking each other's work.

## 1. What we're building

A smart CCTV system for two Q-Tech Wi-Fi cameras that detects 10 kinds of incidents.

```
cameras ──> go2rtc (Go) ──frames/audio──> AI worker (Python) ──events──> go2rtc API ──> live view + event list
```

- **go2rtc** (Go, already done) connects to the cameras once and gives live view at http://localhost:1984.
  It also detects a camera going offline.
- **The AI worker** (Python, `ai/`) captures the latest RTSP frame per camera, shares it with use-case
  plugins, and drops stale frames instead of allowing latency to grow. **Your job is one plugin.**
- A plugin looks at a frame (or 1 s of sound) and returns events like
  `{"camera": "cam1", "use_case": "fire", "label": "smoke", "confidence": 0.8}`.

You don't need to touch Go. You don't need a camera to start (see step 3).

## 2. One-time setup (Windows)

1. Accept the GitHub invite to **VisweshRK/cctv_prototype** (check your email). The repo is private.
2. Install the tools, then **close and reopen your terminal**:
   ```bash
   winget install Git.Git
   ```
   ```bash
   winget install GoLang.Go
   ```
   ```bash
   winget install Python.Python.3.12
   ```
   ```bash
   winget install Gyan.FFmpeg
   ```
3. Get the code:
   ```bash
   git clone https://github.com/VisweshRK/cctv_prototype.git
   ```
   ```bash
   cd cctv_prototype
   ```
4. Set the name and email for this repo (use the email on your GitHub account):
   ```bash
   git config user.name "Your Name"
   ```
   ```bash
   git config user.email "you@example.com"
   ```
5. Python environment:
   ```bash
   cd ai
   py -3.12 -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   cd ..
   ```
6. Build go2rtc (the first build takes a few minutes):
   ```bash
   go build -o go2rtc.exe .
   ```

macOS/Linux: same steps with your package manager. Use `.venv/bin/python` and build `go2rtc` without `.exe`.

## 3. Run it (with or without a camera)

1. Create your `.env` from the example:
   ```bash
   copy .env.example .env
   ```
2. Pick a source for your fake or real cameras:
   - **No camera yet (most of you):** make a test clip as shown in [samples/README.md](../samples/README.md),
     or better, record a short clip of your use case. Then point the cameras at it in `.env`:
     ```
     CAM1_URL=ffmpeg:samples/test.mp4#input=loop#video=copy#audio=copy
     CAM2_URL=ffmpeg:samples/test.mp4#input=loop#video=copy#audio=copy
     ```
   - **Real cameras:** ask Viswesh for the camera addresses. Never post them in the group chat or in git.
3. Start go2rtc and open http://localhost:1984. You should see `cam1` and `cam2`.
   ```bash
   .\go2rtc.exe
   ```
4. In a second terminal, run the worker with the example plugin:
   ```bash
   cd ai
   .venv\Scripts\python -m cctv_ai.worker -v
   ```
   You should see `event #N cam1/example_motion motion ...`. Events are listed at
   http://localhost:1984/api/cctv/events and camera status at http://localhost:1984/api/cctv/status.

If that works, your setup is done.

## 4. Working on your use case

| # | Use case | Folder | Owner | Depends on |
| --- | --- | --- | --- | --- |
| 1 | Intrusion | `ai/plugins/intrusion` | P3 Sanjay + P4 Kishore | shared detector (person) |
| 2 | Tampering (covered/blurred/moved) | `ai/plugins/tampering` | Viswesh | nothing |
| 3 | Face recognition | `ai/plugins/face_recognition` | P2 Krithick | nothing (own face model) |
| 4 | Restricted area | `ai/plugins/restricted_area` | P3 Sanjay + P4 Kishore | shared detector (person) |
| 5 | Fall detection | `ai/plugins/fall` | P3 Sanjay + P4 Kishore | shared detector (person) |
| 6 | Occupancy | `ai/plugins/occupancy` | P3 Sanjay + P4 Kishore | shared detector (person) |
| 7 | Electricity used when nobody is there | `ai/plugins/power_when_absent` | P1 Mohanraj | occupancy (start with "lights on?" from frame brightness) |
| 8 | Fire and smoke | `ai/plugins/fire` | P1 Mohanraj | nothing (own model) |
| 9 | Vehicle + licence plate | `ai/plugins/vehicle_lpr` | P2 Krithick | shared detector (car), then own plate model + OCR |
| 10 | Baby crying | `ai/plugins/baby_crying` | P1 Mohanraj | camera audio |

Viswesh owns the shared code (worker, detector, go2rtc config, cameras) and helps P2 with models.
The integration branch keeps all ten slots in main's structure. Fire, baby crying, and
power-when-absent were ported from the unrelated platform branch to the `Plugin`/`Event` contract;
do not import that branch's FastAPI, Vite, database, or `StandardAIEvent` architecture.

**Shared detector: don't build your own person/car detector.** One detector runs once per frame for
everyone (so the laptop runs one model, not five) and gives each plugin a list of boxes:

```python
[{"label": "person", "box": [x1, y1, x2, y2], "score": 0.9}, ...]   # pixels, top-left / bottom-right
```

Until it lands in `ai/cctv_ai/`, write your logic (zones, counting, fall rules, plate reading) against that
list, and test it with hand-made boxes.

1. Read your folder's `README.md` (goal, events to send, which Frigate code to read, suggested approach)
   and the working example `ai/plugins/example_motion/__init__.py`.
2. Start a branch named after your folder:
   ```bash
   git switch main
   git pull
   git switch -c usecase/intrusion
   ```
3. Write your code in **your folder only**. Run only your plugin:
   ```bash
   cd ai
   .venv\Scripts\python -m cctv_ai.worker --plugins intrusion -v
   ```
4. Save your work to GitHub often (at least at the end of every day you work on it):
   ```bash
   git add ai/plugins/intrusion ai/tests/test_intrusion.py
   git commit -m "intrusion: run person detector on motion frames"
   git push -u origin usecase/intrusion
   ```
5. When it works, open a **Pull Request** on GitHub from your branch into `main`. CI must be green and one teammate reviews before merging.
6. Keep your branch up to date when `main` changes:
   ```bash
   git switch main
   git pull
   git switch usecase/intrusion
   git merge main
   ```

## 5. The rules

1. **Never commit secrets.** Camera addresses and passwords live only in `.env`, which git ignores. Run
   `git status` before every commit and never use `git add -f`.
2. **Never push to `main`.** Always use your `usecase/<folder>` branch and a pull request.
3. **Stay in your folder.** Changes to shared code (`ai/cctv_ai/`, `internal/`, `go2rtc.yaml`,
   `requirements.txt`) go in their own small PR, and you tell the team about it in the group first.
4. **Use the event format exactly:** `use_case` = your folder name; `label` from your README;
   `confidence` between 0 and 1; extra details in `data`. Use `self.event(...)` to build events.
5. **One event per incident, not per frame.** Use a cooldown, like `example_motion` does.
6. **Load models once, in `setup()`.** `on_frame()`/`on_objects()` must be fast; use `heavy = True`
   and a bounded model workload rather than blocking capture or event delivery.
7. **No big files in git:** no model files (`.onnx`, `.pt`, `.tflite` are ignored), no videos. Add a
   download script or a link in your README instead.
8. **No people's faces or footage in git.** Keep face photos and recordings on your laptop
   (`samples/` is ignored).
9. **Add new Python packages to `ai/requirements.txt`** with a minimum version, e.g. `onnxruntime>=1.18`.
10. **Tests must pass** before you open a PR:
    ```bash
    cd ai
    .venv\Scripts\python -m unittest discover -s tests
    ```
11. **Credit Frigate when you copy its code.** Frigate is MIT: copying ideas is fine; if you copy actual
    code, put its copyright line (`Copyright (c) 2026 Frigate, Inc.`, MIT) in a comment above it.
12. **Check a model's licence before you use it.** Our project is MIT, so models and libraries must be
    MIT, Apache 2.0 or BSD (for example YOLOX, RT-DETR, OpenCV's YuNet/SFace, PaddleOCR).
    **Don't use Ultralytics YOLOv5/v8/v11 (AGPL) or YOLOv9 (GPL)**: they would force the whole project
    under their licence. Write the model's name, link and licence in your folder README.

## 6. When is a use case done?

- [ ] It sends the right events when run on a clip that shows the incident.
- [ ] It sends no events (or very few) on a clip where nothing happens.
- [ ] One incident gives one event, not a flood.
- [ ] It keeps up with 2 cameras on a normal laptop CPU.
- [ ] `enabled = True`, and it has a test in `ai/tests/test_<folder>.py`.
- [ ] Your folder README says how to get the model and what the settings mean.

## 7. Full-system commands

From the repository root, configure `.env.example`, build go2rtc, and start the stream service:

```bash
go build -o go2rtc .
./go2rtc
```

In a second terminal, start the worker with all enabled plugins:

```bash
cd ai
.venv/bin/python -m cctv_ai.worker -v --cameras cam1,cam2
```

Open `http://localhost:1984/` for the dashboard. Use `python -m cctv_ai.worker --list` to inspect
which slots are enabled. The face/vehicle models and optional YAMNet weights are downloaded by their
slot scripts into ignored model folders; never commit weights, camera URLs, or credentials.

The performance contract is: one latest-frame buffer per camera, no synchronous event POST in the
inference path, bounded queues, and no recreation of existing `<video-stream>` elements during alert
or status updates.

## 7. Help

- Something broken in setup, the worker or go2rtc: ask in the group with the error text (a screenshot
  of the terminal is fine; hide camera addresses).
- Your step-by-step plan: [plans/](plans/README.md). How Frigate's engine works: [how-frigate-works.md](how-frigate-works.md).
- Background on how Frigate does each use case: [frigate-teardown.md](frigate-teardown.md).
- More detail: [README.md](../README.md), [ai/README.md](../ai/README.md), [CONTRIBUTING.md](../CONTRIBUTING.md).
