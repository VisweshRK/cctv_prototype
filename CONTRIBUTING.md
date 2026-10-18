# Contributing

Setup and the full team rules are in [docs/TEAM_GUIDE.md](docs/TEAM_GUIDE.md). This page is the short version.

## Taking a use case

1. Pick an open slot from the table in [README.md](README.md) and tell the team.
2. Create a branch for it (the slot's folder name):
   ```bash
   git switch -c usecase/intrusion
   ```
3. Work only inside your slot folder `ai/plugins/<slot>/` and your tests `ai/tests/test_<slot>.py`.
   If you need a change in shared code (`ai/cctv_ai/`, `internal/cctv/`, `go2rtc.yaml`), make it a
   separate, small PR so others can review it.
4. Open a pull request into `main`. Add your name to the Owner column in README.md in that PR.

`main` is protected: changes go in through pull requests with passing CI.

## What a plugin must do

Read [ai/README.md](ai/README.md) and the working example [ai/plugins/example_motion](ai/plugins/example_motion).
In short:

- Your class is called `Plugin` and subclasses `cctv_ai.plugin.Plugin`.
- `use_case` equals your folder name.
- Load models once in `setup()`, never per frame.
- `on_frame()` / `on_audio()` return a list of `Event`s and should be quick; send one event per
  incident (use a cooldown), not one per frame.
- Set `enabled = True` only when it works.

## Rules

- **No secrets in git.** Camera passwords live in `.env` only. Check `git status` before committing.
- **No model files in git.** `*.onnx`, `*.pt` and `*.tflite` are ignored. Add a download script or a
  link in your slot README instead.
- **No faces or footage in git** except small test samples you have permission to use.
- New Python packages go in `ai/requirements.txt` with a minimum version.
- Frigate is MIT-licensed. If you copy a piece of its code (not only the idea), keep its copyright
  line in a comment above it and mention it in your PR.

## Before you open a PR

```bash
cd ai
.venv\Scripts\python -m unittest discover -s tests
```

If you changed Go code:

```bash
gofmt -l internal/cctv
go vet ./internal/cctv/
go test ./internal/cctv/
```

## Keeping up with go2rtc

The `upstream` remote points to https://github.com/AlexxIT/go2rtc. Only the maintainer merges upstream
updates, on a separate branch, because they can conflict with `main.go`.
