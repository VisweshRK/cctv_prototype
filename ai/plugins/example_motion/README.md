# Example: motion (working)

The reference plugin. It is enabled and really works: it compares each frame with the previous one
and sends a `motion` event when enough of the image changed (at most one event per camera every 10 s).

Copy this folder when you start a slot, or read it next to `cctv_ai/plugin.py` to see the contract:

- `setup(cameras)`: called once; load models and create per-camera state here.
- `on_frame(camera, frame, ts)`: `frame` is a BGR `numpy` image. Return a list of `Event`s (or `[]`).
- `on_audio(camera, samples, sample_rate, ts)`: for sound plugins (`needs = {"audio"}`).
- `self.event(camera, label, confidence, data={...})` builds an event with your `use_case` filled in.

Run it:
```bash
cd ai
python -m cctv_ai.worker --plugins example_motion -v
```
