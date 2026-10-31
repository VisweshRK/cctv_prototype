"""Run from ai/:  python -m unittest tests.test_tampering -v

The camera view is a synthetic scene (seeded random shapes), so no footage is needed:
- covered:  a black frame, or one flat colour
- blurred:  a heavy Gaussian blur of the normal view
- moved:    a crop of the same scene shifted sideways, or a different scene
- day/night: the same view turned grey-scale (the camera switching to IR)
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np

from cctv_ai.loader import discover

H, W = 360, 640


def make_world(seed: int) -> np.ndarray:
    """A colourful scene bigger than the view, so views can be cropped from it."""
    rng = np.random.default_rng(seed)
    world = np.full((H + 240, W + 480, 3), (60, 110, 160), np.uint8)
    for _ in range(450):
        colour = tuple(int(c) for c in rng.integers(0, 256, 3))
        x, y = (int(v) for v in rng.integers(0, (W + 480, H + 240)))
        size = int(rng.integers(6, 40))
        if rng.random() < 0.5:
            cv2.rectangle(world, (x, y), (x + size, y + int(rng.integers(6, 40))), colour, -1)
        else:
            cv2.circle(world, (x, y), size // 2, colour, -1)
    return world


WORLD = make_world(1)
OTHER_WORLD = make_world(2)
NOISE = np.random.default_rng(7)


def view(dx: int = 0, dy: int = 0, world: np.ndarray = WORLD) -> np.ndarray:
    """The camera's normal view, with a little sensor noise like a real camera."""
    crop = world[120 + dy:120 + dy + H, 240 + dx:240 + dx + W].astype(np.int16)
    crop += NOISE.integers(-4, 5, crop.shape, dtype=np.int16)
    return np.clip(crop, 0, 255).astype(np.uint8)


def black() -> np.ndarray:
    return np.full((H, W, 3), 4, np.uint8)


def flat(colour=(150, 150, 150)) -> np.ndarray:
    return np.full((H, W, 3), colour, np.uint8)


def blurred() -> np.ndarray:
    return cv2.GaussianBlur(view(), (0, 0), 6)


def clock(frame: np.ndarray) -> np.ndarray:
    """The date and time a real camera burns onto every frame, even when the lens is covered."""
    frame = frame.copy()
    for colour, thickness in (((0, 0, 0), 5), ((255, 255, 255), 2)):
        cv2.putText(frame, "2026-10-31 12:00:00", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, thickness)
        cv2.putText(frame, "CAM1", (W - 90, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, thickness)
    return frame


def grey(frame: np.ndarray, gain: float = 0.8) -> np.ndarray:
    """IR mode: no colour, a different brightness."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) * gain
    return cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


class TamperingTest(unittest.TestCase):
    def setUp(self):
        self.plugin = discover()["tampering"]()
        self.plugin.learn_checks = 5  # 60 in production; short so the tests are quick
        self.plugin.relearn_checks = 5
        self.plugin.setup(["cam1", "cam2"])
        self.ts = datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)

    def feed(self, frames, camera: str = "cam1"):
        """One check per frame, one second apart. Returns all events."""
        events = []
        for frame in frames:
            events += self.plugin.on_frame(camera, frame, self.ts)
            self.ts += timedelta(seconds=1)
        return events

    def learn(self, camera: str = "cam1", make=view):
        self.assertEqual(self.feed([make() for _ in range(self.plugin.learn_checks)], camera), [])

    def labels(self, events):
        return [e.label for e in events]

    def test_contract(self):
        cls = type(self.plugin)
        self.assertTrue(cls.enabled)
        self.assertEqual(cls.use_case, "tampering")
        self.assertEqual(cls.needs, {"video"})

    def test_normal_view_gives_nothing(self):
        self.learn()
        self.assertEqual(self.feed([view() for _ in range(30)]), [])
        # small shake and a darker room are not tampering
        self.assertEqual(self.feed([view(dx=6, dy=-4), view(dx=-5, dy=3)]), [])
        darker = [(view().astype(np.float32) * 0.6).astype(np.uint8) for _ in range(6)]
        self.assertEqual(self.feed(darker), [])

    def test_no_alert_while_learning(self):
        self.assertEqual(self.feed([black() for _ in range(self.plugin.learn_checks)]), [])

    def test_covered_black(self):
        self.learn()
        self.assertEqual(self.feed([black(), black()]), [])  # 2 checks: not yet
        events = self.feed([black()])
        self.assertEqual(self.labels(events), ["covered"])
        e = events[0]
        self.assertEqual((e.camera, e.use_case), ("cam1", "tampering"))
        self.assertTrue(0.5 <= e.confidence <= 1)
        self.assertEqual(e.data["area"], "Whole view")
        self.assertIn("detail", e.data)
        # one event per incident, not one per frame
        self.assertEqual(self.feed([black() for _ in range(20)]), [])

    def test_covered_flat_colour(self):
        self.learn()
        self.assertEqual(self.labels(self.feed([flat() for _ in range(3)])), ["covered"])

    def test_restored_once(self):
        self.learn()
        self.feed([black() for _ in range(5)])
        self.assertEqual(self.feed([view(), view()]), [])  # must stay normal for 3 checks
        events = self.feed([view()])
        self.assertEqual(self.labels(events), ["restored"])
        self.assertEqual(events[0].data["was"], "covered")
        self.assertEqual(self.feed([view() for _ in range(10)]), [])
        # a new incident after that gives a new event
        self.assertEqual(self.labels(self.feed([black() for _ in range(3)])), ["covered"])

    def test_short_glitch_is_ignored(self):
        """Headlights, a bird, a passing hand: fewer than 3 checks in a row."""
        self.learn()
        frames = [black(), black(), view(), flat(), view(), blurred(), blurred(), view()]
        self.assertEqual(self.feed(frames), [])

    def test_blurred(self):
        self.learn()
        events = self.feed([blurred() for _ in range(3)])
        self.assertEqual(self.labels(events), ["blurred"])
        self.assertIn("sharpness", events[0].data)

    def test_clock_on_the_picture(self):
        """The camera's sharp, bright clock text must not hide a covered or smeared lens."""
        self.learn(make=lambda: clock(view()))
        self.assertEqual(self.feed([clock(view()) for _ in range(10)]), [])
        self.assertEqual(self.labels(self.feed([clock(black()) for _ in range(3)])), ["covered"])
        self.feed([clock(view()) for _ in range(3)])
        self.assertEqual(self.labels(self.feed([clock(flat()) for _ in range(3)])), ["covered"])
        self.feed([clock(view()) for _ in range(3)])
        self.assertEqual(self.labels(self.feed([clock(blurred()) for _ in range(3)])), ["blurred"])

    def test_moved_sideways(self):
        self.learn()
        events = self.feed([view(dx=160) for _ in range(3)])
        self.assertEqual(self.labels(events), ["moved"])

    def test_rotated(self):
        """A rotation keeps every feature matching but leaves the centre where it was."""
        self.learn()
        events = self.feed([cv2.rotate(view(), cv2.ROTATE_180) for _ in range(3)])
        self.assertEqual(self.labels(events), ["moved"])

    def test_turned_away(self):
        self.learn()
        events = self.feed([view(world=OTHER_WORLD) for _ in range(3)])
        self.assertEqual(self.labels(events), ["moved"])
        self.assertEqual(self.labels(self.feed([view() for _ in range(3)])), ["restored"])

    def test_day_night_switch_is_not_tampering(self):
        self.learn()
        night = [grey(view()) for _ in range(30)]
        self.assertEqual(self.feed(night), [])
        # after re-learning at night, tampering is still caught
        self.assertEqual(self.labels(self.feed([black() for _ in range(3)])), ["covered"])
        self.feed([grey(view()) for _ in range(3)])
        # and back to day
        self.assertEqual(self.feed([view() for _ in range(30)]), [])

    def test_covering_at_dusk_is_not_a_day_night_switch(self):
        """A black frame has no colour either, but it must not trigger a re-learn."""
        self.learn()
        self.assertEqual(self.labels(self.feed([black() for _ in range(3)])), ["covered"])

    def test_dark_camera_does_not_alarm(self):
        """A camera whose normal view is dark or plain never reports covered."""
        self.learn(make=black)
        self.assertEqual(self.feed([black() for _ in range(10)]), [])
        self.learn(camera="cam2", make=flat)
        self.assertEqual(self.feed([flat() for _ in range(10)], "cam2"), [])

    def test_cameras_are_independent(self):
        self.learn("cam1")
        self.learn("cam2")
        self.assertEqual(self.feed([view() for _ in range(3)], "cam2"), [])
        events = self.feed([black() for _ in range(3)], "cam1")
        self.assertEqual([(e.camera, e.label) for e in events], [("cam1", "covered")])
        self.assertEqual(self.feed([view() for _ in range(3)], "cam2"), [])

    def test_any_frame_size(self):
        self.learn()
        big = cv2.resize(view(), (1920, 1080))
        self.assertEqual(self.feed([big, big]), [])
        self.assertEqual(self.feed([cv2.cvtColor(view(), cv2.COLOR_BGR2GRAY)]), [])  # one channel


if __name__ == "__main__":
    unittest.main()
