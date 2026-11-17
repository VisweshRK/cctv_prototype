"""Run from ai/:  python -m unittest discover -s tests -v"""

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from cctv_ai.loader import discover
from cctv_ai.plugin import Event, Plugin
from cctv_ai.worker import run_plugin

USE_CASES = {
    "intrusion", "tampering", "face_recognition", "restricted_area", "fall",
    "occupancy", "power_when_absent", "fire", "vehicle_lpr", "baby_crying",
}


class TestSlots(unittest.TestCase):
    def test_all_slots_present(self):
        found = discover()
        self.assertTrue(USE_CASES <= set(found), USE_CASES - set(found))
        self.assertIn("example_motion", found)

    def test_p2_slots_are_enabled(self):
        """P2's two slots ship enabled, so the worker runs them by default."""
        found = discover()
        self.assertTrue(found["face_recognition"].enabled)
        self.assertTrue(found["vehicle_lpr"].enabled)

    def test_slots_follow_contract(self):
        frame = np.zeros((360, 640, 3), np.uint8)
        audio = np.zeros(16000, np.int16)
        now = datetime.now(timezone.utc)
        for name, cls in discover().items():
            with self.subTest(plugin=name):
                self.assertEqual(cls.use_case, name, "use_case must match the folder name")
                self.assertTrue(cls.needs and cls.needs <= {"video", "audio"})
                p = cls()
                p.setup(["cam1"])
                for e in p.on_frame("cam1", frame, now) + p.on_audio("cam1", audio, 16000, now):
                    self.assertIsInstance(e, Event)
                    self.assertEqual(e.use_case, name)
                    self.assertTrue(0 <= e.confidence <= 1)


class TestExampleMotion(unittest.TestCase):
    def setUp(self):
        self.plugin = discover()["example_motion"]()
        self.plugin.setup(["cam1"])
        self.t0 = datetime.now(timezone.utc)

    def test_detects_change_with_cooldown(self):
        dark = np.zeros((360, 640, 3), np.uint8)
        bright = dark.copy()
        bright[100:300, 200:500] = 255

        self.assertEqual(self.plugin.on_frame("cam1", dark, self.t0), [])  # first frame: nothing to compare
        self.assertEqual(self.plugin.on_frame("cam1", dark, self.t0 + timedelta(seconds=1)), [])  # no change

        events = self.plugin.on_frame("cam1", bright, self.t0 + timedelta(seconds=2))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].label, "motion")
        self.assertEqual(events[0].camera, "cam1")

        # change again within cooldown: no second event
        self.assertEqual(self.plugin.on_frame("cam1", dark, self.t0 + timedelta(seconds=3)), [])
        # after cooldown: event again
        self.assertEqual(len(self.plugin.on_frame("cam1", bright, self.t0 + timedelta(seconds=20))), 1)


class TestEventJSON(unittest.TestCase):
    def test_matches_go_format(self):
        e = Event(camera="cam1", use_case="fire", label="smoke", confidence=0.7)
        d = e.to_json()
        self.assertEqual(set(d), {"camera", "use_case", "label", "confidence", "time"})
        datetime.fromisoformat(d["time"])
        d = Event(camera="c", use_case="u", label="l", confidence=1, snapshot="a.jpg", data={"n": 1}).to_json()
        self.assertEqual(d["snapshot"], "a.jpg")
        self.assertEqual(d["data"], {"n": 1})


class TestWorkerSafety(unittest.TestCase):
    def test_broken_plugin_does_not_crash(self):
        class Broken(Plugin):
            use_case = "broken"

            def on_frame(self, camera, frame, ts):
                raise ValueError("boom")

        with self.assertLogs("cctv_ai", level="ERROR"):
            self.assertEqual(run_plugin(Broken(), "on_frame", "cam1", None, None), [])


if __name__ == "__main__":
    unittest.main()
