import unittest
from datetime import datetime, timedelta, timezone
import cv2
import numpy as np

from cctv_ai.objects import TrackedObject
from plugins.fire import Plugin as Fire
from plugins.baby_crying import Plugin as Baby
from plugins.power_when_absent import Plugin as Power

class PortedPluginsTest(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now(timezone.utc)

    def test_fire_persists_and_emits_frontend_fields(self):
        plugin = Fire(); plugin.setup(["cam1"])
        frame = np.zeros((240, 320, 3), np.uint8)
        frame[80:150, 120:190] = (20, 180, 255)
        events = []
        for i in range(3):
            events += plugin.on_frame("cam1", frame, self.now + timedelta(seconds=i))
        self.assertTrue(events)
        self.assertEqual(events[-1].use_case, "fire")
        self.assertIn("box", events[-1].data)
        self.assertIn("detail", events[-1].data)

    def test_baby_crying_harmonic_fallback_starts_and_ends_episode(self):
        plugin = Baby(); plugin.setup(["cam1"])
        t = np.arange(16000) / 16000.0
        samples = (0.8 * np.sin(2 * np.pi * 450 * t)).astype(np.float32)
        first = plugin.on_audio("cam1", samples, 16000, self.now)
        self.assertTrue(first)
        self.assertEqual(first[0].label, "crying")
        silence = np.zeros(16000, np.float32)
        ended = []
        for seconds in range(1, 10):
            ended += plugin.on_audio("cam1", silence, 16000, self.now + timedelta(seconds=seconds))
        self.assertTrue(any(e.label == "crying_ended" for e in ended))

    def test_power_requires_empty_room_and_rearms(self):
        plugin = Power(); plugin.empty_minutes = 0.01; plugin.hysteresis = 0
        plugin.setup(["cam1"])
        plugin.configure_rois("cam1", [{"id": "lamp", "type": "light", "box": [10, 10, 60, 60], "threshold": 100}])
        frame = np.zeros((100, 100, 3), np.uint8); frame[10:60, 10:60] = 220
        person = TrackedObject("p", "person", (10, 10, 40, 90), .9, .9, True, first_seen=self.now, last_seen=self.now)
        plugin.on_objects("cam1", frame, self.now, [person], [])
        self.assertFalse(plugin.on_frame("cam1", frame, self.now + timedelta(seconds=2)))
        plugin.on_objects("cam1", frame, self.now + timedelta(seconds=1), [], [person])
        events = plugin.on_frame("cam1", frame, self.now + timedelta(seconds=3))
        self.assertTrue(events)
        self.assertEqual(events[0].label, "power_wasted")
        self.assertIn("kwh", events[0].data)

if __name__ == "__main__":
    unittest.main()
