import unittest
from datetime import datetime, timedelta, timezone
import cv2
import numpy as np

from cctv_ai.objects import TrackedObject
from plugins.baby_crying import Plugin as Baby
from plugins.power_when_absent import Plugin as Power

class PortedPluginsTest(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now(timezone.utc)

    def test_baby_crying_harmonic_fallback_starts_and_ends_episode(self):
        plugin = Baby(); plugin.setup(["cam1"])
        plugin.session = None  # exercise the fallback even when local model assets are installed
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

    def test_yamnet_scores_baby_cry_not_whimper(self):
        plugin = Baby()

        class Session:
            def run(self, _outputs, _inputs):
                scores = np.zeros((1, 521), dtype=np.float32)
                scores[0, 20] = 0.37  # Baby cry, infant cry
                scores[0, 21] = 0.95  # Whimper
                return [scores]

        plugin.session = Session()
        plugin.input_name = "waveform"
        score, _ = plugin._score(np.ones(16000, dtype=np.int16), 16000)
        self.assertAlmostEqual(score, 0.37, places=5)

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
