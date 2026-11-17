"""ai/tests/test_fall.py: Unit tests for Fall Detection plugin.
Per ZoneFallPlan.md Section 9.3 and docs/plans/p3-p4-sanjay-kishore.md Section 4.
"""
import unittest
from datetime import datetime, timedelta, timezone
import numpy as np

from cctv_ai.objects import TrackedObject
from cctv_ai.testing import make_object
from plugins.fall import Plugin as FallPlugin


class TestFallPlugin(unittest.TestCase):
    def setUp(self):
        self.plugin = FallPlugin()
        self.plugin.setup(["cam1"])
        self.t0 = datetime.now(timezone.utc)
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def test_clear_fall_event_and_deduplication(self):
        """Simulate a person falling:
        - Frames 0-4: standing upright (w=100, h=300, y1=100, y2=400, AR=0.33)
        - Frame 5: fast drop (y1 drops from 100 to 450, box becomes w=350, h=150, AR=2.33)
        - Frames 6-25: stays down for > 3.0 seconds.
        -> Exactly one fall event emitted!
        """
        # 1. Standing history
        history = []
        for i in range(5):
            box = (500, 100, 600, 400)
            history.append(box)
            p = make_object(obj_id="p1", box=box, history=list(history))
            t = self.t0 + timedelta(seconds=0.2 * i)
            evts = self.plugin.on_objects("cam1", self.frame, t, [p], [])
            self.assertEqual(evts, [])

        # 2. Fast drop to lying position (y1 drops by 350px > 40% of 300px standing height)
        fallen_box = (400, 450, 750, 600)  # w=350, h=150 -> AR = 2.33 > 1.2
        history.append(fallen_box)

        fall_events = []
        # Simulate staying down for 3.6 seconds (18 frames at 0.2s)
        t_drop = self.t0 + timedelta(seconds=1.0)
        for i in range(18):
            t = t_drop + timedelta(seconds=0.2 * i)
            p = make_object(obj_id="p1", box=fallen_box, history=list(history))
            evts = self.plugin.on_objects("cam1", self.frame, t, [p], [])
            for e in evts:
                if e.label == "fall":
                    fall_events.append(e)

        # Verified: exactly one fall event emitted!
        self.assertEqual(len(fall_events), 1)
        self.assertEqual(fall_events[0].data["object_id"], "p1")
        self.assertGreaterEqual(fall_events[0].data["seconds_down"], 2.0)

        # 3. Continued frames on ground do NOT spam duplicate events
        t_cont = t_drop + timedelta(seconds=4.0)
        p = make_object(obj_id="p1", box=fallen_box, history=list(history))
        evts = self.plugin.on_objects("cam1", self.frame, t_cont, [p], [])
        self.assertEqual(evts, [])

    def test_recovery_cancels_fall(self):
        """Bending down or brief fall where person gets up within 2 seconds -> NO event."""
        history = [(500, 100, 600, 400)] * 5
        fallen_box = (400, 450, 750, 600)
        upright_box = (500, 100, 600, 400)

        # Down for 1.6 seconds (< 3.0s persistence)
        for i in range(8):
            t = self.t0 + timedelta(seconds=0.2 * i)
            p = make_object(obj_id="p1", box=fallen_box, history=history + [fallen_box])
            evts = self.plugin.on_objects("cam1", self.frame, t, [p], [])
            self.assertEqual(evts, [])

        # Stood back up!
        t_up = self.t0 + timedelta(seconds=2.0)
        p = make_object(obj_id="p1", box=upright_box, history=history + [upright_box])
        evts = self.plugin.on_objects("cam1", self.frame, t_up, [p], [])
        self.assertEqual(evts, [])

        # Continues standing for 3 more seconds
        for i in range(15):
            t = t_up + timedelta(seconds=0.2 * i)
            p = make_object(obj_id="p1", box=upright_box, history=history + [upright_box])
            evts = self.plugin.on_objects("cam1", self.frame, t, [p], [])
            self.assertEqual(evts, [])

    def test_sitting_down_does_not_trigger_fall(self):
        """Sitting down compresses box height, but AR stays upright (w=150, h=180, AR=0.83 < 1.2)."""
        history = [(500, 100, 600, 400)] * 5
        sitting_box = (480, 250, 630, 430)  # w=150, h=180 -> AR = 0.83

        for i in range(20):
            t = self.t0 + timedelta(seconds=0.2 * i)
            p = make_object(obj_id="p1", box=sitting_box, history=history + [sitting_box])
            evts = self.plugin.on_objects("cam1", self.frame, t, [p], [])
            fall_evts = [e for e in evts if e.label == "fall"]
            self.assertEqual(fall_evts, [])

    def test_bed_sofa_exclude_zone_suppresses_alert(self):
        """Person lying horizontally on bed/sofa (exclude zone) -> NO event emitted."""
        from cctv_ai.zones import ZoneConfig, ZoneTracker
        # Explicitly configure an exclude zone for testing without requiring hardcoded zones.yaml entries
        self.plugin.trackers["cam1"] = ZoneTracker("cam1", {
            "test_bed": ZoneConfig(name="test_bed", points=[[0.6, 0.6], [0.95, 0.6], [0.95, 0.95], [0.6, 0.95]], exclude=True)
        })
        self.plugin.trackers["cam1"].set_resolution(1280, 720)

        bed_box = (800, 480, 1150, 600)  # w=350, h=120 -> AR=2.91. Feet at (975, 600) -> inside bed
        history = [(850, 200, 950, 500)] * 5 + [bed_box]

        fall_evts = []
        for i in range(25):  # 5.0 seconds
            t = self.t0 + timedelta(seconds=0.2 * i)
            p = make_object(obj_id="p1", box=bed_box, history=history)
            evts = self.plugin.on_objects("cam1", self.frame, t, [p], [])
            fall_evts.extend([e for e in evts if e.label == "fall"])

        self.assertEqual(fall_evts, [])

    def test_rearming_after_recovery(self):
        """After a fall, if person stands up for >= 3.0s, detector re-arms for next fall."""
        history = [(500, 100, 600, 400)] * 5
        fallen_box = (400, 450, 750, 600)
        upright_box = (500, 100, 600, 400)

        # 1. First fall
        for i in range(20):
            t = self.t0 + timedelta(seconds=0.2 * i)
            p = make_object(obj_id="p1", box=fallen_box, history=history + [fallen_box])
            self.plugin.on_objects("cam1", self.frame, t, [p], [])

        self.assertTrue(self.plugin.detector.states["p1"].event_emitted)

        # 2. Stand up for 3.6s (18 frames) -> re-arms
        t_up = self.t0 + timedelta(seconds=5.0)
        for i in range(18):
            t = t_up + timedelta(seconds=0.2 * i)
            p = make_object(obj_id="p1", box=upright_box, history=[upright_box] * 5)
            self.plugin.on_objects("cam1", self.frame, t, [p], [])

        self.assertFalse(self.plugin.detector.states["p1"].event_emitted)

        # 3. Second fall -> triggers second event!
        t_second_fall = t_up + timedelta(seconds=4.0)
        new_fall_events = []
        for i in range(20):
            t = t_second_fall + timedelta(seconds=0.2 * i)
            p = make_object(obj_id="p1", box=fallen_box, history=[upright_box] * 3 + [fallen_box])
            evts = self.plugin.on_objects("cam1", self.frame, t, [p], [])
            new_fall_events.extend([e for e in evts if e.label == "fall"])

        self.assertEqual(len(new_fall_events), 1)


if __name__ == "__main__":
    unittest.main()
