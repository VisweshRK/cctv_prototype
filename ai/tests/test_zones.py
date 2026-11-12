"""ai/tests/test_zones.py: Unit tests for shared ZoneTracker and feet() geometry.
Per ZoneFallPlan.md Section 9.1 and docs/plans/p3-p4-sanjay-kishore.md.
"""
import unittest
from datetime import datetime, timedelta, timezone

from cctv_ai.objects import TrackedObject
from cctv_ai.testing import make_object
from cctv_ai.zones import ZoneConfig, ZoneTracker, feet, point_in_polygon


class TestZoneHelper(unittest.TestCase):
    def setUp(self):
        self.t0 = datetime.now(timezone.utc)
        # Define a zone covering [0.2, 0.2] to [0.8, 0.8]
        self.zone_cfg = {
            "room": ZoneConfig(
                name="room",
                points=[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
                objects=["person"],
                inertia=3,
                loitering_time=5.0,
            ),
            "bed": ZoneConfig(
                name="bed",
                points=[[0.6, 0.6], [0.8, 0.6], [0.8, 0.8], [0.6, 0.8]],
                objects=["person"],
                inertia=3,
                exclude=True,
            ),
        }
        self.tracker = ZoneTracker(camera="cam1", zones=self.zone_cfg)
        self.tracker.set_resolution(width=1000, height=1000)

    def test_feet_calculation(self):
        """Feet is bottom-centre ((x1 + x2) / 2, y2)."""
        box = (100, 200, 300, 600)  # x1=100, x2=300, y2=600
        self.assertEqual(feet(box), (200.0, 600.0))

    def test_center_inside_but_feet_outside(self):
        """A person standing in front of a doorway has box center inside, but feet outside."""
        # Zone covers y from 200 to 800.
        # Person box from y1=100 to y2=190. Box centre is y=145 (outside),
        # but let's test box overlapping the zone:
        # Box from y1=100 to y2=250. Box centre is y=175 (outside).
        # Box from y1=150 to y2=195. Feet is at 195 (outside < 200).
        # Box from y1=180 to y2=199: feet at 199 is outside.
        # Box from y1=190 to y2=210: feet at 210 is inside (> 200).
        box_outside = (400, 100, 600, 199)
        self.assertFalse(self.tracker.is_in_any_exclude_zone(box_outside))
        p = make_object(box=box_outside)
        tr = self.tracker.update([p], self.t0)
        self.assertEqual(tr, [])

    def test_inertia_entry_and_exit(self):
        """Object must be observed inside for inertia=3 consecutive frames to enter,
        and outside for 3 consecutive frames to exit."""
        # Inside box: x=500, feet y=500 (inside room)
        inside_box = (450, 300, 550, 500)
        # Outside box: x=100, feet y=100 (outside room)
        outside_box = (50, 50, 150, 100)

        p_in = make_object(obj_id="p1", box=inside_box)
        p_out = make_object(obj_id="p1", box=outside_box)

        # Frame 1: inside count = 1 -> no event
        t1 = self.t0
        self.assertEqual(self.tracker.update([p_in], t1), [])
        self.assertEqual(self.tracker.count("room"), 0)

        # Frame 2: inside count = 2 -> no event
        t2 = t1 + timedelta(seconds=0.2)
        self.assertEqual(self.tracker.update([p_in], t2), [])
        self.assertEqual(self.tracker.count("room"), 0)

        # Frame 3: inside count = 3 -> ENTERED!
        t3 = t2 + timedelta(seconds=0.2)
        trans = self.tracker.update([p_in], t3)
        self.assertEqual(len(trans), 1)
        self.assertEqual(trans[0].change, "entered")
        self.assertEqual(trans[0].zone, "room")
        self.assertEqual(self.tracker.count("room"), 1)

        # Frame 4: stays inside -> no duplicate entered event
        t4 = t3 + timedelta(seconds=0.2)
        self.assertEqual(self.tracker.update([p_in], t4), [])
        self.assertEqual(self.tracker.count("room"), 1)

        # Frame 5: steps outside (1 outside observation) -> still inside
        t5 = t4 + timedelta(seconds=0.2)
        self.assertEqual(self.tracker.update([p_out], t5), [])
        self.assertEqual(self.tracker.count("room"), 1)

        # Frame 6: still outside (2 outside observations) -> still inside
        t6 = t5 + timedelta(seconds=0.2)
        self.assertEqual(self.tracker.update([p_out], t6), [])
        self.assertEqual(self.tracker.count("room"), 1)

        # Frame 7: outside for 3rd observation -> LEFT!
        t7 = t6 + timedelta(seconds=0.2)
        trans_left = self.tracker.update([p_out], t7)
        self.assertEqual(len(trans_left), 1)
        self.assertEqual(trans_left[0].change, "left")
        self.assertEqual(self.tracker.count("room"), 0)

    def test_intermittent_jitter_resets_pending_counter(self):
        """If an object is inside for 2 frames then outside for 1, the inside counter resets."""
        inside_box = (450, 300, 550, 500)
        outside_box = (50, 50, 150, 100)
        p_in = make_object(obj_id="p1", box=inside_box)
        p_out = make_object(obj_id="p1", box=outside_box)

        # 2 frames inside
        self.tracker.update([p_in], self.t0)
        self.tracker.update([p_in], self.t0 + timedelta(seconds=0.2))
        self.assertEqual(self.tracker.count("room"), 0)

        # 1 jitter frame outside -> resets inside_count back to 0
        self.tracker.update([p_out], self.t0 + timedelta(seconds=0.4))
        self.assertEqual(self.tracker.count("room"), 0)

        # 1 frame inside -> inside_count is 1 (not 3!) -> no event
        trans = self.tracker.update([p_in], self.t0 + timedelta(seconds=0.6))
        self.assertEqual(trans, [])
        self.assertEqual(self.tracker.count("room"), 0)

    def test_loitering_event(self):
        """Loitering triggers once when stay duration reaches loitering_time."""
        inside_box = (450, 300, 550, 500)
        p_in = make_object(obj_id="p1", box=inside_box)

        # Enter
        for i in range(3):
            t = self.t0 + timedelta(seconds=0.2 * i)
            self.tracker.update([p_in], t)
        self.assertTrue(self.tracker.is_in_zone("p1", "room"))

        # Advance time by 4 seconds (total 4.6s < 5.0s loitering threshold)
        t_4s = self.t0 + timedelta(seconds=4.6)
        self.assertEqual(self.tracker.update([p_in], t_4s), [])

        # Advance past 5.0 seconds
        t_6s = self.t0 + timedelta(seconds=6.0)
        trans = self.tracker.update([p_in], t_6s)
        self.assertEqual(len(trans), 1)
        self.assertEqual(trans[0].change, "loitering")
        self.assertGreaterEqual(trans[0].seconds_inside, 5.0)

        # Further frames do not repeat the loitering event
        t_7s = self.t0 + timedelta(seconds=7.0)
        self.assertEqual(self.tracker.update([p_in], t_7s), [])

    def test_exclude_zone_detection(self):
        """Bed/sofa exclusion matches when feet are inside the exclude zone."""
        # Inside bed: x in [600, 800], y in [600, 800]
        on_bed_box = (650, 650, 750, 750)
        off_bed_box = (300, 300, 400, 500)
        self.assertTrue(self.tracker.is_in_any_exclude_zone(on_bed_box))
        self.assertFalse(self.tracker.is_in_any_exclude_zone(off_bed_box))

    def test_resolution_change(self):
        """Resolution change recomputes pixel polygons correctly."""
        self.tracker.set_resolution(width=2000, height=2000)
        # Point at (1000, 1000) is normalized (0.5, 0.5) -> inside room
        box_scaled = (950, 900, 1050, 1000)
        p = make_object(box=box_scaled)
        for i in range(3):
            trans = self.tracker.update([p], self.t0 + timedelta(seconds=0.2 * i))
        self.assertEqual(trans[0].change, "entered")


if __name__ == "__main__":
    unittest.main()
