"""ai/tests/test_occupancy.py: Unit tests for Occupancy Counting plugin.
Per ZoneFallPlan.md Section 9.2 and docs/plans/p3-p4-sanjay-kishore.md Section 3.
"""
import unittest
from datetime import datetime, timedelta, timezone
import numpy as np

from cctv_ai.objects import TrackedObject
from cctv_ai.testing import make_object
from plugins.occupancy import Plugin as OccupancyPlugin


class TestOccupancyPlugin(unittest.TestCase):
    def setUp(self):
        self.plugin = OccupancyPlugin()
        self.plugin.setup(["cam1"])
        self.plugin.debouncer.capacity_limit = 2
        self.plugin.debouncer.summary_interval = 300.0
        self.t0 = datetime.now(timezone.utc)
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def test_ignores_non_person_and_false_positive(self):
        """Only true_positive person tracks are counted."""
        car = make_object(obj_id="c1", label="car", true_positive=True)
        shadow = make_object(obj_id="p1", label="person", true_positive=False)

        events = self.plugin.on_objects("cam1", self.frame, self.t0, [car, shadow], [])
        self.assertEqual(events, [])
        self.assertEqual(self.plugin.get_count("cam1", "room"), 0)
        self.assertEqual(self.plugin.get_count("cam1", "all"), 0)

    def test_counts_stationary_person(self):
        """Stationary person (sitting at desk/couch) is still counted."""
        person = make_object(obj_id="p1", label="person", true_positive=True, stationary=True)
        # 3 frames inertia (0.6s) + 3.0s debounce = 3.6s. Run 20 frames (4.0s).
        for i in range(20):
            t = self.t0 + timedelta(seconds=0.2 * i)
            events = self.plugin.on_objects("cam1", self.frame, t, [person], [])

        # Stable count should now be 1
        self.assertEqual(self.plugin.get_count("cam1", "room"), 1)
        self.assertEqual(self.plugin.get_count("cam1", "all"), 1)

    def test_debounce_ignores_transient_dropouts(self):
        """A person walking behind a pillar (1 or 2 second drop) does not make count flap."""
        p1 = make_object(obj_id="p1", label="person", true_positive=True)
        p2 = make_object(obj_id="p2", label="person", true_positive=True)

        # 1. Establish count of 2 people over 4.0s
        for i in range(20):
            t = self.t0 + timedelta(seconds=0.2 * i)
            events = self.plugin.on_objects("cam1", self.frame, t, [p1, p2], [])

        self.assertEqual(self.plugin.get_count("cam1", "room"), 2)

        # 2. Transient dropout of p2 for 1.4 seconds (walking behind pillar)
        t_base = self.t0 + timedelta(seconds=4.0)
        for i in range(7):  # 1.4 seconds < 3.0s debounce
            t = t_base + timedelta(seconds=0.2 * i)
            events = self.plugin.on_objects("cam1", self.frame, t, [p1], [])
            # Must NOT emit any count_changed event for room
            room_count_events = [
                e for e in events if e.label == "occupancy" and e.data.get("zone") == "room"
            ]
            self.assertEqual(room_count_events, [])

        # Count is STILL 2
        self.assertEqual(self.plugin.get_count("cam1", "room"), 2)

        # 3. p2 reappears: candidate count is back to 2, timer cleared
        t_reappear = t_base + timedelta(seconds=1.6)
        events = self.plugin.on_objects("cam1", self.frame, t_reappear, [p1, p2], [])
        self.assertEqual(self.plugin.get_count("cam1", "room"), 2)

    def test_count_transition_sequence(self):
        """Count transitions: 0 -> 1 -> 2 -> 1 -> 0 with debounce."""
        p1 = make_object(obj_id="p1", label="person", true_positive=True)
        p2 = make_object(obj_id="p2", label="person", true_positive=True)

        # 0 -> 1: p1 enters and stays for 4.0s (3 frames inertia + 3s debounce)
        for i in range(20):
            t = self.t0 + timedelta(seconds=0.2 * i)
            events = self.plugin.on_objects("cam1", self.frame, t, [p1], [])

        self.assertEqual(self.plugin.get_count("cam1", "room"), 1)

        # 1 -> 2: p2 enters and stays for 4.0s
        t_base = self.t0 + timedelta(seconds=5.0)
        for i in range(20):
            t = t_base + timedelta(seconds=0.2 * i)
            events = self.plugin.on_objects("cam1", self.frame, t, [p1, p2], [])

        self.assertEqual(self.plugin.get_count("cam1", "room"), 2)

        # 2 -> 0: both leave and stay absent for 4.0s
        t_base = self.t0 + timedelta(seconds=10.0)
        for i in range(20):
            t = t_base + timedelta(seconds=0.2 * i)
            events = self.plugin.on_objects("cam1", self.frame, t, [], [])

        self.assertEqual(self.plugin.get_count("cam1", "room"), 0)

    def test_over_capacity_alert_and_rearm(self):
        """Capacity limit = 2. Alert fires when stable count reaches 3, re-arms when dropping back."""
        p1 = make_object(obj_id="p1", label="person", true_positive=True)
        p2 = make_object(obj_id="p2", label="person", true_positive=True)
        p3 = make_object(obj_id="p3", label="person", true_positive=True)

        # 3 people enter and stay 4.0s -> count becomes 3 (over limit 2)
        over_cap_events = []
        for i in range(20):
            t = self.t0 + timedelta(seconds=0.2 * i)
            evts = self.plugin.on_objects("cam1", self.frame, t, [p1, p2, p3], [])
            for e in evts:
                if e.label == "over_capacity" and e.data.get("zone") == "room":
                    over_cap_events.append(e)

        self.assertEqual(len(over_cap_events), 1)
        self.assertEqual(over_cap_events[0].data["count"], 3)
        self.assertEqual(over_cap_events[0].data["capacity_limit"], 2)

        # Further frames with 3 people do NOT spam over_capacity
        t_next = self.t0 + timedelta(seconds=5.0)
        evts = self.plugin.on_objects("cam1", self.frame, t_next, [p1, p2, p3], [])
        self.assertEqual(
            [e for e in evts if e.label == "over_capacity" and e.data.get("zone") == "room"],
            [],
        )

        # Drop back to 2 people for 4.0s
        t_drop = self.t0 + timedelta(seconds=6.0)
        for i in range(20):
            t = t_drop + timedelta(seconds=0.2 * i)
            self.plugin.on_objects("cam1", self.frame, t, [p1, p2], [])

        self.assertEqual(self.plugin.get_count("cam1", "room"), 2)
        self.assertFalse(self.plugin.debouncer.get_scope("cam1", "room").is_over_capacity)

    def test_summary_event_at_interval(self):
        """Periodic summary is emitted at summary_interval (300 s)."""
        p1 = make_object(obj_id="p1", label="person", true_positive=True)
        self.plugin.on_objects("cam1", self.frame, self.t0, [p1], [])

        # Fast forward time by 305 seconds
        t_summary = self.t0 + timedelta(seconds=305)
        evts = self.plugin.on_objects("cam1", self.frame, t_summary, [p1], [])
        room_summary = [
            e for e in evts if e.label == "summary" and e.data.get("zone") == "room"
        ]
        self.assertEqual(len(room_summary), 1)
        self.assertEqual(room_summary[0].data["summary_interval"], 300.0)

    def test_frontend_occupancy_now_contract_no_double_counting(self):
        """Fine-tuning for frontend UI: Ensure 'all' is not emitted as a zone,
        preventing double-counting in cctv-logic.js occupancyNow().
        """
        p1 = make_object(obj_id="p1", label="person", true_positive=True)
        p2 = make_object(obj_id="p2", label="person", true_positive=True)

        emitted_events = []
        for i in range(20):
            t = self.t0 + timedelta(seconds=0.2 * i)
            evts = self.plugin.on_objects("cam1", self.frame, t, [p1, p2], [])
            emitted_events.extend(evts)

        # Check emitted zones: "room" should be present, "all" MUST NOT be an emitted event
        zone_names = [e.data.get("zone") for e in emitted_events if e.label == "occupancy"]
        self.assertIn("room", zone_names)
        self.assertNotIn("all", zone_names)

        # Simulate frontend occupancyNow() logic exactly from cctv-logic.js
        # (newest event per camera + zone, summed to total)
        latest = {}
        for idx, e in enumerate(emitted_events):
            d = e.data or {}
            zone = d.get("zone") or ""
            key = (e.camera, zone)
            latest[key] = {
                "id": idx + 1,
                "camera": e.camera,
                "zone": zone,
                "count": round(float(d.get("count", 0))),
            }

        zones = list(latest.values())
        total = sum(z["count"] for z in zones)

        # Total MUST match exact headcount (2), not doubled (4)
        self.assertEqual(total, 2)
        self.assertEqual(len(zones), 1)
        self.assertEqual(zones[0]["zone"], "room")
        self.assertEqual(zones[0]["count"], 2)

    def test_subzones_aggregation_matches_ui(self):
        """Verify multi-zone setup with subzones (e.g. Desk and Counter) aggregates properly."""
        from cctv_ai.zones import ZoneConfig, ZoneTracker

        # Configure custom subzones for cam1
        subzones = {
            "Desk": ZoneConfig(name="Desk", points=[[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]),
            "Counter": ZoneConfig(name="Counter", points=[[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]),
        }
        self.plugin.trackers["cam1"] = ZoneTracker(camera="cam1", zones=subzones)

        # Person 1 on the left (Desk), Person 2 on the right (Counter)
        p1 = make_object(obj_id="p1", label="person", true_positive=True, box=(100, 100, 200, 300))
        p2 = make_object(obj_id="p2", label="person", true_positive=True, box=(800, 100, 900, 300))

        emitted_events = []
        for i in range(20):
            t = self.t0 + timedelta(seconds=0.2 * i)
            evts = self.plugin.on_objects("cam1", self.frame, t, [p1, p2], [])
            emitted_events.extend(evts)

        # Simulate frontend occupancyNow()
        latest = {}
        for idx, e in enumerate(emitted_events):
            d = e.data or {}
            zone = d.get("zone") or ""
            key = (e.camera, zone)
            latest[key] = {
                "id": idx + 1,
                "camera": e.camera,
                "zone": zone,
                "count": round(float(d.get("count", 0))),
            }

        zones = sorted(latest.values(), key=lambda z: z["zone"])
        total = sum(z["count"] for z in zones)

        self.assertEqual(total, 2)
        self.assertEqual([z["zone"] for z in zones], ["Counter", "Desk"])
        self.assertEqual([z["count"] for z in zones], [1, 1])


if __name__ == "__main__":
    unittest.main()
