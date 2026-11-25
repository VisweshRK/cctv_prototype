from __future__ import annotations

import threading
import logging
import time
import unittest

from cctv_ai.scheduler import LatestTaskLane


class LatestTaskLaneTests(unittest.TestCase):
    def test_slow_plugin_drops_old_frames(self):
        started, release = threading.Event(), threading.Event()
        seen = []

        def slow(value):
            seen.append(value)
            if value == 0:
                started.set()
                release.wait(2)

        lane = LatestTaskLane(slow, "test-slow")
        lane.submit(0)
        self.assertTrue(started.wait(1))
        lane.submit(1)
        lane.submit(2)
        lane.submit(3)
        release.set()
        deadline = time.monotonic() + 1
        while len(seen) < 2 and time.monotonic() < deadline:
            time.sleep(.005)
        lane.close()
        self.assertEqual(seen, [0, 3])
        self.assertEqual(lane.stats()["dropped"], 2)

    def test_lane_exception_does_not_stop_later_tasks(self):
        seen = []

        def sometimes_fails(value):
            started.set()
            if value == "bad":
                raise RuntimeError("expected test failure")
            seen.append(value)

        started = threading.Event()
        lane = LatestTaskLane(sometimes_fails, "test-failure")
        logging.disable(logging.CRITICAL)
        try:
            lane.submit("bad")
            self.assertTrue(started.wait(1))
            deadline = time.monotonic() + 1
            while lane.stats()["completed"] < 1 and time.monotonic() < deadline:
                time.sleep(.005)
            lane.submit("good")
            deadline = time.monotonic() + 1
            while not seen and time.monotonic() < deadline:
                time.sleep(.005)
            lane.close()
        finally:
            logging.disable(logging.NOTSET)
        self.assertEqual(seen, ["good"])
        self.assertEqual(lane.stats()["completed"], 2)

    def test_runtime_metrics_include_queue_and_drop_counts(self):
        started, release = threading.Event(), threading.Event()

        def slow(_):
            started.set()
            release.wait(2)

        lane = LatestTaskLane(slow, "test-metrics")
        lane.submit(1)
        self.assertTrue(started.wait(1))
        lane.submit(2)
        metrics = lane.stats()
        self.assertEqual(metrics["queued"], 1)
        self.assertEqual(metrics["dropped"], 0)
        release.set()
        lane.close()
        self.assertEqual(lane.stats()["completed"], 2)


if __name__ == "__main__":
    unittest.main()
