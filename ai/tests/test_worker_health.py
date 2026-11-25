from __future__ import annotations

import unittest

from cctv_ai.worker import health_document


class FakePlugin:
    def __init__(self, use_case, needs=None, **attrs):
        self.use_case = use_case
        self.needs = needs or {"video"}
        for key, value in attrs.items():
            setattr(self, key, value)


class WorkerHealthTests(unittest.TestCase):
    def test_reports_missing_models_gallery_and_appliance_regions_as_degraded(self):
        plugins = [
            FakePlugin("face_recognition", ready=True, people_known=0),
            FakePlugin("baby_crying", needs={"audio"}, model_version="harmonic-fallback"),
            FakePlugin("fall", needs={"video", "objects"}, pose_ready=False),
            FakePlugin("vehicle_lpr", ready=False),
            FakePlugin("power_when_absent", needs={"video", "objects"}, rois={}),
        ]
        health = health_document(plugins, detector=None)
        self.assertEqual([c["id"] for c in health["use_cases"]], [p.use_case for p in plugins])
        self.assertEqual([c["state"] for c in health["use_cases"]],
                         ["degraded", "degraded", "degraded", "degraded", "degraded"])
        self.assertTrue(all(c["reason"] for c in health["use_cases"]))

    def test_shared_detector_failure_disables_object_use_cases(self):
        plugin = FakePlugin("occupancy", needs={"objects"})
        health = health_document([plugin], detector=None)
        self.assertEqual(health["use_cases"][0]["state"], "disabled")

    def test_missing_audio_track_disables_baby_crying_case(self):
        plugin = FakePlugin("baby_crying", needs={"audio"}, model_version="yamnet-onnx")
        health = health_document([plugin], detector=None,
                                 audio_state={"cam1": {"state": "unavailable"}, "cam2": {"state": "unavailable"}})
        self.assertEqual(health["use_cases"][0]["state"], "disabled")
        self.assertIn("no audio stream", health["use_cases"][0]["reason"])

    def test_stopped_worker_is_reported_offline(self):
        health = health_document([FakePlugin("tampering")], detector=None, worker_state="offline")
        self.assertEqual(health["worker"], "offline")

    def test_health_includes_capture_and_lane_metrics(self):
        runtime = {"capture": {"cam1": {"fps": 14.2}},
                   "lanes": {"cam1:vehicle_lpr": {"last_ms": 620.0, "dropped": 4, "queued": 0}}}
        health = health_document([FakePlugin("vehicle_lpr")], detector=None, runtime=runtime)
        self.assertEqual(health["metrics"]["capture"]["cam1"]["fps"], 14.2)
        self.assertEqual(health["metrics"]["lanes"]["cam1:vehicle_lpr"]["dropped"], 4)


if __name__ == "__main__":
    unittest.main()
