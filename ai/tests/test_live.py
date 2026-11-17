"""Run from ai/:  python -m unittest discover -s tests -v"""

import unittest

from test_bench.live import (
    EventCooldown,
    LiveStream,
    LiveTuning,
    mask_source,
    normalise_source,
)


class TestSourceURLs(unittest.TestCase):
    def test_accepts_rtsp_and_restreams(self):
        for url in (
            "rtsp://admin:secret@192.168.1.50:554/stream1",
            "rtsps://camera.local/live",
            "rtsp://localhost:8554/cam1",
            "http://localhost:1984/api/stream.mjpeg?src=cam1",
        ):
            with self.subTest(url=url):
                self.assertEqual(normalise_source(f"  {url}  "), url)

    def test_rejects_bad_sources(self):
        for raw in ("", "   ", "192.168.1.10/stream1", "ftp://host/file", "rtsp://"):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    normalise_source(raw)

    def test_rejects_whitespace_inside_url(self):
        with self.assertRaises(ValueError):
            normalise_source("rtsp://user:pa ss@host/stream1")

    def test_mask_hides_only_the_password(self):
        masked = mask_source("rtsp://admin:secret@192.168.1.50:554/stream1")
        self.assertEqual(masked, "rtsp://admin:***@192.168.1.50:554/stream1")
        self.assertNotIn("secret", masked)

    def test_mask_leaves_credential_free_urls_alone(self):
        url = "rtsp://localhost:8554/cam1"
        self.assertEqual(mask_source(url), url)

    def test_mask_hides_password_repeated_in_hikvision_path(self):
        url = (
            "rtsp://admin:secret@192.168.1.50:554/"
            "user=admin_password=secret_channel=0_stream=0&onvif=0.sdp?real_stream"
        )
        masked = mask_source(url)
        self.assertNotIn("secret", masked)
        self.assertIn("admin:***@192.168.1.50:554", masked)
        self.assertIn("password=***_channel=0_stream=0", masked)
        self.assertTrue(masked.endswith("?real_stream"))

    def test_mask_hides_password_in_query_without_userinfo(self):
        masked = mask_source("rtsp://192.168.1.50/s?password=secret&channel=0")
        self.assertNotIn("secret", masked)
        self.assertIn("password=***&channel=0", masked)

    def test_mask_keeps_suffix_when_inline_password_is_empty(self):
        url = (
            "rtsp://192.168.1.50:554/"
            "user=admin_password=_channel=0_stream=0&onvif=0.sdp?real_stream"
        )
        self.assertEqual(mask_source(url), url)

    def test_rejects_unfilled_placeholders(self):
        with self.assertRaises(ValueError):
            normalise_source(
                "rtsp://admin:<PASSWORD>@192.168.1.50:554/"
                "user=admin_password=<PASSWORD>_channel=0"
            )


class TestLiveTuning(unittest.TestCase):
    def test_clamps_untrusted_values(self):
        tuning = LiveTuning.from_payload({
            "mode": "nonsense",
            "fps": 0,
            "face_rec_thresh": 5,
            "min_face_area": -10,
            "contrast_boost": "true",
            "plate_every": 999,
        })
        self.assertEqual(tuning.mode, "both")
        self.assertEqual(tuning.infer_fps, 0.2)
        self.assertEqual(tuning.face_rec_thresh, 0.99)
        self.assertEqual(tuning.min_face_area, 0)
        self.assertTrue(tuning.contrast_boost)
        self.assertEqual(tuning.plate_every, 30)

    def test_plate_cadence_is_at_least_one_frame(self):
        for value, expected in ((0, 1), (-3, 1), ("2", 2), (None, 5)):
            with self.subTest(value=value):
                self.assertEqual(
                    LiveTuning.from_payload({"plate_every": value}).plate_every,
                    expected,
                )

    def test_keeps_base_values_when_payload_is_empty(self):
        base = LiveTuning(mode="faces", infer_fps=2.0, max_width=960)
        tuning = LiveTuning.from_payload({}, base)
        self.assertEqual(tuning.mode, "faces")
        self.assertEqual(tuning.infer_fps, 2.0)
        self.assertEqual(tuning.max_width, 960)

    def test_update_tuning_keeps_event_cooldown_state(self):
        stream = LiveStream(_NoopPipeline(), "rtsp://localhost:8554/cam1")
        stream._cooldown.ready("face:krithick", now=0.0)
        stream.update_tuning({"mode": "faces"})
        self.assertFalse(stream._cooldown.ready("face:krithick", now=1.0))
        stream.update_tuning({"event_cooldown": 0})
        self.assertTrue(stream._cooldown.ready("face:krithick", now=1.0))


class TestEventCooldown(unittest.TestCase):
    def test_suppresses_same_key_until_window_expires(self):
        cooldown = EventCooldown(10.0)
        self.assertTrue(cooldown.ready("face:krithick", now=100.0))
        self.assertFalse(cooldown.ready("face:krithick", now=105.0))
        self.assertTrue(cooldown.ready("face:other", now=105.0))
        self.assertTrue(cooldown.ready("face:krithick", now=111.0))


class TestPlateCadence(unittest.TestCase):
    def test_scans_every_nth_frame(self):
        due = [n for n in range(10) if LiveStream._plate_due(n, 5)]
        self.assertEqual(due, [0, 5])

    def test_never_skips_every_frame_mode(self):
        self.assertTrue(all(LiveStream._plate_due(n, 1) for n in range(5)))

    def test_clamps_to_at_least_one(self):
        self.assertTrue(LiveStream._plate_due(3, 0))
        self.assertTrue(LiveStream._plate_due(3, -2))


class TestSnapshot(unittest.TestCase):
    def setUp(self):
        self.stream = LiveStream(_NoopPipeline(), "rtsp://user:secret@10.0.0.5:554/s1")
        self.payload = {
            "width": 640,
            "height": 360,
            "faces": [],
            "plates": [],
            "telemetry": {"total_ms": 12.5},
            "events": [{"type": "known_person", "label": "krithick", "confidence": 0.9}],
            "image": "data:image/jpeg;base64,AAAA",
        }

    def test_returns_result_when_seq_is_new(self):
        self.stream._publish(self.payload, capture_lag_ms=80.0, inference_ms=12.5, finished=0.0)
        snapshot = self.stream.snapshot(after=0)
        self.assertFalse(snapshot["unchanged"])
        self.assertEqual(snapshot["seq"], 1)
        self.assertEqual(snapshot["image"], self.payload["image"])
        self.assertEqual(snapshot["stats"]["frames_processed"], 1)
        self.assertEqual(snapshot["events"][0]["label"], "krithick")
        self.assertEqual(snapshot["source"], "rtsp://user:***@10.0.0.5:554/s1")

    def test_returns_only_status_when_nothing_new(self):
        self.stream._publish(self.payload, capture_lag_ms=80.0, inference_ms=12.5, finished=0.0)
        snapshot = self.stream.snapshot(after=1)
        self.assertTrue(snapshot["unchanged"])
        self.assertNotIn("image", snapshot)
        self.assertEqual(snapshot["stats"]["capture_lag_ms"], 80.0)

    def test_snapshot_masks_credentials(self):
        self.stream._publish(self.payload, capture_lag_ms=0.0, inference_ms=0.0, finished=0.0)
        self.assertNotIn("secret", str(self.stream.snapshot(after=0)))


class _NoopPipeline:
    """The snapshot tests never start the inference loop."""

    def process_bgr(self, *args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("_NoopPipeline should not run inference")


if __name__ == "__main__":
    unittest.main()
