"""Vehicle LPR tests. Run from ai/: python -m unittest discover -s tests -v

Cleaning, format check, box shaping, merging and clustering are pure logic.
The OCR test needs rapidocr-onnxruntime (and a few seconds); it is skipped if
the package or the sample plate image is missing.
"""

import unittest
from datetime import datetime, timezone
from pathlib import Path
from collections import deque
import numpy as np

from cctv_ai.objects import TrackedObject
from plugins.vehicle_lpr import plates
from plugins.vehicle_lpr import Plugin as LprPlugin, VehicleObjectTracker
from plugins.vehicle_lpr.clustering import PlateRead, best_plate, cluster_reads

SAMPLE_PLATE = Path(__file__).resolve().parents[0] / "test_images" / "plate_sample.png"


class TestCleaningAndFormat(unittest.TestCase):
    def test_clean_text(self):
        self.assertEqual(plates.clean_text("tn 09-ab 1234"), "TN09AB1234")
        self.assertEqual(plates.clean_text("TN-09-AB-1234!"), "TN09AB1234")

    def test_valid_formats(self):
        self.assertTrue(plates.is_valid("TN09AB1234"))
        self.assertTrue(plates.is_valid("TN9AB1234"))  # 1-digit district
        self.assertTrue(plates.is_valid("TN09ABC1234"))  # 3-letter series
        self.assertTrue(plates.is_valid("22BH1234AB"))  # Bharat series
        self.assertTrue(plates.is_valid("22BH1234A"))

    def test_invalid_plates(self):
        self.assertFalse(plates.is_valid("TN09AB123"))  # 3-digit number
        self.assertFalse(plates.is_valid("TN09AB12345"))  # 5-digit number
        self.assertFalse(plates.is_valid("T909AB1234"))  # digit in state code
        self.assertFalse(plates.is_valid("HELLO"))

    def test_normalize_passes_good_text(self):
        self.assertEqual(plates.normalize("tn 09 ab 1234"), "TN09AB1234")

    def test_repair_letter_digit_swaps(self):
        self.assertEqual(plates.normalize("TNO9AB1234"), "TN09AB1234")  # O -> 0
        self.assertEqual(plates.normalize("TN09ABI234"), "TN09AB1234")  # I -> 1
        self.assertEqual(plates.normalize("TN09ABIZ34"), "TN09AB1234")
        self.assertEqual(plates.normalize("2ZBHIZ34AB"), "22BH1234AB")  # Bharat
        self.assertIsNone(plates.normalize("RANDOMTEXT"))  # nothing fits

    def test_min_length(self):
        # min_length is a pre-filter; no valid Indian plate is shorter than 7
        self.assertIsNone(plates.normalize("TN09"))
        self.assertIsNone(plates.normalize("TN09AB1234", min_length=15))
        self.assertEqual(plates.normalize("TN09AB1234", min_length=7), "TN09AB1234")


class TestPlateBoxes(unittest.TestCase):
    def test_shape_filter(self):
        self.assertTrue(plates.is_plate_shaped(200, 50))  # car plate 4:1
        self.assertTrue(plates.is_plate_shaped(120, 60))  # bike plate 2:1
        self.assertTrue(plates.is_plate_shaped(388, 54))  # long single-line text box 7:1
        self.assertTrue(plates.is_plate_shaped(348, 41))  # tight OCR box 8.5:1 (real read)
        self.assertFalse(plates.is_plate_shaped(100, 100))  # square sign
        self.assertFalse(plates.is_plate_shaped(400, 30))  # 13:1: a banner, not a plate
        self.assertFalse(plates.is_plate_shaped(40, 10))  # under min_area
        self.assertTrue(plates.is_plate_shaped(60, 45, two_line=True))  # bike 2-line

    def test_merge_two_line_stack(self):
        top, bottom = (100.0, 100.0, 60.0, 25.0), (104.0, 133.0, 55.0, 25.0)
        noise = (400.0, 100.0, 60.0, 25.0)  # same line as `top`, far away
        merged = plates.merge_two_line([top, bottom, noise])
        self.assertEqual(len(merged), 2)
        stacked = [b for b in merged if b[3] > 50]  # only the stack is tall
        self.assertEqual(len(stacked), 1)
        x, y, w, h = stacked[0]
        self.assertAlmostEqual(x, 100.0)
        self.assertAlmostEqual(y, 100.0)
        self.assertAlmostEqual(h, 58.0)

    def test_merge_keeps_single_line_unchanged(self):
        line = (50.0, 60.0, 200.0, 50.0)
        self.assertEqual(plates.merge_two_line([line]), [line])

    def test_merge_does_not_join_side_by_side(self):
        a, b = (100.0, 100.0, 60.0, 25.0), (200.0, 100.0, 60.0, 25.0)  # no overlap
        self.assertEqual(len(plates.merge_two_line([a, b])), 2)


class TestClustering(unittest.TestCase):
    def test_identical_reads_cluster(self):
        reads = [PlateRead("TN09AB1234", 0.95), PlateRead("TN09AB1234", 0.9),
                 PlateRead("TN09AB1234", 0.85)]
        result = best_plate(reads)
        self.assertEqual(result, ("TN09AB1234", 0.95, 3))

    def test_ocr_noise_loses_to_majority(self):
        # greedy clustering (Frigate): the O/0 variant is similar enough to join
        # the big cluster, and the cluster's most confident read is the rep.
        # (In the plugin such a read is repaired to TN09AB1234 by normalize
        # before it ever reaches the clustering.)
        reads = ([PlateRead("TN09AB1234", 0.9)] * 5
                 + [PlateRead("TNO9AB1234", 0.95), PlateRead("XX12", 0.99)])
        plate, conf, agreeing = best_plate(reads)
        self.assertEqual(agreeing, 6)  # the short garbage read is its own cluster
        self.assertIn(plate, {"TN09AB1234", "TNO9AB1234"})

    def test_close_variations_cluster_together(self):
        reads = [PlateRead("TN09AB1234", 0.9), PlateRead("TN09AB1235", 0.92),
                 PlateRead("TN09AB8234", 0.88)]
        clusters = cluster_reads(reads, threshold=0.85)
        self.assertEqual(len(clusters), 1)  # Jaro-Winkler >= 0.85 keeps them one

    def test_different_plates_split(self):
        reads = [PlateRead("TN09AB1234", 0.9)] * 3 + [PlateRead("KA05MJ4321", 0.95)] * 2
        clusters = cluster_reads(reads)
        self.assertEqual(len(clusters), 2)
        plate, _, agreeing = best_plate(reads)
        self.assertEqual((plate, agreeing), ("TN09AB1234", 3))  # biggest cluster wins

    def test_empty(self):
        self.assertIsNone(best_plate([]))


class TestCandidateGating(unittest.TestCase):
    def test_ocr_runs_only_for_vehicle_candidates(self):
        class OCR:
            def __init__(self): self.calls = 0
            def detect_text(self, _crop):
                self.calls += 1
                return []

        plugin = LprPlugin()
        plugin.ready = True
        plugin.ocr = OCR()
        plugin.reads = {"cam1": deque(maxlen=plugin.max_reads)}
        plugin.already_read = set()
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        ts = datetime.now(timezone.utc)
        person = TrackedObject("person1", "person", (10, 10, 80, 190), 0.9, 0.9, True)
        plugin.on_objects("cam1", frame, ts, [person], [])
        self.assertEqual(plugin.ocr.calls, 0)
        car = TrackedObject("car1", "car", (40, 40, 280, 180), 0.8, 0.8, True)
        plugin.on_objects("cam1", frame, ts, [car], [])
        self.assertEqual(plugin.ocr.calls, 1)

    def test_vehicle_candidate_tracker_keeps_identity_and_ends_stale_tracks(self):
        tracker = VehicleObjectTracker(max_missed=2)
        ts = datetime.now(timezone.utc)
        detection = {"label": "car", "box": (10, 10, 210, 110), "score": 0.9}
        active, ended = tracker.update("cam1", [detection], ts)
        self.assertEqual(len(active), 1)
        self.assertEqual(ended, [])
        track_id = active[0].id
        active, ended = tracker.update("cam1", [{**detection, "box": (12, 10, 212, 110)}], ts)
        self.assertEqual(active[0].id, track_id)
        self.assertEqual(ended, [])
        tracker.update("cam1", [], ts)
        active, ended = tracker.update("cam1", [], ts)
        self.assertEqual(active, [])
        self.assertEqual([obj.id for obj in ended], [track_id])


class TestKnownPlates(unittest.TestCase):
    def test_levenshtein_match(self):
        from rapidfuzz.distance import Levenshtein

        self.assertLessEqual(Levenshtein.distance("TN09AB1234", "TN09AB1235"), 1)
        self.assertGreater(Levenshtein.distance("TN09AB1234", "KA05MJ4321"), 1)


@unittest.skipUnless(SAMPLE_PLATE.exists(), "sample plate image missing")
class TestTextInBox(unittest.TestCase):
    TOP = ((10.0, 10.0, 100.0, 20.0), "TN 09", 0.90)
    BOTTOM = ((10.0, 34.0, 100.0, 20.0), "AB 1234", 0.99)

    def test_joins_top_to_bottom_regardless_of_confidence_order(self):
        # the bottom line is more confident, but the text must still read
        # top line first: "TN 09AB 1234", not "AB 1234TN 09"
        assembled = plates.text_in_box([self.BOTTOM, self.TOP], (10.0, 10.0, 100.0, 44.0))
        self.assertIsNotNone(assembled)
        text, conf, count, two_line = assembled
        self.assertEqual(plates.clean_text(text), "TN09AB1234")
        self.assertAlmostEqual(conf, (0.90 + 0.99) / 2, places=6)
        self.assertEqual(count, 2)
        self.assertTrue(two_line)

    def test_single_line_is_not_two_line(self):
        line = ((10.0, 10.0, 200.0, 40.0), "TN09AB1234", 0.95)
        text, _conf, count, two_line = plates.text_in_box([line], (10.0, 10.0, 200.0, 40.0))
        self.assertEqual(text, "TN09AB1234")
        self.assertEqual(count, 1)
        self.assertFalse(two_line)

    def test_lines_outside_the_box_are_ignored(self):
        other = ((300.0, 300.0, 100.0, 20.0), "XXXX", 0.99)
        assembled = plates.text_in_box([self.TOP, other], (10.0, 10.0, 100.0, 44.0))
        self.assertEqual(assembled[0], "TN 09")
        self.assertEqual(assembled[2], 1)

    def test_empty_box_has_no_text(self):
        self.assertIsNone(plates.text_in_box([], (0.0, 0.0, 10.0, 10.0)))


class TestPluginReadAssembly(unittest.TestCase):
    def _plugin(self):
        from plugins.vehicle_lpr import Plugin

        plugin = Plugin()  # no setup(): no models needed for _reads_from_lines
        return plugin

    def test_two_line_reads_are_assembled_top_first(self):
        plugin = self._plugin()
        lines = [
            ((10.0, 34.0, 100.0, 20.0), "AB 1234", 0.99),  # bottom, more confident
            ((10.0, 10.0, 100.0, 20.0), "TN 09", 0.90),    # top
        ]
        reads = plugin._reads_from_lines(lines)
        self.assertEqual(len(reads), 1)
        self.assertEqual(reads[0].plate, "TN09AB1234")

    def test_text_that_is_not_a_plate_is_dropped(self):
        plugin = self._plugin()
        lines = [((10.0, 10.0, 200.0, 40.0), "SALE 50 PERCENT OFF", 0.99)]
        self.assertEqual(plugin._reads_from_lines(lines), [])

    def test_low_confidence_reads_are_dropped(self):
        plugin = self._plugin()
        lines = [((10.0, 10.0, 200.0, 40.0), "TN09AB1234", 0.5)]
        self.assertEqual(plugin._reads_from_lines(lines), [])


class TestPerCameraThrottle(unittest.TestCase):
    def test_each_camera_is_throttled_independently(self):
        """Two cameras on one plugin must each get their own OCR slot."""
        import numpy as np

        from plugins.vehicle_lpr import Plugin

        class FakeOCR:
            def __init__(self):
                self.calls = 0

            def detect_text(self, frame):
                self.calls += 1
                return []

        plugin = Plugin()
        plugin.ready = True
        plugin.ocr = FakeOCR()
        plugin.reads = {}
        plugin.announced = {}
        plugin.last_ocr = {}
        frame = np.zeros((360, 640, 3), np.uint8)
        now = datetime.now(timezone.utc)

        plugin.on_frame("cam1", frame, now)
        plugin.on_frame("cam2", frame, now)  # camera 2 must not wait behind camera 1
        self.assertEqual(plugin.ocr.calls, 2)

        plugin.on_frame("cam1", frame, now)  # same camera within 1 s: throttled
        self.assertEqual(plugin.ocr.calls, 2)


class TestPrepareCrop(unittest.TestCase):
    def test_small_crop_is_upscaled(self):
        import numpy as np

        from plugins.vehicle_lpr.ocr import prepare_crop

        crop = np.zeros((80, 160, 3), np.uint8)
        out = prepare_crop(crop, min_height=200, max_zoom=2.0)
        self.assertEqual(out.shape[0], 160)  # zoom capped at 2x
        self.assertEqual(out.shape[1], 320)

    def test_zoom_is_uncapped_below_the_limit(self):
        import numpy as np

        from plugins.vehicle_lpr.ocr import prepare_crop

        crop = np.zeros((100, 400, 3), np.uint8)
        out = prepare_crop(crop, min_height=200, max_zoom=3.0)
        self.assertEqual(out.shape[0], 200)

    def test_large_crop_unchanged(self):
        import numpy as np

        from plugins.vehicle_lpr.ocr import prepare_crop

        crop = np.zeros((240, 600, 3), np.uint8)
        self.assertIs(prepare_crop(crop), crop)

    def test_empty_crop_unchanged(self):
        import numpy as np

        from plugins.vehicle_lpr.ocr import prepare_crop

        empty = np.zeros((0, 0, 3), np.uint8)
        self.assertIs(prepare_crop(empty), empty)


class TestOCROnSample(unittest.TestCase):
    def test_reads_sample_plate(self):
        import cv2
        import numpy as np

        from plugins.vehicle_lpr.ocr import PlateOCR

        ocr = PlateOCR()
        if not ocr.load():
            self.skipTest("rapidocr-onnxruntime not installed")
        image = cv2.imdecode(np.fromfile(str(SAMPLE_PLATE), np.uint8), cv2.IMREAD_COLOR)
        lines = ocr.detect_text(image)
        self.assertGreaterEqual(len(lines), 1, f"no text found in {SAMPLE_PLATE}")
        text, conf = lines[0][1], lines[0][2]
        self.assertGreater(conf, 0.8)
        self.assertEqual(plates.normalize(text), "TN09AB1234")

    def test_vehicle_candidate_crop_emits_a_confirmed_plate(self):
        import cv2
        import numpy as np

        plugin = LprPlugin()
        plugin.setup(["cam1"])
        if not plugin.ready:
            self.skipTest("rapidocr-onnxruntime not installed")
        image = cv2.imdecode(np.fromfile(str(SAMPLE_PLATE), np.uint8), cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        vehicle = TrackedObject("cam1-vehicle1", "car", (0, 0, w, h), 0.9, 0.9, True)
        events = []
        for _ in range(plugin.confirm_reads):
            events.extend(plugin.on_objects("cam1", image, datetime.now(timezone.utc), [vehicle], []))
        self.assertTrue(any(e.data.get("plate") == "TN09AB1234" for e in events), events)


if __name__ == "__main__":
    unittest.main()
