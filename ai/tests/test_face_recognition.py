"""Face recognition tests. Run from ai/: python -m unittest discover -s tests -v

The voting and gallery tests are pure logic. The engine tests need the ONNX
models (download_models.py) and the OpenCV Zoo sample face, and are skipped
when those are missing, so CI without models still passes.
"""

import unittest
from pathlib import Path

import numpy as np

from plugins.face_recognition.engine import (
    MAX_POSE_OFFSET,
    MIN_SHARPNESS,
    SFACE_MATCH_THRESHOLD,
    Face,
    FaceEngine,
    blur_penalty,
    iou,
    pose_offset,
    similarity_to_confidence,
)
from plugins.face_recognition.gallery import build_class_mean, drop_outliers, trim_mean_rows
from plugins.face_recognition.voting import FaceVote, weighted_average

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
SAMPLE_FACE = Path(__file__).resolve().parents[0] / "test_images" / "face_sample.jpg"


class TestWeightedAverage(unittest.TestCase):
    def test_empty_and_unknown_only(self):
        self.assertEqual(weighted_average([], 0.8, 3), (None, 0.0))
        self.assertEqual(weighted_average([("unknown", 0.9, 4000)], 0.8, 3), (None, 0.0))

    def test_accepts_clear_majority(self):
        attempts = [("krithick", 0.95, 4000)] * 3 + [("unknown", 0.5, 2000)] * 4
        name, score = weighted_average(attempts, unknown_score=0.8, min_faces=3)
        self.assertEqual(name, "krithick")
        self.assertGreater(score, 0.9)

    def test_rejects_below_min_faces(self):
        name, score = weighted_average([("krithick", 0.95, 4000)] * 2, 0.8, min_faces=3)
        self.assertIsNone(name)
        self.assertEqual(score, 0.0)

    def test_rejects_count_tie(self):
        attempts = [("krithick", 0.95, 4000)] * 3 + [("viswesh", 0.9, 4000)] * 3
        self.assertEqual(weighted_average(attempts, 0.8, min_faces=3), (None, 0.0))

    def test_low_scores_drag_average_down(self):
        # weights are (score - 0.8) * 10 * area: a weaker attempt is outvoted
        attempts = [("krithick", 0.95, 4000)] * 3 + [("krithick", 0.85, 4000)]
        name, score = weighted_average(attempts, unknown_score=0.8, min_faces=3)
        self.assertEqual(name, "krithick")
        self.assertGreater(score, 0.9)
        # scores below the unknown level get negative weight: if they cancel the
        # positive weights entirely there is no confidence left at all
        attempts = [("krithick", 0.9, 4000)] * 3 + [("krithick", 0.5, 4000)]
        name, score = weighted_average(attempts, unknown_score=0.8, min_faces=3)
        self.assertEqual((name, score), (None, 0.0))

    def test_area_is_capped(self):
        big = [("krithick", 0.9, 100000)] * 3
        capped = [("krithick", 0.9, 4000)] * 3
        self.assertEqual(weighted_average(big, 0.8, 3), weighted_average(capped, 0.8, 3))


class TestFaceVote(unittest.TestCase):
    def test_accept_then_stop(self):
        vote = FaceVote()
        for _ in range(3):
            vote.add("krithick", 0.95, 4000)
        self.assertTrue(vote.reached_threshold())
        self.assertEqual(vote.accepted, "krithick")
        self.assertFalse(vote.should_attempt())

    def test_unknown_after_max_attempts(self):
        vote = FaceVote()
        for _ in range(vote.max_attempts):
            vote.add("unknown", 0.3, 2000)
        self.assertFalse(vote.should_attempt())
        self.assertTrue(vote.exhausted())
        name, _ = vote.decision()
        self.assertIsNone(name)

    def test_attempts_below_limit_continue(self):
        vote = FaceVote()
        vote.add("unknown", 0.3, 2000)
        self.assertTrue(vote.should_attempt())
        self.assertFalse(vote.exhausted())


class TestPluginEvents(unittest.TestCase):
    def test_stranger_is_reported_when_attempts_run_out(self):
        from plugins.face_recognition import Plugin

        plugin = Plugin()  # no setup(): this exercises the voting state only
        plugin.votes = {"cam1:visit1": FaceVote()}
        plugin.snapshots = {}
        plugin.boxes = {}
        plugin.announced = {}
        plugin.reported = set()
        vote = plugin.votes["cam1:visit1"]
        for _ in range(vote.max_attempts):
            vote.add(None, 0.3, 2000)

        events = plugin._events_for("cam1", "cam1:visit1", ended=False)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].use_case, "face_recognition")
        self.assertEqual(events[0].label, "unknown_person")
        self.assertEqual(events[0].confidence, 0.3)

        # the same visit must not be announced again while the face stays in view
        self.assertEqual(plugin._events_for("cam1", "cam1:visit1", ended=False), [])

    def test_rebuilt_gallery_reloads_while_worker_is_running(self):
        import tempfile
        from plugins.face_recognition import Plugin

        class Engine:
            names = []
            means = None
            def load_gallery(self, _path):
                self.names = ["alice"]
                self.means = np.ones((1, 128), np.float32)
                return 1

        plugin = Plugin()
        plugin.ready = True
        plugin.people_known = 0
        plugin.engine = Engine()
        with tempfile.TemporaryDirectory() as folder:
            plugin.gallery_path = Path(folder) / "faces.npz"
            plugin.gallery_mtime_ns = -1
            np.savez(plugin.gallery_path, names=np.array(["alice"]), means=np.ones((1, 128), np.float32))
            plugin._refresh_gallery()
            self.assertEqual(plugin.people_known, 1)
            plugin.gallery_path.unlink()
            plugin._refresh_gallery()
            self.assertEqual(plugin.people_known, 0)
            self.assertIsNone(plugin.engine.means)


class TestGallery(unittest.TestCase):
    def test_trim_mean_ignores_outliers_per_dimension(self):
        rows = np.full((10, 4), 1.0, np.float32)
        rows[0, 0] = 100.0  # one extreme value in one dimension
        mean = trim_mean_rows(rows, trim=0.15)
        self.assertEqual(mean[0], 1.0)  # 15% cut removes it
        self.assertTrue((mean == 1.0).all())

    def test_trim_mean_small_input(self):
        np.testing.assert_array_equal(trim_mean_rows(np.array([[2.0, 4.0]])), [2.0, 4.0])

    def test_drop_outliers_removes_wrong_face(self):
        rng = np.random.default_rng(7)
        base = np.ones(16, np.float32)  # the person's real direction
        good = [base + rng.normal(0, 0.01, 16).astype(np.float32) for _ in range(8)]
        good = [g / np.linalg.norm(g) for g in good]
        wrong = np.zeros(16, np.float32)  # a different face entirely
        wrong[3] = 1.0
        kept = drop_outliers(np.vstack([good + [wrong]]), min_similarity=0.363)
        self.assertEqual(kept.shape[0], 8)

    def test_drop_outliers_keeps_small_groups(self):
        rows = np.eye(3, dtype=np.float32)
        self.assertEqual(drop_outliers(rows, 0.363).shape[0], 3)  # never below 3

    def test_build_class_mean_is_normalised(self):
        rng = np.random.default_rng(1)
        rows = rng.normal(1.0, 0.05, (6, 8)).astype(np.float32)
        mean = build_class_mean(rows)
        self.assertAlmostEqual(float(np.linalg.norm(mean)), 1.0, places=5)


class TestScores(unittest.TestCase):
    def test_sigmoid_mapping(self):
        self.assertAlmostEqual(similarity_to_confidence(SFACE_MATCH_THRESHOLD), 0.5, places=6)
        self.assertGreater(similarity_to_confidence(0.7), 0.95)
        self.assertLess(similarity_to_confidence(0.1), 0.15)

    def test_blur_penalty(self):
        import cv2

        rng = np.random.default_rng(2)
        sharp = rng.normal(128, 40, (112, 112)).astype(np.uint8)
        blurry = cv2.GaussianBlur(sharp, (0, 0), 8)
        self.assertEqual(blur_penalty(blurry), 0.06)
        self.assertLessEqual(blur_penalty(sharp), 0.01)


def _face(eye_xs: tuple[float, float], nose_x: float) -> Face:
    """A fake YuNet row: box, right eye, left eye, nose, mouth, score."""
    (rx, lx), nx = eye_xs, nose_x
    row = np.array([10, 10, 80, 80, rx, 40, lx, 40, nx, 55, 35, 70, 65, 70, 0.9])
    return Face((10, 10, 80, 80), 0.9, row)


class TestGeometry(unittest.TestCase):
    def test_iou_identical_and_disjoint(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(iou((0, 0, 10, 10), (100, 100, 10, 10)), 0.0)

    def test_iou_partial_overlap(self):
        # two 10x10 boxes offset by 5: intersection 50, union 150
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 10, 10)), 1.0 / 3.0, places=6)

    def test_iou_touching_edges(self):
        self.assertEqual(iou((0, 0, 10, 10), (10, 0, 10, 10)), 0.0)


class TestQualityGate(unittest.TestCase):
    def test_frontal_face_passes_pose(self):
        offset = pose_offset(_face((30.0, 70.0), 50.0))  # nose centred on the eyes
        self.assertAlmostEqual(offset, 0.0, places=6)
        self.assertLessEqual(offset, MAX_POSE_OFFSET)

    def test_profile_face_fails_pose(self):
        # nose pushed to the right of the eye line: strong yaw
        offset = pose_offset(_face((30.0, 70.0), 84.0))
        self.assertGreater(offset, MAX_POSE_OFFSET)

    def test_degenerate_eye_distance_counts_as_profile(self):
        self.assertGreater(pose_offset(_face((50.0, 50.0), 50.0)), MAX_POSE_OFFSET)


def _yuunet_row(x: float, y: float) -> np.ndarray:
    """A minimal valid YuNet row anchored at (x, y) with frontal landmarks."""
    return np.array([x, y, 80, 80, x + 20, y + 30, x + 60, y + 30,
                     x + 40, y + 45, x + 25, y + 60, x + 55, y + 60, 0.9])


@unittest.skipUnless(
    (MODELS_DIR / "face_detection_yunet_2023mar.onnx").exists()
    and (MODELS_DIR / "face_recognition_sface_2021dec.onnx").exists()
    and SAMPLE_FACE.exists(),
    "models or sample face missing (run download_models.py)",
)
class TestEngineQualityGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = FaceEngine(MODELS_DIR)
        assert cls.engine.load_models()
        data = np.fromfile(str(SAMPLE_FACE), dtype=np.uint8)
        import cv2

        cls.image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    def test_recognise_all_without_gallery_is_unknown(self):
        matches = self.engine.recognise_all(self.image)
        self.assertGreaterEqual(len(matches), 1)
        match = matches[0]
        self.assertIsNone(match.name)  # no gallery loaded: everyone is unknown
        self.assertEqual(match.similarity, 0.0)
        self.assertIsNone(match.embedding)  # no gallery to compare against
        self.assertTrue(match.usable)  # a clean sample photo passes the gate

    def test_sharp_face_is_usable_blurry_is_not(self):
        import cv2

        match = self.engine.recognise_all(self.image)[0]
        self.assertTrue(match.usable)
        blurry = cv2.GaussianBlur(self.image, (0, 0), 8)
        blurred_match = self.engine.recognise_all(blurry)[0]
        self.assertFalse(blurred_match.usable)  # below MIN_SHARPNESS

    def test_set_detection_threshold_updates_live_detector(self):
        self.engine.set_detection_threshold(0.5)
        try:
            self.assertEqual(self.engine.detection_threshold, 0.5)
            # a lower threshold must find at least as many faces as 0.7
            self.engine.set_detection_threshold(0.7)
            strict = len(self.engine.detect(self.image))
            self.engine.set_detection_threshold(0.5)
            loose = len(self.engine.detect(self.image))
            self.assertGreaterEqual(loose, strict)
        finally:
            self.engine.set_detection_threshold(0.7)


@unittest.skipUnless(
    (MODELS_DIR / "face_detection_yunet_2023mar.onnx").exists()
    and (MODELS_DIR / "face_recognition_sface_2021dec.onnx").exists()
    and SAMPLE_FACE.exists(),
    "models or sample face missing (run download_models.py)",
)
class TestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = FaceEngine(MODELS_DIR)
        assert cls.engine.load_models()
        data = np.fromfile(str(SAMPLE_FACE), dtype=np.uint8)
        import cv2

        cls.image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    def test_detects_face_with_landmarks(self):
        faces = self.engine.detect(self.image)
        self.assertGreaterEqual(len(faces), 1)
        x, y, w, h = faces[0].box
        self.assertGreater(w * h, 0)
        self.assertTrue(0 <= x and 0 <= y)

    def test_embedding_shape(self):
        faces = self.engine.detect(self.image)
        embedding = self.engine.embed(self.image, faces[0])
        self.assertEqual(embedding.shape, (128,))
        self.assertGreater(float(np.linalg.norm(embedding)), 0.5)  # roughly unit length

    def test_recognise_without_gallery_is_unknown(self):
        name, score, face = self.engine.recognise(self.image)
        self.assertIsNone(name)
        self.assertIsNotNone(face)

    def test_same_face_matches_itself(self):
        faces = self.engine.detect(self.image)
        e1 = self.engine.embed(self.image, faces[0])
        aligned = self.engine.align_crop(self.image, faces[0])
        faces2 = self.engine.detect(aligned)
        e2 = self.engine.embed(aligned, faces2[0])
        similarity = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
        self.assertGreater(similarity, SFACE_MATCH_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
