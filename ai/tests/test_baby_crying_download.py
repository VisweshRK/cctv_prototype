import hashlib
import unittest

from plugins.baby_crying.download_models import checksum_matches


class BabyCryingModelDownloadTest(unittest.TestCase):
    def test_checksum_matches_expected_model_bytes(self):
        data = b"yamnet weights"
        digest = hashlib.sha256(data).hexdigest()
        self.assertTrue(checksum_matches(data, digest))
        self.assertFalse(checksum_matches(data, "0" * 64))


if __name__ == "__main__":
    unittest.main()
