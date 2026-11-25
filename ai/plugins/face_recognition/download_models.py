"""Download the YuNet and SFace ONNX models into ai/models/ (git-ignored).

Run from ai/:  .venv/Scripts/python -m plugins.face_recognition.download_models
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import requests

MODELS = {
    # OpenCV Zoo (https://github.com/opencv/opencv_zoo), Apache 2.0 / MIT.
    # SFace is stored with Git LFS, so it needs the media.githubusercontent URL.
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    models_dir = Path(__file__).resolve().parents[2] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in MODELS.items():
        target = models_dir / name
        if target.exists() and sha256(target) == expected:
            print(f"already present: {name}")
            continue
        print(f"downloading {name} ...")
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        target.write_bytes(response.content)
        actual = sha256(target)
        if actual != expected:
            print(f"  sha256 mismatch: {actual}\n  expected:        {expected}", file=sys.stderr)
            return 1
        print(f"  ok ({target.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
