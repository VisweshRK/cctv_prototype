"""download_models.py: Download lightweight pre-trained ONNX pose model (MoveNet Lightning, Apache 2.0).
Model files (*.onnx, *.pt, *.tflite) are git-ignored.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

# MoveNet SinglePose Lightning ONNX (Apache 2.0, ~9 MB)
HF_URL = "https://huggingface.co/onnx-community/movenet-lightning-web/resolve/main/onnx/model.onnx"
SHA256 = "3c841c042a998dc102a13532deafca5a0554d2faaab4511fe977c63c26cc5538"

DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
DEFAULT_PATH = os.path.join(DEFAULT_DIR, "movenet_lightning.onnx")


def download_model(dest_path: str = DEFAULT_PATH) -> str:
    """Download the MoveNet ONNX model if not already present."""
    target = Path(dest_path)
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() == SHA256:
            return dest_path
        raise ValueError(f"Existing MoveNet model failed SHA-256 verification: {target}")

    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    print(f"[Fall Plugin] Downloading MoveNet Lightning ONNX to {dest_path}...")
    temp = target.with_suffix(target.suffix + ".download")
    req = urllib.request.Request(HF_URL, headers={"User-Agent": "CCTV-Prototype/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, temp.open("wb") as f:
            f.write(resp.read())
        actual = hashlib.sha256(temp.read_bytes()).hexdigest()
        if actual != SHA256:
            raise ValueError(f"MoveNet SHA-256 mismatch: got {actual}, expected {SHA256}")
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    print(f"[Fall Plugin] Downloaded and verified MoveNet (SHA-256 {SHA256})")
    return dest_path


if __name__ == "__main__":
    download_model()
