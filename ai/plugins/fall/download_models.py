"""download_models.py: Download lightweight pre-trained ONNX pose model (MoveNet Lightning, Apache 2.0).
Model files (*.onnx, *.pt, *.tflite) are git-ignored.
"""
from __future__ import annotations

import os
import urllib.request

# MoveNet SinglePose Lightning ONNX (Apache 2.0, ~9 MB)
HF_URL = "https://huggingface.co/onnx-community/movenet-lightning-web/resolve/main/onnx/model.onnx"
MOVENET_ONNX_URL = "https://raw.githubusercontent.com/onnx/models/main/validated/vision/body_analysis/movenet/movenet_singlepose_lightning.onnx"
FALLBACK_URL = "https://github.com/PINTO0309/PINTO_model_zoo/raw/main/115_MoveNet/movenet_single_pose_lightning_4.onnx"

DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
DEFAULT_PATH = os.path.join(DEFAULT_DIR, "movenet_lightning.onnx")


def download_model(dest_path: str = DEFAULT_PATH) -> str:
    """Download the MoveNet ONNX model if not already present."""
    if os.path.exists(dest_path):
        return dest_path

    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    print(f"[Fall Plugin] Downloading MoveNet Lightning ONNX to {dest_path}...")

    urls = [HF_URL, FALLBACK_URL, MOVENET_ONNX_URL]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as f:
                f.write(resp.read())
            print(f"[Fall Plugin] Downloaded model successfully from {url}")
            return dest_path
        except Exception as err:
            print(f"[Fall Plugin] Download from {url} failed: {err}")

    return dest_path


if __name__ == "__main__":
    download_model()
