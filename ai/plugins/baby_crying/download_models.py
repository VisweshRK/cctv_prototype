"""Download the pinned YAMNet ONNX classifier into the ignored local models folder."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

MODEL_URL = (
    "https://raw.githubusercontent.com/uhstray-io/zerds-public/main/"
    "models/yamnet/model.onnx"
)
MODEL_SHA256 = "9a6aef46391df10a4ff804ea196266855a532a200ec13894024da24e16aaf7b1"


def checksum_matches(data: bytes, expected: str = MODEL_SHA256) -> bool:
    return hashlib.sha256(data).hexdigest().lower() == expected.lower()


def model_path() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / "yamnet.onnx"


def download(target: Path | None = None) -> Path:
    target = target or model_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and checksum_matches(target.read_bytes()):
        return target

    temporary = None
    try:
        with urlopen(MODEL_URL, timeout=60) as response:
            data = response.read()
        if not checksum_matches(data):
            raise ValueError("YAMNet download failed SHA-256 verification")
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
        os.replace(temporary, target)
        return target
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> None:
    print(f"YAMNet ready: {download()}")


if __name__ == "__main__":
    main()
