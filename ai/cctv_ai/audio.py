"""Read a camera's audio as 16 kHz mono PCM through ffmpeg.

Only used when an enabled plugin needs audio (e.g. baby_crying).
Requires ffmpeg on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Iterator

import numpy as np

SAMPLE_RATE = 16000


def audio_chunks(rtsp_url: str, seconds: float = 1.0) -> Iterator[np.ndarray]:
    """Yield int16 mono chunks of `seconds` length until the stream ends."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH (needed for audio plugins)")

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-rtsp_transport", "tcp", "-i", rtsp_url,
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-",
    ]
    chunk_bytes = int(SAMPLE_RATE * seconds) * 2
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        assert proc.stdout is not None
        while True:
            data = proc.stdout.read(chunk_bytes)
            if len(data) < chunk_bytes:
                return
            yield np.frombuffer(data, np.int16)
    finally:
        proc.kill()
