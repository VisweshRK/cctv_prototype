"""Small client for the go2rtc HTTP API (default http://localhost:1984)."""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np
import requests

from .plugin import Event


class Go2rtc:
    def __init__(self, url: str = "http://localhost:1984", timeout: float = 10.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.http = requests.Session()

    def status(self) -> list[dict[str, Any]]:
        """Camera list with online/offline state (GET /api/cctv/status)."""
        r = self.http.get(f"{self.url}/api/cctv/status", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def frame(self, camera: str) -> Optional[np.ndarray]:
        """Latest frame as a BGR image, or None if the camera has none.

        Uses go2rtc's /api/frame.jpeg. For H.264/H.265 cameras go2rtc needs
        ffmpeg installed to turn the frame into a JPEG.
        """
        r = self.http.get(f"{self.url}/api/frame.jpeg", params={"src": camera}, timeout=self.timeout)
        if r.status_code != 200 or not r.content:
            return None
        return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)

    def rtsp_url(self, camera: str, rtsp_port: int = 8554) -> str:
        """go2rtc restream address, e.g. for reading audio with ffmpeg."""
        host = self.url.split("://", 1)[-1].split(":", 1)[0].split("/", 1)[0]
        return f"rtsp://{host}:{rtsp_port}/{camera}"

    def send_event(self, event: Event) -> dict[str, Any]:
        r = self.http.post(f"{self.url}/api/cctv/events", json=event.to_json(), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def events(self, camera: str = "", use_case: str = "", limit: int = 100) -> list[dict[str, Any]]:
        params = {k: v for k, v in {"camera": camera, "use_case": use_case, "limit": limit}.items() if v}
        r = self.http.get(f"{self.url}/api/cctv/events", params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
