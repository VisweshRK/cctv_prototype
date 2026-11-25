"""Run from ai/:  python -m unittest discover -s tests -v"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from cctv_ai.client import Go2rtc
from cctv_ai.plugin import Event


class FakeResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return {"id": 1}


class FakeSession:
    """Stands in for requests.Session: records each POST, no network."""

    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []

    def post(self, url: str, json: Any = None, **kwargs: Any) -> FakeResponse:
        self.posts.append({"url": url, "json": json, **kwargs})
        return FakeResponse()


def make_client() -> tuple[Go2rtc, FakeSession]:
    client = Go2rtc("http://go2rtc.invalid:1984")
    fake = FakeSession()
    client.http = fake  # type: ignore[assignment]
    return client, fake


class TestSendEvent(unittest.TestCase):
    def test_sends_image_as_base64_jpeg(self):
        client, fake = make_client()
        frame = np.zeros((90, 160, 3), np.uint8)
        frame[20:60, 40:100] = (0, 0, 255)
        event = Event("cam1", "tampering", "lens_covered", 0.9, data={"area": "Server room door"})

        self.assertEqual(client.send_event(event, image=frame), {"id": 1})

        self.assertEqual(len(fake.posts), 1)
        post = fake.posts[0]
        self.assertEqual(post["url"], "http://go2rtc.invalid:1984/api/cctv/events")
        body = post["json"]
        self.assertTrue(body["image"].startswith("/9j/"), body["image"][:8])  # base64 of FF D8
        self.assertEqual(body["camera"], "cam1")
        self.assertEqual(body["data"], {"area": "Server room door"})

    def test_no_image_key_without_image(self):
        client, fake = make_client()
        event = Event("cam1", "fall", "person", 0.8)
        client.send_event(event)
        self.assertNotIn("image", fake.posts[0]["json"])
        self.assertEqual(fake.posts[0]["json"], event.to_json())


if __name__ == "__main__":
    unittest.main()
