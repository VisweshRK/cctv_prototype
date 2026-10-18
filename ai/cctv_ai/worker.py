"""The AI worker: pulls frames/audio from go2rtc, runs plugins, sends events.

Run from the ai/ folder:
    python -m cctv_ai.worker                     # all plugins with enabled = True
    python -m cctv_ai.worker --plugins example_motion
    python -m cctv_ai.worker --list              # show every slot and its state

Each frame is fetched once per camera and shared by all video plugins, so
adding plugins doesn't multiply the load on the cameras.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from datetime import datetime, timezone

from .audio import SAMPLE_RATE, audio_chunks
from .client import Go2rtc
from .loader import discover
from .plugin import Event, Plugin

log = logging.getLogger("cctv_ai")


def run_plugin(plugin: Plugin, method: str, *args) -> list[Event]:
    """Call a plugin safely: a bug in one slot must not stop the others."""
    try:
        return getattr(plugin, method)(*args) or []
    except Exception:
        log.exception("plugin %s failed in %s", plugin.use_case, method)
        return []


def send(client: Go2rtc, events: list[Event]) -> None:
    for e in events:
        try:
            saved = client.send_event(e)
            log.info("event #%s %s/%s %s %.2f", saved.get("id"), e.camera, e.use_case, e.label, e.confidence)
        except Exception as err:
            log.warning("can't send event: %s", err)


class Online:
    """Caches which cameras go2rtc reports online, so offline cameras are skipped."""

    def __init__(self, client: Go2rtc, refresh: float = 5.0):
        self.client = client
        self.refresh = refresh
        self.checked = 0.0
        self.state: dict[str, bool] = {}
        self.lock = threading.Lock()

    def __call__(self, camera: str) -> bool:
        with self.lock:
            if time.monotonic() - self.checked > self.refresh:
                self.checked = time.monotonic()
                try:
                    self.state = {c["name"]: c["online"] for c in self.client.status()}
                except Exception as err:
                    log.warning("can't read camera status: %s", err)
            return self.state.get(camera, False)


def video_loop(client: Go2rtc, camera: str, plugins: list[Plugin], online: Online, stop: threading.Event) -> None:
    tick = min(p.interval for p in plugins)
    last_run = {id(p): 0.0 for p in plugins}
    while not stop.is_set():
        if not online(camera):
            stop.wait(online.refresh)
            continue
        started = time.monotonic()
        due = [p for p in plugins if started - last_run[id(p)] >= p.interval]
        if due:
            try:
                frame = client.frame(camera)
            except Exception as err:
                log.debug("%s: no frame (%s)", camera, err)
                frame = None
            if frame is not None:
                ts = datetime.now(timezone.utc)
                for p in due:
                    last_run[id(p)] = started
                    send(client, run_plugin(p, "on_frame", camera, frame, ts))
        stop.wait(max(0.0, tick - (time.monotonic() - started)))


def audio_loop(client: Go2rtc, camera: str, plugins: list[Plugin], online: Online, stop: threading.Event) -> None:
    while not stop.is_set():
        if not online(camera):
            stop.wait(online.refresh)
            continue
        try:
            for samples in audio_chunks(client.rtsp_url(camera)):
                if stop.is_set():
                    return
                ts = datetime.now(timezone.utc)
                for p in plugins:
                    send(client, run_plugin(p, "on_audio", camera, samples, SAMPLE_RATE, ts))
        except Exception as err:
            log.warning("%s: audio stopped (%s)", camera, err)
        stop.wait(5)  # camera offline or stream ended: retry


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CCTV prototype AI worker")
    parser.add_argument("--url", default="http://localhost:1984", help="go2rtc API address")
    parser.add_argument("--plugins", default="", help="comma-separated slot names (default: all enabled)")
    parser.add_argument("--cameras", default="", help="comma-separated cameras (default: all from go2rtc)")
    parser.add_argument("--list", action="store_true", help="list plugin slots and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    available = discover()
    if args.list:
        for name, cls in sorted(available.items()):
            state = "enabled" if cls.enabled else "stub"
            print(f"{name:20} {state:8} needs={','.join(sorted(cls.needs))}")
        return

    if args.plugins:
        names = [n.strip() for n in args.plugins.split(",") if n.strip()]
        unknown = [n for n in names if n not in available]
        if unknown:
            parser.error(f"unknown plugins: {', '.join(unknown)} (see --list)")
    else:
        names = [n for n, cls in available.items() if cls.enabled]
    if not names:
        parser.error("no plugins enabled; pass --plugins or set enabled = True in a plugin")

    client = Go2rtc(args.url)
    if args.cameras:
        cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    else:
        cameras = [c["name"] for c in client.status()]
    if not cameras:
        parser.error("no cameras: list them under cctv.cameras in go2rtc.yaml")

    plugins = [available[n]() for n in names]
    for p in plugins:
        p.setup(cameras)
    video = [p for p in plugins if "video" in p.needs]
    audio = [p for p in plugins if "audio" in p.needs]
    log.info("plugins=%s cameras=%s", names, cameras)

    stop = threading.Event()
    online = Online(client)
    threads = []
    for cam in cameras:
        if video:
            threads.append(threading.Thread(target=video_loop, args=(client, cam, video, online, stop), daemon=True))
        if audio:
            threads.append(threading.Thread(target=audio_loop, args=(client, cam, audio, online, stop), daemon=True))
    for t in threads:
        t.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop.set()


if __name__ == "__main__":
    main()
