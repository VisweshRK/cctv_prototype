"""Test tampering on a video instead of a live camera.

Run from ai/:
    python -m plugins.tampering.video_test make  ../samples/dual.mp4   # make the test video
    python -m plugins.tampering.video_test check ../samples/dual.mp4   # run the plugin on it

`make` draws a fake dual-lens camera: two scenes stacked one above the other (like the real
1920x2160 camera), each with a burned-in clock like a real camera, people walking past, and
scripted tampering (SCRIPT below). `check` cuts each frame into its two lenses, runs the
plugin once per second of video and compares the events with what SCRIPT expects.

`check` also works on any other clip (`--split` off, `--expect` off) to see what fires.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np

W, H, FPS = 1920, 1080, 10  # one lens; the video is W x 2H
LENSES = ("cam1", "cam2")  # top, bottom

# (start s, end s, lens, what). Lens None = both. The video loops after LENGTH seconds.
SCRIPT = [
    (55, 57, "cam1", "walk_past"),  # someone walks right past the lens: 2 s, not tampering
    (90, 110, "cam2", "hand"),
    (130, 150, "cam1", "spray"),
    (170, 190, "cam2", "turned"),
    (210, 230, "cam1", "tape"),
    (250, 280, None, "night"),  # the camera switches to IR: not tampering
]
LENGTH = 300
EXPECT = {"hand": "covered", "tape": "covered", "spray": "blurred", "turned": "moved"}


# ---------------------------------------------------------------- scenes

def corridor(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = np.linspace(215, 175, h, dtype=np.uint8)[:, None, None] * np.array([1, 1.02, 1.05])[None, None]
    floor = int(h * 0.68)
    img[floor:] = (95, 110, 125)
    for x in range(0, w, 90):  # floor tiles
        cv2.line(img, (x, floor), (int(w / 2 + (x - w / 2) * 2.2), h), (70, 85, 100), 2)
    for y in (floor + 40, floor + 110, floor + 230):
        cv2.line(img, (0, y), (w, y), (70, 85, 100), 2)
    for x in range(120, w - 200, 520):  # doors with frames and handles
        cv2.rectangle(img, (x - 12, 180), (x + 232, floor), (60, 70, 80), -1)
        cv2.rectangle(img, (x, 195), (x + 220, floor), (40, 75, 120), -1)
        cv2.rectangle(img, (x + 25, 230), (x + 195, 420), (55, 95, 145), 3)
        cv2.circle(img, (x + 190, int(floor * 0.62)), 10, (180, 190, 200), -1)
    nb = (int(w * 0.42), 230)  # notice board covered in papers
    cv2.rectangle(img, nb, (nb[0] + 330, nb[1] + 230), (60, 110, 160), -1)
    for _ in range(26):
        x, y = nb[0] + int(rng.integers(10, 270)), nb[1] + int(rng.integers(10, 170))
        c = tuple(int(v) for v in rng.integers(150, 256, 3))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(30, 60)), y + int(rng.integers(35, 60))), c, -1)
        for k in range(3):
            cv2.line(img, (x + 5, y + 10 + 10 * k), (x + 25, y + 10 + 10 * k), (60, 60, 60), 1)
    for x in range(200, w, 480):  # ceiling lights
        cv2.ellipse(img, (x, 60), (110, 22), 0, 0, 360, (250, 252, 255), -1)
    for _ in range(40):  # a plant
        cx, cy = int(w * 0.9) + int(rng.integers(-60, 60)), floor - 120 + int(rng.integers(-110, 60))
        cv2.circle(img, (cx, cy), int(rng.integers(14, 30)), (40, int(rng.integers(110, 170)), 50), -1)
    cv2.rectangle(img, (int(w * 0.9) - 50, floor - 70), (int(w * 0.9) + 50, floor + 20), (40, 60, 120), -1)
    return texture(rng, img)


def gate(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    sky = int(h * 0.35)
    img[:sky] = np.linspace(235, 200, sky, dtype=np.uint8)[:, None, None] * np.array([1, 0.85, 0.65])[None, None]
    img[sky:int(h * 0.62)] = (120, 140, 170)  # compound wall, bricks
    for y in range(sky, int(h * 0.62), 28):
        cv2.line(img, (0, y), (w, y), (95, 110, 140), 2)
        off = 0 if (y // 28) % 2 else 40
        for x in range(off, w, 80):
            cv2.line(img, (x, y), (x, y + 28), (95, 110, 140), 2)
    img[int(h * 0.62):] = (105, 105, 100)  # road
    for x in range(0, w, 220):
        cv2.rectangle(img, (x, int(h * 0.82)), (x + 120, int(h * 0.84)), (220, 220, 220), -1)
    gx = int(w * 0.35)  # the gate
    cv2.rectangle(img, (gx, sky - 40), (gx + 560, int(h * 0.64)), (50, 50, 55), 6)
    for x in range(gx, gx + 560, 32):
        cv2.line(img, (x, sky - 40), (x, int(h * 0.64)), (50, 50, 55), 5)
    cv2.rectangle(img, (gx - 60, sky - 90), (gx, int(h * 0.64)), (150, 160, 175), -1)
    cv2.rectangle(img, (gx + 560, sky - 90), (gx + 620, int(h * 0.64)), (150, 160, 175), -1)
    car = (int(w * 0.72), int(h * 0.6))  # a parked car
    cv2.rectangle(img, car, (car[0] + 420, car[1] + 150), (30, 30, 150), -1)
    cv2.rectangle(img, (car[0] + 70, car[1] - 80), (car[0] + 330, car[1]), (30, 30, 150), -1)
    cv2.rectangle(img, (car[0] + 95, car[1] - 65), (car[0] + 305, car[1] - 5), (190, 170, 140), -1)
    for dx in (85, 335):
        cv2.circle(img, (car[0] + dx, car[1] + 150), 42, (25, 25, 25), -1)
    for _ in range(90):  # trees behind the wall
        cx, cy = int(rng.integers(0, w)), sky - int(rng.integers(0, 140))
        if gx - 80 < cx < gx + 640:
            continue
        cv2.circle(img, (cx, cy), int(rng.integers(20, 45)), (40, int(rng.integers(100, 160)), 45), -1)
    return texture(rng, img)


def texture(rng: np.random.Generator, img: np.ndarray) -> np.ndarray:
    """Fine fixed grain so flat areas aren't perfectly flat, like a real surface."""
    grain = cv2.GaussianBlur(rng.normal(0, 9, img.shape[:2]).astype(np.float32), (0, 0), 1.2)
    return np.clip(img + grain[:, :, None], 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- one lens, one frame

class Lens:
    def __init__(self, name: str, world: np.ndarray, seed: int) -> None:
        self.name, self.world = name, world  # world is wider/taller than the view, for "turned"
        rng = np.random.default_rng(seed)
        self.noise = [(rng.integers(0, 6, (H, W, 3), dtype=np.uint8), rng.integers(0, 6, (H, W, 3), dtype=np.uint8))
                      for _ in range(4)]
        self.x0, self.y0 = (world.shape[1] - W) // 2, (world.shape[0] - H) // 2

    def frame(self, t: float, n: int, what: str | None) -> np.ndarray:
        if what == "turned":
            x, y = self.x0 + int(W * 0.3), self.y0 - int(H * 0.15)
            img = self.world[y:y + H, x:x + W].copy()
            img = cv2.warpAffine(img, cv2.getRotationMatrix2D((W / 2, H / 2), 8, 1.0), (W, H),
                                 borderMode=cv2.BORDER_REFLECT)
        else:
            img = self.world[self.y0:self.y0 + H, self.x0:self.x0 + W].copy()
        self.people(img, t, what == "walk_past")

        if what == "hand":  # palm right on the lens: out of focus, lit by the camera's IR LEDs
            img[:] = (110, 135, 175)
            yy, xx = np.mgrid[0:H, 0:W]
            shade = 1.0 - 0.25 * (((xx - W * 0.4) / W) ** 2 + ((yy - H * 0.6) / H) ** 2)
            img = (img * shade[:, :, None]).astype(np.uint8)
        elif what == "tape":
            img[:] = (9, 9, 10)
        elif what == "spray":
            img = cv2.GaussianBlur(img, (0, 0), 14)
            img = cv2.addWeighted(img, 0.6, np.full_like(img, 225), 0.4, 0)
        elif what == "night":
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            g = cv2.convertScaleAbs(g, alpha=1.3, beta=-60)
            img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

        plus, minus = self.noise[n % len(self.noise)]  # sensor noise
        img = cv2.subtract(cv2.add(img, plus), minus)
        osd(img, t, self.name)  # the camera draws its clock on top of everything
        return img

    def people(self, img: np.ndarray, t: float, close: bool) -> None:
        if close:  # a person right in front of the camera fills half the view
            x = int((t % 2) / 2 * W * 1.4) - W // 3
            cv2.ellipse(img, (x, H // 2), (W // 4, H), 0, 0, 360, (50, 55, 65), -1)
            return
        for speed, phase, y, colour in ((140, 0, 0.72, (60, 60, 140)), (90, 7, 0.78, (130, 80, 40))):
            x = int(((t + phase) * speed) % (W + 400)) - 200
            top = int(H * y) - 260
            cv2.circle(img, (x, top), 32, (90, 120, 160), -1)  # head
            cv2.rectangle(img, (x - 45, top + 35), (x + 45, top + 190), colour, -1)  # body
            cv2.rectangle(img, (x - 38, top + 190), (x + 38, top + 300), (50, 50, 50), -1)  # legs


def osd(img: np.ndarray, t: float, name: str) -> None:
    stamp = (datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc) + timedelta(seconds=int(t))).strftime(
        "%Y-%m-%d %H:%M:%S")
    for text, org in ((stamp, (40, 70)), (name.upper(), (W - 260, H - 40))):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)


def happening(t: float, lens: str) -> str | None:
    for start, end, who, what in SCRIPT:
        if start <= t < end and who in (None, lens):
            return what
    return None


def make(path: str) -> None:
    rng = np.random.default_rng(3)
    lenses = [Lens("cam1", corridor(rng, int(W * 1.8), int(H * 1.5)), 11),
              Lens("cam2", gate(rng, int(W * 1.8), int(H * 1.5)), 12)]
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        sys.exit("ffmpeg not found on PATH")
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{2 * H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "26", "-g", str(2 * FPS), "-pix_fmt", "yuv420p", path]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for n in range(LENGTH * FPS):
        t = n / FPS
        frame = np.vstack([lens.frame(t, n, happening(t, lens.name)) for lens in lenses])
        enc.stdin.write(frame.tobytes())
        if n % (30 * FPS) == 0:
            print(f"{int(t):4d} s / {LENGTH} s", flush=True)
    enc.stdin.close()
    if enc.wait() != 0:
        sys.exit("ffmpeg failed")
    print(f"wrote {path}: {W}x{2 * H}, {LENGTH} s at {FPS} fps")


# ---------------------------------------------------------------- check

def check(path: str, split: bool, expect: bool, loops: int) -> int:
    from cctv_ai.loader import discover

    plugin = discover()["tampering"]()
    cameras = list(LENSES) if split else ["video"]
    plugin.setup(cameras)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"can't open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    start = datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)
    got = []
    for loop in range(loops):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if n % round(fps * plugin.interval) == 0:  # one check per second, like the worker
                t = n / fps
                views = np.vsplit(frame, 2) if split else [frame]
                for camera, view in zip(cameras, views):
                    for e in plugin.on_frame(camera, view, start + timedelta(seconds=t + loop * LENGTH)):
                        got.append((loop, t, e))
                        print(f"loop {loop + 1}  {t:6.1f} s  {e.camera}  {e.label:9s} {e.confidence:.2f}  "
                              f"{e.data.get('detail', '')}")
            n += 1
    if not got:
        print("no events")
    if not (split and expect):
        return 0
    return compare(got, loops)


def compare(got: list, loops: int) -> int:
    """Every scripted incident must give its event within 5 s and `restored` after it;
    nothing else may fire."""
    print()
    wanted = []
    for loop in range(loops):
        for start, end, lens, what in SCRIPT:
            if what in EXPECT:
                wanted.append((loop, start, lens, EXPECT[what]))
                wanted.append((loop, end, lens, "restored"))
    left = list(got)
    failed = 0
    for loop, t, lens, label in wanted:
        hit = next((g for g in left if g[0] == loop and g[2].camera == lens and g[2].label == label
                    and t <= g[1] <= t + 5), None)
        if hit:
            left.remove(hit)
            print(f"ok    loop {loop + 1}  {lens}  {label:9s} at {t} s -> {hit[1] - t:.0f} s later")
        else:
            failed += 1
            print(f"MISS  loop {loop + 1}  {lens}  {label:9s} expected from {t} s")
    for loop, t, e in left:
        failed += 1
        print(f"EXTRA loop {loop + 1}  {e.camera}  {e.label} at {t:.0f} s")
    print("\nall as expected" if not failed else f"\n{failed} problem(s)")
    return 1 if failed else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="write the fake dual-lens test video")
    m.add_argument("path")
    c = sub.add_parser("check", help="run the tampering plugin on a video")
    c.add_argument("path")
    c.add_argument("--no-split", dest="split", action="store_false", help="one camera, not two lenses")
    c.add_argument("--no-expect", dest="expect", action="store_false", help="just print the events")
    c.add_argument("--loops", type=int, default=2, help="play the video this many times (default 2)")
    a = p.parse_args()
    if a.cmd == "make":
        make(a.path)
    else:
        sys.exit(check(a.path, a.split, a.expect, a.loops))


if __name__ == "__main__":
    main()
