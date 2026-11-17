"""Build the known-person gallery: faces/<name>/*.jpg -> data/faces.npz.

Run from ai/ (models must be downloaded first):
    .venv/Scripts/python -m plugins.face_recognition.build_gallery

Every photo is embedded (YuNet finds the biggest face, SFace aligns and
embeds it), bad photos are dropped (gallery.drop_outliers), and each person
gets one 15% trimmed mean (gallery.build_class_mean). Re-run whenever the
photos change; data/ is git-ignored, so this never leaves your laptop.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from plugins.face_recognition.engine import FaceEngine

log = logging.getLogger("cctv_ai.face_recognition")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def collect_photos(gallery_dir: Path) -> dict[str, list[Path]]:
    people: dict[str, list[Path]] = {}
    if not gallery_dir.is_dir():
        return people
    for person in sorted(p for p in gallery_dir.iterdir() if p.is_dir()):
        photos = [p for p in person.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.is_file()]
        if photos:
            people[person.name] = photos
    return people


def embed_photo(engine: FaceEngine, path: Path) -> np.ndarray | None:
    """Biggest face in the photo, aligned and embedded; None when no face."""
    image = cv2_imread(path)
    if image is None:
        log.warning("cannot read %s", path)
        return None
    faces = engine.detect(image)
    if not faces:
        log.warning("no face found in %s", path)
        return None
    return engine.embed(image, faces[0])


def cv2_imread(path: Path) -> np.ndarray | None:
    # imread fails on non-ASCII Windows paths; read the bytes ourselves
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = repo_root()
    gallery_dir = Path(__file__).resolve().parent / "faces"
    people = collect_photos(gallery_dir)
    if not people:
        print(f"no photos: put them in {gallery_dir / '<name>'}/*.jpg", file=sys.stderr)
        return 1

    engine = FaceEngine(root / "ai" / "models")
    if not engine.load_models():
        print("models missing; run: python -m plugins.face_recognition.download_models", file=sys.stderr)
        return 1

    names: list[str] = []
    means: list[np.ndarray] = []
    for name, photos in people.items():
        embeddings = [e for p in photos if (e := embed_photo(engine, p)) is not None]
        if len(embeddings) < 3:
            print(f"{name}: only {len(embeddings)} usable photos of {len(photos)}, need at least 3", file=sys.stderr)
            continue
        try:
            from plugins.face_recognition.gallery import build_class_mean

            mean = build_class_mean(np.stack(embeddings))
        except ValueError as err:
            print(f"{name}: {err}", file=sys.stderr)
            continue
        names.append(name)
        means.append(mean)
        print(f"{name}: {len(embeddings)}/{len(photos)} photos -> mean embedding")

    if not names:
        print("no usable people", file=sys.stderr)
        return 1

    out = root / "data" / "faces.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, names=np.array(names), means=np.stack(means).astype(np.float32))
    print(f"wrote {out} with {len(names)} people")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
