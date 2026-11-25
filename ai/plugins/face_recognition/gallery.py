"""Build one mean embedding per known person (Frigate's build_class_mean).

Pure numpy so it is testable without model files. The gallery itself is built
by build_gallery.py: YuNet -> alignCrop -> SFace for every photo in
faces/<name>/*.jpg, then the functions here clean them up:

1. Drop photos whose cosine similarity to the person's running mean is far
   below the rest (blurry photo, wrong face, two faces in one frame).
2. Take a 15% trimmed mean per dimension: sort each dimension, cut 15% from
   each end, average the middle. One weird dimension value cannot shift the
   mean.
3. L2-normalise the result, so cosine similarity is a plain dot product.
"""

from __future__ import annotations

import numpy as np


def trim_mean_rows(embeddings: np.ndarray, trim: float = 0.15) -> np.ndarray:
    """Per-dimension 15% trimmed mean of rows (n, d) -> (d,)."""
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(f"expected a non-empty (n, d) array, got {embeddings.shape}")
    n = embeddings.shape[0]
    cut = int(n * trim)
    if 2 * cut >= n:
        cut = max(0, (n - 1) // 2)  # keep at least one middle value
    ordered = np.sort(embeddings, axis=0)
    if 2 * cut >= n:
        return ordered[cut].copy() if n % 2 == 1 else (ordered[cut - 1] + ordered[cut]) / 2
    return ordered[cut : n - cut].mean(axis=0)


def drop_outliers(embeddings: np.ndarray, min_similarity: float, max_drops: int = 3) -> np.ndarray:
    """Drop rows far below the group mean, one worst row per round.

    A photo whose embedding similarity to the running mean is below
    `min_similarity` (SFace match threshold 0.363) while most other photos
    agree with each other is likely a wrong or unreadable face.
    """
    kept = embeddings.astype(np.float32)
    for _ in range(max_drops):
        if kept.shape[0] <= 3:  # keep a usable minimum
            break
        mean = kept.mean(axis=0)
        mean /= np.linalg.norm(mean) + 1e-9
        sims = kept @ mean
        worst = int(np.argmin(sims))
        others = np.delete(sims, worst)
        if sims[worst] >= min_similarity or sims[worst] > others.mean() - 2 * others.std():
            break  # the worst photo is not far below the rest: stop
        kept = np.delete(kept, worst, axis=0)
    return kept


def build_class_mean(embeddings: np.ndarray, trim: float = 0.15, min_similarity: float = 0.363) -> np.ndarray:
    """Cleaned, L2-normalised mean embedding for one person."""
    kept = drop_outliers(embeddings, min_similarity)
    mean = trim_mean_rows(kept, trim)
    norm = np.linalg.norm(mean)
    if norm < 1e-9:
        raise ValueError("mean embedding collapsed to zero")
    return (mean / norm).astype(np.float32)
