"""Merge repeated reads of the same vehicle into one plate (Frigate's
_get_cluster_rep, ported from frigate/data_processing/common/license_plate/
mixin.py, MIT licence, Copyright (c) Blake Burgess, Scott Hasselman).

Reads are clustered greedily: each read joins the first cluster whose average
Jaro-Winkler similarity to the cluster's plates is >= the threshold (0.85).
The biggest cluster wins (ties: higher confidence), and its most confident
read is the plate. Pure logic, tested with fake reads.
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz.distance import JaroWinkler


@dataclass
class PlateRead:
    plate: str
    conf: float  # average character confidence 0..1
    area: int = 0


def cluster_reads(reads: list[PlateRead], threshold: float = 0.85) -> list[list[PlateRead]]:
    clusters: list[list[PlateRead]] = []
    for read in reads:
        for cluster in clusters:
            sims = [JaroWinkler.similarity(read.plate, other.plate) for other in cluster]
            if sum(sims) / len(sims) >= threshold:
                cluster.append(read)
                break
        else:
            clusters.append([read])
    return clusters


def best_plate(reads: list[PlateRead], threshold: float = 0.85) -> tuple[str, float, int] | None:
    """(plate, confidence, agreeing reads) of the best cluster, or None."""
    if not reads:
        return None
    clusters = cluster_reads(reads, threshold)
    best = max(clusters, key=lambda c: (len(c), max(r.conf for r in c)))
    rep = max(best, key=lambda r: r.conf)
    return rep.plate, rep.conf, len(best)
