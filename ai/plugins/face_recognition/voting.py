"""Per-track face voting: decide a person's name from many noisy attempts.

Port of Frigate's weighted_average (frigate/data_processing/real_time/face.py,
MIT licence, Copyright (c) Blake Burgess, Scott Hasselman, Bernd Prager).

Every recognition attempt on a person track stores (name, score, face area).
"unknown" attempts are ignored. Each name's score is a weighted average where
weight = min(area, 4000) x (score - unknown_score) x 10: bigger, clearer, more
confident faces count more, and attempts below the unknown level actively drag
the average down. A name is only reported when it has at least `min_faces`
votes and no other name has the same number of votes (a tie means we are not
sure). Accept when the weighted average reaches `recognition_threshold`.

This module is pure logic (no OpenCV), so it is unit-tested with fake attempts.
"""

from __future__ import annotations

UNKNOWN = "unknown"


def weighted_average(
    attempts: list[tuple[str, float, int]],
    unknown_score: float,
    min_faces: int,
    max_weight: int = 4000,
) -> tuple[str | None, float]:
    """Best (name, weighted score) over attempts, or (None, 0.0) if not sure.

    attempts: list of (name, score 0..1, face area in px).
    """
    counts: dict[str, int] = {}
    weighted_scores: dict[str, float] = {}
    total_weights: dict[str, float] = {}

    for name, score, face_area in attempts:
        if name == UNKNOWN:
            continue
        weight = min(face_area, max_weight) * (score - unknown_score) * 10
        counts[name] = counts.get(name, 0) + 1
        weighted_scores[name] = weighted_scores.get(name, 0.0) + score * weight
        total_weights[name] = total_weights.get(name, 0.0) + weight

    if not weighted_scores:
        return None, 0.0

    best_name = max(weighted_scores, key=lambda k: weighted_scores[k])

    if counts[best_name] < min_faces:
        return None, 0.0

    # another name with the same number of votes: not confident
    for name, count in counts.items():
        if name != best_name and count == counts[best_name]:
            return None, 0.0

    if total_weights[best_name] <= 0:
        return None, 0.0

    return best_name, weighted_scores[best_name] / total_weights[best_name]


class FaceVote:
    """Voting state for one person track.

    Limits follow Frigate: at most 12 attempts in total, or 6 once a name was
    recognised (the cost should drop to zero for people we already know).
    """

    def __init__(
        self,
        unknown_score: float = 0.8,
        recognition_threshold: float = 0.8,
        min_faces: int = 3,
        max_attempts: int = 12,
        max_attempts_recognised: int = 6,
    ) -> None:
        self.unknown_score = unknown_score
        self.recognition_threshold = recognition_threshold
        self.min_faces = min_faces
        self.max_attempts = max_attempts
        self.max_attempts_recognised = max_attempts_recognised
        self.attempts: list[tuple[str, float, int]] = []
        self.accepted: str | None = None  # name once the threshold was reached
        self.best_area = 0  # of the best face attempt, for the snapshot
        self.best_score = 0.0

    def add(self, name: str | None, score: float, face_area: int) -> None:
        """Record one attempt. name None means unknown (below the match level)."""
        if name is None:
            name = UNKNOWN
        self.attempts.append((name, score, face_area))
        if score > self.best_score:  # track the best attempt, for the snapshot
            self.best_score = score
            self.best_area = face_area

    def should_attempt(self) -> bool:
        """False when the track is decided or the attempt limits are used up."""
        if self.accepted is not None:
            return False
        limit = self.max_attempts
        if self.reached_threshold():
            limit = min(limit, self.max_attempts_recognised)
        return len(self.attempts) < limit

    def reached_threshold(self) -> bool:
        name, score = self.decision()
        return name is not None and score >= self.recognition_threshold

    def decision(self) -> tuple[str | None, float]:
        """Current best (name, weighted score); (None, 0.0) while unsure."""
        name, score = weighted_average(self.attempts, self.unknown_score, self.min_faces)
        if name is not None and score >= self.recognition_threshold:
            self.accepted = name
        return name, score

    def exhausted(self) -> bool:
        """True when the track ran out of attempts without being recognised."""
        return self.accepted is None and not self.should_attempt()
