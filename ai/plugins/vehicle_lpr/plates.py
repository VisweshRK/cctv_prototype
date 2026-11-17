"""Plate text helpers: cleaning, Indian format check, OCR repairs, box shaping.

Pure logic (no model), tested with fake reads in ai/tests/test_vehicle_lpr.py.
The pipeline keeps only plate-shaped text boxes, merges the two lines of bike
plates, cleans the OCR text, fixes typical OCR confusions by position, and
keeps the read only if it looks like a real Indian plate.
"""

from __future__ import annotations

import re

VEHICLE_LABELS = {"car", "motorcycle", "bus", "truck"}

# normal: TN09AB1234 (also 3-letter series TN09ABC1234 and old 1-digit district)
INDIAN_PLATE_RE = re.compile(r"[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}")
# Bharat series: 22BH1234AB
BHARAT_PLATE_RE = re.compile(r"[0-9]{2}BH[0-9]{4}[A-Z]{1,2}")

# OCR confusions fixed by position: in a digit slot these letters mean digits,
# in a letter slot these digits mean letters.
DIGIT_FIXES = {"O": "0", "Q": "0", "D": "0", "I": "1", "S": "5", "B": "8", "Z": "2", "G": "6"}
LETTER_FIXES = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}

MIN_PLATE_LENGTH = 4  # Frigate's min_plate_length


def text_in_box(
    lines: list[tuple[tuple[float, float, float, float], str, float]],
    box: tuple[float, float, float, float],
    margin: float = 1.0,
) -> tuple[str, float, int, bool] | None:
    """Assemble the plate text of the OCR lines that sit inside `box`.

    Lines are joined top-to-bottom (bike plates stack two lines; joining in
    any other order scrambles the plate). Returns (text, mean confidence,
    line count, is_two_line), or None when no line falls inside.
    """
    x, y, w, h = box
    members = [
        (lx, ly, text, conf)
        for (lx, ly, lw, lh), text, conf in lines
        if lx >= x - margin and ly >= y - margin
        and lx + lw <= x + w + margin and ly + lh <= y + h + margin
    ]
    if not members:
        return None
    members.sort(key=lambda m: m[1])  # reading order: top line first
    text = "".join(m[2] for m in members)
    conf = sum(m[3] for m in members) / len(members)
    return text, conf, len(members), len(members) >= 2


def clean_text(text: str) -> str:
    """Uppercase and keep only A-Z and 0-9."""
    return "".join(c for c in text.upper() if c.isalnum())


def is_valid(plate: str, min_length: int = MIN_PLATE_LENGTH) -> bool:
    """True for a cleaned plate that matches an Indian format."""
    if len(plate) < min_length:
        return False
    return bool(INDIAN_PLATE_RE.fullmatch(plate) or BHARAT_PLATE_RE.fullmatch(plate))


def _shapes(length: int) -> list[str]:
    """Templates of one plate format: L = letter, D = digit, others literal."""
    shapes = []
    for district in (1, 2):
        for series in range(4):  # 0..3 letters
            shapes.append("LL" + "D" * district + "L" * series + "DDDD")
    shapes += ["DDBH" + "DDDD" + "L", "DDBH" + "DDDD" + "LL"]
    return [s for s in shapes if len(s) == length]


def repair(text: str) -> str | None:
    """Fix OCR confusions by position so the text fits an Indian format.

    Tries every format shape with the text's length, applies DIGIT_FIXES /
    LETTER_FIXES per position, and returns the valid candidate with the fewest
    changes. None when nothing fits.
    """
    best: tuple[int, str] | None = None
    for shape in _shapes(len(text)):
        chars = list(text)
        changes = 0
        ok = True
        for i, kind in enumerate(shape):
            c = chars[i]
            if kind == "D":
                if not c.isdigit():
                    if c in DIGIT_FIXES:
                        chars[i], changes = DIGIT_FIXES[c], changes + 1
                    else:
                        ok = False
                        break
            elif kind == "L":
                if not c.isalpha():
                    if c in LETTER_FIXES:
                        chars[i], changes = LETTER_FIXES[c], changes + 1
                    else:
                        ok = False
                        break
            elif c != kind:  # literal, e.g. the BH in the Bharat series
                if c in LETTER_FIXES and LETTER_FIXES[c] == kind:
                    chars[i], changes = kind, changes + 1
                else:
                    ok = False
                    break
        if ok:
            candidate = "".join(chars)
            if is_valid(candidate):
                if best is None or changes < best[0]:
                    best = (changes, candidate)
    return best[1] if best else None


def normalize(text: str, min_length: int = MIN_PLATE_LENGTH) -> str | None:
    """Clean an OCR text and return it only if it is a valid Indian plate."""
    plate = clean_text(text)
    if len(plate) < min_length:
        return None
    if is_valid(plate, min_length):
        return plate
    return repair(plate)


def is_plate_shaped(w: float, h: float, min_area: float = 1000, two_line: bool = False) -> bool:
    """Text-box proportions and size (plan: car plates 2:1..5:1, bikes 1:1..2:1).

    Text boxes run wider than the plate itself (characters fill the width,
    the OCR box hugs the glyphs), so the single-line cap is 10:1, not 5:1;
    the lower bound kills square signs and stickers. The Indian-format check
    downstream is what really keeps non-plate text out.
    """
    if w <= 0 or h <= 0 or w * h < min_area:
        return False
    ratio = w / h
    return 1.0 <= ratio <= 2.5 if two_line else 1.8 <= ratio <= 10.0


def merge_two_line(boxes: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    """Stack the two lines of bike plates into one box each.

    Input boxes are (x, y, w, h) text lines. Lines with similar height,
    overlapping x-range and a small vertical gap become one box. Car plates
    (one line) pass through unchanged.
    """
    merged = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                ax, ay, aw, ah = merged[i]
                bx, by, bw, bh = merged[j]
                if not (0.6 < ah / bh < 1.6):
                    continue
                overlap = min(ax + aw, bx + bw) - max(ax, bx)
                if overlap < 0.5 * min(aw, bw):
                    continue
                gap = abs(ay - by)
                if gap > 1.5 * max(ah, bh):
                    continue
                x, y = min(ax, bx), min(ay, by)
                w = max(ax + aw, bx + bw) - x
                h = max(ay + ah, by + bh) - y
                merged[i] = [x, y, w, h]
                merged.pop(j)
                changed = True
                break
            if changed:
                break
    return [tuple(b) for b in merged]
