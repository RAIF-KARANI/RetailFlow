"""Read the on-screen clock burned into CAVIAR frames ("10:07:33.997 20-01-2004").

Why: the two CAVIAR views of the same scene (corridor "cor" and shop "front") were
recorded at the same time but do NOT start at the same moment, and the dataset
says to use the on-screen time code to align them. Reading it gives every frame a
real recording time, so both cameras can be put on one timeline.

How (no OCR library needed): the clock is a fixed bitmap font at fixed positions,
8 pixels tall. For each of the 9 time digits (HH MM SS mmm) we cut out its slot,
binarise it (white text on black), trim it to the lit pixels and compare it with
digit templates. The templates were made once: the distinct glyph shapes seen
in the frames were labelled by eye and saved in retailflow/osd_digits.json
(31 shapes: a digit can render 1 px narrower or wider depending on its slot).
First 25 came from 4 clips; a check over all 52 clips found 6 more, after a
missing '0' variant had been misread as '8' (docs/TROUBLESHOOTING.md). A frame whose glyphs do not all match a template
closely is returned as None instead of guessing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROWS = slice(0, 8)                       # glyph rows
# Left edge of each digit slot (pixels) and slot width: H H : M M : S S . m m m
SLOTS = [3, 10, 27, 36, 54, 62, 80, 88, 97]
SLOT_W = 8
THRESH = 170                             # grey level separating text from the black band
TEMPLATE_FILE = Path(__file__).with_name("osd_digits.json")
MAX_MISMATCH = 2                         # pixels allowed to differ (MPEG noise)
MIN_MARGIN = 2                           # best digit must beat the next digit by this many pixels


def glyph(gray: np.ndarray, x: int) -> np.ndarray:
    """Binary glyph in a slot, trimmed to its lit columns and left-aligned in an 8x8 box."""
    cell = gray[ROWS, x:x + SLOT_W] > THRESH
    cols = np.where(cell.any(axis=0))[0]
    out = np.zeros((8, SLOT_W), bool)
    if len(cols):
        part = cell[:, cols[0]:cols[-1] + 1]
        out[:, :part.shape[1]] = part
    return out


def key(g: np.ndarray) -> str:
    """Compact text form of a glyph, used as a dictionary key."""
    return "".join("1" if v else "0" for v in g.ravel())


def load_templates(path: Path = TEMPLATE_FILE) -> list[tuple[int, np.ndarray]]:
    """[(digit, glyph)]. Several glyphs per digit: the same digit renders 1 px
    narrower or wider depending on its slot."""
    raw = json.loads(path.read_text())
    return [(int(d), np.array([c == "1" for c in k]).reshape(8, SLOT_W)) for d, k in raw]


def read_clock(image: np.ndarray, templates: list[tuple[int, np.ndarray]]) -> float | None:
    """Seconds since midnight shown in the frame's clock, or None if it cannot be read."""
    gray = image if image.ndim == 2 else image[..., 1]   # green channel is enough for white text
    digits = []
    for x in SLOTS:
        g = glyph(gray, x)
        err_by_digit: dict[int, int] = {}
        for d, t in templates:
            err_by_digit[d] = min(err_by_digit.get(d, 99), int((g != t).sum()))
        ranked = sorted(err_by_digit.items(), key=lambda kv: kv[1])
        (best, err), (_, second) = ranked[0], ranked[1]
        # Accept only a close AND unambiguous match. A looser rule once read an unseen
        # '0' variant as '8' (3 px off), shifting the clock by 8 minutes.
        if err > MAX_MISMATCH or second - err < MIN_MARGIN:
            return None
        digits.append(best)
    h = digits[0] * 10 + digits[1]
    m = digits[2] * 10 + digits[3]
    s = digits[4] * 10 + digits[5]
    ms = digits[6] * 100 + digits[7] * 10 + digits[8]
    if h > 23 or m > 59 or s > 59:
        return None
    return h * 3600 + m * 60 + s + ms / 1000


def fit_start(clock: list[float | None], fps: float = 25.0, tol_s: float = 0.2) -> dict:
    """Robust clock time of frame 0 for one clip.

    The recording sometimes repeats a frame and then skips one (clock steps of 0
    and 80 ms instead of 40), and a digit can be misread now and then. So the
    start is the MEDIAN of (clock_i - i / fps) over all readable frames, and the
    frames that disagree with it by more than `tol_s` are counted as outliers.
    """
    i = np.array([k for k, c in enumerate(clock) if c is not None])
    c = np.array([v for v in clock if v is not None])
    if len(c) == 0:
        return {"start_s": None, "readable": 0, "outliers": 0, "frames": len(clock)}
    est = c - i / fps
    start = float(np.median(est))
    resid = est - start
    return {"start_s": round(start, 3), "readable": int(len(c)), "frames": len(clock),
            "outliers": int((np.abs(resid) > tol_s).sum()),
            "resid_p95_s": round(float(np.percentile(np.abs(resid), 95)), 3)}


def fmt_clock(seconds: float) -> str:
    """Seconds since midnight -> 'HH:MM:SS.mmm'."""
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"
