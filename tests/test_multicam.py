"""Tests for the on-screen clock reader (osd_clock.py) and two-camera fusion (multicam.py)."""

from pathlib import Path

import numpy as np
import pytest

from retailflow.multicam import fuse, load_site
from retailflow.osd_clock import SLOT_W, SLOTS, fit_start, fmt_clock, load_templates, read_clock

CLIP = Path("data/caviar/TwoEnterShop1cor/TwoEnterShop1cor.mpg")


def synthetic_frame(digits: list[int]) -> np.ndarray:
    """A frame whose clock slots contain the given digits, drawn from the templates."""
    tmpl = {}
    for d, g in load_templates():
        tmpl.setdefault(d, g)
    img = np.zeros((288, 384, 3), np.uint8)
    for x, d in zip(SLOTS, digits):
        # Real glyphs start 1-2 px inside their slot; flush-left drawing would let a
        # glyph touch the previous slot's window.
        img[0:8, x + 1:x + 1 + SLOT_W][tmpl[d]] = 255
    return img


def test_read_clock_on_synthetic_digits():
    img = synthetic_frame([1, 0, 0, 7, 3, 3, 9, 9, 7])           # 10:07:33.997
    assert read_clock(img, load_templates()) == pytest.approx(10 * 3600 + 7 * 60 + 33.997)


def test_unclear_glyph_is_rejected_not_guessed():
    img = synthetic_frame([1, 0, 0, 7, 3, 3, 9, 9, 7])
    img[0:8, SLOTS[3]:SLOTS[3] + SLOT_W] = 255                    # a solid block: no digit
    assert read_clock(img, load_templates()) is None


@pytest.mark.skipif(not CLIP.exists(), reason="CAVIAR data not downloaded")
def test_read_clock_on_real_frame():
    from retailflow.sources import FrameSource
    frame = next(iter(FrameSource(str(CLIP), max_frames=1)))
    assert fmt_clock(read_clock(frame.image, load_templates())) == "10:07:33.997"


def test_fit_start_ignores_repeats_and_misreads():
    clock = [100.0 + i * 0.04 for i in range(200)]
    clock[50] = clock[49]                       # repeated frame
    clock[51] = clock[52]                       # then a skipped one
    clock[120] = 580.0                          # misread digit
    clock[121] = None                           # unreadable
    r = fit_start(clock)
    assert r["start_s"] == pytest.approx(100.0) and r["outliers"] == 1 and r["readable"] == 199


def test_fuse_matches_same_direction_within_window():
    front = [(10.0, "in"), (20.0, "out"), (30.0, "in")]
    corr = [(10.6, "in"), (20.3, "in"), (45.0, "out")]          # 20.3 is the wrong direction
    f = fuse(front, corr, window_s=1.0)
    assert f["in"] == {"a": 2, "b": 2, "matched": 1, "union": 3, "intersection": 1}
    assert f["out"] == {"a": 1, "b": 1, "matched": 0, "union": 2, "intersection": 0}
    assert f["dt"] == [0.6]


def test_fuse_uses_each_event_once():
    f = fuse([(1.0, "in"), (1.2, "in")], [(1.1, "in")], window_s=1.0)
    assert f["in"]["matched"] == 1 and f["in"]["union"] == 2


def test_site_file_loads():
    s = load_site("scenes/sites/caviar_promod.json")
    cams = {c.camera_id: c for c in s.cameras}
    assert cams["caviar_front"].line is None and cams["caviar_cor"].line.p1 == (70.0, 130.0)
    assert s.match_window_s == 1.0
