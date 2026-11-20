"""Tests for heatmap.py, staff.py and privacy.py."""

import numpy as np
import pandas as pd
import pytest

from retailflow.config import Homography, StaffRule, Zone
from retailflow.heatmap import Heatmap, heatmap_from_table, median_background, similarity
from retailflow.privacy import blur_heads, head_box
from retailflow.staff import StaffFilter, staff_tracks

SHOP = Zone("shop", ((0, 0), (10, 0), (10, 10), (0, 10)))


# ------------------------------------------------------------ heatmap
def test_heatmap_accumulates_person_seconds():
    hm = Heatmap((40, 20), fps=10, cell_px=4)
    for _ in range(10):                      # one person, 10 frames = 1 s
        hm.add([(5, 5)])
    hm.add([(100, 100)])                     # off the frame: ignored
    assert hm.grid.shape == (5, 10)
    assert hm.grid[1, 1] == pytest.approx(1.0) and hm.grid.sum() == pytest.approx(1.0)


def test_heatmap_from_table_uses_feet_point():
    df = pd.DataFrame({"x1": [0.0], "x2": [8.0], "y1": [0.0], "y2": [18.0]})
    hm = heatmap_from_table(df, (40, 20), fps=1, cell_px=4)
    assert hm.grid[4, 1] == 1.0              # feet (4, 18) -> row 4, col 1


def test_homography_maps_to_floor_cells():
    # image square 0..100 px maps to a 0..5 m floor square; 1 m cells
    h = Homography(((0, 0), (100, 0), (100, 100), (0, 100)), ((0, 0), (5, 0), (5, 5), (0, 5)))
    hm = Heatmap((100, 100), fps=1, homography=h, floor_cell=1.0)
    hm.add([(50, 90)])                       # floor (2.5, 4.5) m
    assert hm.grid[4, 2] == 1.0


def test_similarity():
    a = np.zeros((10, 10)); a[2, 2] = 5; a[7, 7] = 1
    s = similarity(a, a)
    assert s["pearson"] == 1.0 and s["hotspot_iou"] == 1.0 and s["mass_ratio"] == 1.0
    assert similarity(a, a.T[::-1])["pearson"] < 0.5


def test_overlay_only_tints_visited_area():
    from retailflow.heatmap import blend_overlay, overlay_colors
    hm = Heatmap((80, 40), fps=1, cell_px=4)
    for _ in range(5):
        hm.add([(10, 10)])
    col, alpha = overlay_colors(hm.grid, (80, 40), sigma_cells=1.0)
    assert alpha.shape == (40, 80) and alpha[10, 10] > 0.3 and alpha[35, 75] == 0
    img = np.full((40, 80, 3), 100, np.uint8)
    blend_overlay(img, col, alpha)
    assert (img[35, 75] == 100).all() and not (img[10, 10] == 100).all()


def test_median_background_removes_moving_person():
    frames = [np.full((4, 4, 3), 100, np.uint8) for _ in range(5)]
    frames[0][1, 1] = 255                    # a "person" in one frame only
    assert median_background(frames)[1, 1, 0] == 100


# ------------------------------------------------------------ staff
def test_staff_filter_flags_after_min_seconds():
    sf = StaffFilter(StaffRule(("shop",), min_seconds=2), [SHOP], fps=5)
    flagged = []
    for _ in range(12):
        flagged += sf.update({1: (5, 5), 2: (50, 50)})   # 1 inside, 2 outside
    assert flagged == [1] and sf.is_staff(1) and not sf.is_staff(2)


def test_staff_filter_without_rule_does_nothing():
    sf = StaffFilter(None, [SHOP], fps=5)
    assert sf.update({1: (5, 5)}) == [] and not sf.is_staff(1)


def test_staff_tracks_offline():
    rows = [(f, 1, 5, 5) for f in range(10)] + [(f, 2, 5, 5) for f in range(3)]
    df = pd.DataFrame(rows, columns=["frame", "track_id", "feet_x", "feet_y"])
    assert staff_tracks(df, StaffRule(("shop",), 1.0), [SHOP], fps=5) == {1}


# ------------------------------------------------------------ static objects
def _track(tid, xs, ys, h=50):
    return pd.DataFrame({"frame": range(len(xs)), "track_id": tid,
                         "x1": np.array(xs) - 10, "x2": np.array(xs) + 10,
                         "y1": np.array(ys) - h, "y2": np.array(ys)})


def test_mannequin_is_static_but_walker_and_waiter_are_not():
    fps, n = 5, 60                                             # 12 s
    rng = np.random.default_rng(1)
    mannequin = _track(1, 100 + rng.normal(0, 2, n), 200 + rng.normal(0, 2, n))   # jitter only
    walker = _track(2, np.linspace(0, 300, n), np.full(n, 200))
    # waits 10 s in one spot, but walked there first: whole-life range is large
    waiter = _track(3, np.r_[np.linspace(0, 100, 10), np.full(n - 10, 100)], np.full(n, 200))
    short = _track(4, np.full(20, 50.0), np.full(20, 200.0))   # static but only 4 s
    df = pd.concat([mannequin, walker, waiter, short])
    from retailflow.static_filter import static_tracks
    assert static_tracks(df, fps) == {1}


# ------------------------------------------------------------ privacy
def test_head_box_is_top_of_person_and_clipped():
    x1, y1, x2, y2 = head_box(10, 0, 30, 100, 200, 200)
    assert (y1, y2) == (0, 22) and x1 < 10 and x2 > 30


def test_blur_heads_changes_only_head_region():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    before = img.copy()
    blur_heads(img, [(40, 20, 60, 80)])
    hx1, hy1, hx2, hy2 = head_box(40, 20, 60, 80, 100, 100)
    assert img[hy1:hy2, hx1:hx2].std() < before[hy1:hy2, hx1:hx2].std()   # detail removed
    assert np.array_equal(img[hy2:], before[hy2:])                        # body untouched
