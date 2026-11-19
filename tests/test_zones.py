"""Tests for retailflow.zones (polygon test, visit debouncing, dwell, occupancy)."""

import numpy as np
import pandas as pd
import pytest

from retailflow.config import Zone
from retailflow.zones import ZoneConfig, ZoneTracker, points_in_polygon, zone_analysis

SQUARE = Zone("box", ((0, 0), (10, 0), (10, 10), (0, 10)))
IN, OUT = (5.0, 5.0), (50.0, 50.0)
CFG = ZoneConfig(enter_s=0.3, exit_s=0.4)   # 3 and 4 frames at 10 fps


def test_config_converts_seconds_to_frames():
    assert ZoneConfig(2.0, 2.0).frames(25) == (50, 50)
    assert ZoneConfig(2.0, 2.0).frames(30) == (60, 60)
    assert ZoneConfig(0.01, 0.01).frames(25) == (1, 1)


def walk(positions, fps=10.0, cfg=CFG, tid=1):
    """Feed one track through positions (None = not detected); return the tracker."""
    zt = ZoneTracker([SQUARE], fps, cfg)
    for f, p in enumerate(positions):
        zt.update(f, {} if p is None else {tid: p})
    zt.finish()
    return zt


def test_points_in_polygon():
    pts = np.array([[5, 5], [15, 5], [-1, 5], [9.9, 0.1]])
    assert points_in_polygon(pts, SQUARE.polygon).tolist() == [True, False, False, True]
    tri = ((0, 0), (10, 0), (0, 10))
    assert points_in_polygon(np.array([[2, 2], [8, 8]]), tri).tolist() == [True, False]


def test_visit_dwell_is_backdated_and_ends_at_last_inside_frame():
    # frames 0-1 outside, 2-11 inside (10 frames), then outside
    zt = walk([OUT, OUT] + [IN] * 10 + [OUT] * 6)
    assert len(zt.visits) == 1
    v = zt.visits[0]
    assert (v.start_frame, v.end_frame) == (2, 11)
    assert v.dwell_s == pytest.approx(1.0)   # 10 frames at 10 fps
    assert not v.truncated


def test_short_touch_is_not_a_visit():
    zt = walk([OUT, IN, IN, OUT, OUT, OUT, OUT, OUT])  # 2 frames < enter_frames=3
    assert zt.visits == []


def test_jitter_on_edge_gives_one_visit():
    # inside, briefly outside for 2 frames (< exit_frames=4), inside again
    zt = walk([IN] * 5 + [OUT] * 2 + [IN] * 5 + [OUT] * 5)
    assert len(zt.visits) == 1 and zt.visits[0].dwell_s == pytest.approx(1.2)


def test_missed_detections_bridge_the_gap():
    zt = walk([IN] * 5 + [None] * 3 + [IN] * 5 + [None] * 5)
    assert len(zt.visits) == 1


def test_visit_open_at_end_is_truncated():
    zt = walk([OUT] + [IN] * 6)
    assert len(zt.visits) == 1 and zt.visits[0].truncated and zt.visits[0].end_frame == 6


def test_events_and_occupancy():
    zt = ZoneTracker([SQUARE], 10.0, CFG)
    events = []
    for f in range(3):
        events += zt.update(f, {1: IN, 2: IN, 3: OUT})
    assert [e[0] for e in events] == ["enter", "enter"]
    assert zt.occupancy() == {"box": 2}


def test_zone_analysis_on_table_matches_online():
    rows = [(f, 7, 4, 0, 6, 5) for f in range(2, 12)]  # feet (5, 5) frames 2..11
    df = pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2"])
    visits, occ = zone_analysis(df, [SQUARE], fps=10.0, n_frames=20, cfg=CFG)
    assert visits.dwell_s.tolist() == [pytest.approx(1.0)]
    assert occ["box"].iloc[4] == 1 and occ["box"].iloc[0] == 0 and len(occ) == 20
