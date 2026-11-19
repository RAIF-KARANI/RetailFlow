"""Unit tests for backend/app/analytics.py (pure Python, no database needed)."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "backend_analytics", Path(__file__).parents[1] / "backend" / "app" / "analytics.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)


def test_counts_series_buckets():
    rows = A.counts_series([(1.0, "in"), (9.9, "in"), (10.0, "out"), (25.0, "in")], bucket_s=10, end_s=25)
    assert rows == [{"t": 0, "in": 2, "out": 0}, {"t": 10, "in": 0, "out": 1}, {"t": 20, "in": 1, "out": 0}]


def test_pair_visits_closed_and_open():
    ev = [
        {"type": "zone_enter", "track_id": 1, "zone": "a", "ts": 2.0, "value": None},
        {"type": "zone_enter", "track_id": 2, "zone": "a", "ts": 3.0, "value": None},
        {"type": "zone_exit", "track_id": 1, "zone": "a", "ts": 6.0, "value": 4.04},
    ]
    v = A.pair_visits(ev, end_s=10.0)
    closed = [x for x in v if not x["open"]][0]
    still = [x for x in v if x["open"]][0]
    assert (closed["start"], closed["end"], closed["dwell_s"]) == (2.0, 6.0, 4.04)
    assert (still["track_id"], still["end"], still["dwell_s"]) == (2, 10.0, 7.0)


def test_occupancy_is_time_weighted():
    visits = [{"zone": "a", "start": 0.0, "end": 5.0}, {"zone": "a", "start": 2.5, "end": 7.5}]
    rows = A.occupancy_series(visits, ["a"], bucket_s=5, end_s=9)
    # bucket 0-5: person 1 full (1.0) + person 2 half (0.5); bucket 5-10: person 2 half (0.5)
    assert [r["a"] for r in rows] == [pytest.approx(1.5), pytest.approx(0.5)]


def test_dwell_stats():
    visits = [
        {"zone": "a", "dwell_s": 2.0, "open": False},
        {"zone": "a", "dwell_s": 6.0, "open": False},
        {"zone": "a", "dwell_s": 1.0, "open": True},
    ]
    s = A.dwell_stats(visits, ["a", "b"])
    assert s[0] == {"zone": "a", "visits": 2, "mean_dwell_s": 4.0, "median_dwell_s": 4.0,
                    "max_dwell_s": 6.0, "occupancy_now": 1}
    assert s[1]["visits"] == 0 and s[1]["mean_dwell_s"] is None
