"""Unit tests for retailflow.counting (line geometry, band, direction, filters).

Test line: horizontal from (0, 100) to (200, 100); IN side is BELOW (y > 100),
so walking downwards = IN and walking upwards = OUT.
"""

import pandas as pd
import pytest

from retailflow.config import LineConfig
from retailflow.counting import (IN, OUT, CountingConfig, LineCounter, anchor_point,
                                 count_tracks, crosses_segment, signed_distance)

LINE = LineConfig(p1=(0, 100), p2=(200, 100), in_point=(100, 150), band_px=5)
SIMPLE = CountingConfig(use_band=False)
BAND = CountingConfig(use_band=True)


def feed(counter, track_id, ys, x=100.0, start_frame=0):
    """Walk one track vertically through the given y positions; return events."""
    return [e for f, y in enumerate(ys, start_frame) if (e := counter.update(track_id, f, (x, y)))]


# ------------------------------------------------------------ geometry
def test_signed_distance_sign_and_magnitude():
    a, b = (0, 100), (200, 100)
    assert signed_distance((50, 110), a, b) == pytest.approx(10)    # one side
    assert signed_distance((50, 90), a, b) == pytest.approx(-10)    # other side
    assert signed_distance((50, 100), a, b) == 0                    # on the line


def test_in_point_orients_the_sign():
    flipped = LineConfig(p1=(200, 100), p2=(0, 100), in_point=(100, 150))  # reversed endpoints
    assert LineCounter(LINE).distance((100, 120)) > 0
    assert LineCounter(flipped).distance((100, 120)) > 0  # still positive on IN side


def test_crosses_segment_inside_and_outside():
    a, b = (0, 100), (200, 100)
    assert crosses_segment((100, 90), (100, 110), a, b)
    assert not crosses_segment((250, 90), (250, 110), a, b)  # beyond the end


def test_anchor_points():
    assert anchor_point(10, 20, 30, 80, "feet") == (20, 80)
    assert anchor_point(10, 20, 30, 80, "centre") == (20, 50)


# ------------------------------------------------------------ direction
def test_walking_down_is_in_and_up_is_out():
    c = LineCounter(LINE, BAND)
    assert feed(c, 1, [80, 90, 110, 120]) == [IN]
    assert feed(c, 2, [120, 110, 90, 80]) == [OUT]
    assert c.counts == {IN: 1, OUT: 1}


def test_no_count_without_crossing():
    c = LineCounter(LINE, BAND)
    assert feed(c, 1, [50, 60, 70, 80]) == []


def test_walking_around_end_of_line_not_counted():
    c = LineCounter(LINE, BAND)
    assert feed(c, 1, [80, 120], x=300) == []


# ------------------------------------------------------------ band / hysteresis
JITTER = [80, 98, 102, 97, 103, 99, 101, 98]  # hovering on the line, never clearly crosses


def test_simple_line_counts_jitter_many_times():
    c = LineCounter(LINE, SIMPLE)
    assert len(feed(c, 1, JITTER)) > 1  # the baseline weakness


def test_band_ignores_jitter():
    c = LineCounter(LINE, BAND)
    assert feed(c, 1, JITTER) == []


def test_band_counts_once_after_real_crossing_with_jitter():
    c = LineCounter(LINE, BAND)
    assert feed(c, 1, JITTER + [110, 130]) == [IN]


def test_band_scale_widens_band():
    # y=108 is 8 px from the line: outside the 5 px band, inside a 2x (10 px) band.
    assert feed(LineCounter(LINE, BAND), 1, [80, 108]) == [IN]
    assert feed(LineCounter(LINE, CountingConfig(band_scale=2.0)), 1, [80, 108]) == []


def test_point_inside_band_does_not_start_side():
    c = LineCounter(LINE, BAND)
    # starts inside the band, then clearly IN: no previous clear side => no count
    assert feed(c, 1, [100, 103, 120]) == []


def test_worked_example_from_docs_04():
    """The exact table in docs/04_COUNTING_RELIABILITY.md section 2."""
    line = LineConfig(p1=(45, 190), p2=(265, 190), in_point=(150, 250), band_px=5)
    ys = [180, 189, 191, 188, 192, 197]
    simple = LineCounter(line, SIMPLE)
    assert feed(simple, 1, ys, x=150) == [IN, OUT, IN] and simple.counts == {IN: 2, OUT: 1}
    band = LineCounter(line, BAND)
    assert feed(band, 1, ys, x=150) == [IN] and band.counts == {IN: 1, OUT: 0}
    assert band.distance((150, 197)) == pytest.approx(7.0)


# ------------------------------------------------------------ filters
def test_min_track_age_blocks_young_tracks():
    c = LineCounter(LINE, CountingConfig(min_track_age=5))
    assert feed(c, 1, [80, 120]) == []                  # age 2 at crossing
    c2 = LineCounter(LINE, CountingConfig(min_track_age=5))
    assert feed(c2, 1, [70, 75, 80, 85, 90, 120]) == [IN]  # age 6 at crossing


def test_once_per_direction_stops_double_counts():
    ys = [80, 120, 80, 120, 80]  # down, up, down, up
    unlimited = LineCounter(LINE, BAND)
    assert feed(unlimited, 1, ys) == [IN, OUT, IN, OUT]
    limited = LineCounter(LINE, CountingConfig(once_per_direction=True))
    assert feed(limited, 1, ys) == [IN, OUT]


def test_separate_tracks_are_independent():
    c = LineCounter(LINE, CountingConfig(once_per_direction=True))
    feed(c, 1, [80, 120])
    feed(c, 2, [80, 120])
    assert c.counts[IN] == 2


# ------------------------------------------------------------ offline table counting
def test_count_tracks_from_table():
    rows = [  # frame, id, box whose feet go from y=80 to y=120 (IN)
        (0, 7, 90, 40, 110, 80),
        (1, 7, 90, 80, 110, 120),
    ]
    df = pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2"])
    ev = count_tracks(df, LINE, BAND)
    assert ev[["frame", "track_id", "direction"]].values.tolist() == [[1, 7, IN]]
    # Centre point of the same boxes: y=60 -> y=100 (inside band), so no count.
    assert count_tracks(df, LINE, CountingConfig(point="centre")).empty
