"""Live streams: the latest-frame reader skips whole frames when processing is behind."""

import time

import numpy as np
import pytest

from retailflow.heatmap import Heatmap
from retailflow.pipeline import live_lag_stats
from retailflow.sources import LatestFrameReader
from retailflow.zones import ZoneTracker


class FakeCamera:
    """cv2.VideoCapture stand-in delivering `n` frames at `fps` in real time."""

    def __init__(self, n, fps):
        self.n, self.dt, self.i = n, 1 / fps, 0

    def read(self):
        if self.i >= self.n:
            return False, None
        time.sleep(self.dt)
        self.i += 1
        return True, np.full((2, 2), self.i - 1, np.uint8)


def drain(reader, work_s):
    got = []
    while (f := reader.get()) is not None:
        got.append(f[0])
        assert f[1][0, 0] == f[0] % 256          # image belongs to that index
        time.sleep(work_s)
    return got


def test_fast_consumer_gets_every_frame():
    # 50 fps: 20 ms per frame leaves a loaded test machine plenty of time to take each one.
    r = LatestFrameReader(FakeCamera(40, 50), max_frames=None)
    got = drain(r, 0.0)
    assert got == list(range(40)) and r.skipped == 0


def test_slow_consumer_skips_whole_frames_and_keeps_true_index():
    r = LatestFrameReader(FakeCamera(60, 200), max_frames=None)
    got = drain(r, 0.02)                         # 4x slower than the camera
    assert got == sorted(set(got)) and got[-1] == 59      # increasing, ends at the newest frame
    assert 0 < len(got) < 60 and r.skipped == 60 - len(got)


def test_max_frames_stops_the_reader():
    r = LatestFrameReader(FakeCamera(100, 500), max_frames=25)
    got = drain(r, 0.0)
    assert got[-1] == 24 and r.received == 25


def test_skipping_frames_is_not_keeping_up():
    fps = 25
    idx = list(range(0, fps * 60, 2))            # half of the frames processed, lag stays flat
    lags = [0.05] * len(idx)
    s = live_lag_stats(lags, fps, fps * 60, received=fps * 60, skipped=fps * 30, indices=idx)
    assert s["lag_slope"] == pytest.approx(0, abs=1e-6)
    assert s["processed_frac"] == 0.5 and not s["keeps_up"]
    assert len(s["lag_per_s"]) == 60             # still one value per second of video
    ok = live_lag_stats([0.05] * fps * 60, fps, fps * 60, received=fps * 60, skipped=10)
    assert ok["keeps_up"]                        # 10 of 1500 skipped (<1%) is fine


def test_zone_debounce_counts_skipped_frames():
    from retailflow.config import Zone
    zone = Zone("z", [(0, 0), (10, 0), (10, 10), (0, 10)])
    every = ZoneTracker([zone], fps=10)          # default 2 s to enter = 20 frames
    skip = ZoneTracker([zone], fps=10)
    entered_every = [f for f in range(30) if every.update(f, {1: (5, 5)})]
    entered_skip = [f for f in range(0, 30, 3) if skip.update(f, {1: (5, 5)}, step=3)]
    assert entered_every[0] == 19 and entered_skip[0] in (18, 21)   # same 2 s, not 6 s


def test_heatmap_weight_per_step():
    h = Heatmap((100, 100), fps=10, cell_px=10)
    h.add([(5, 5)], frames=3)
    assert h.grid.sum() == pytest.approx(0.3)
