"""Tests for the streaming helpers: lag statistics, FFmpeg command, stream naming."""

import pytest

from retailflow.pipeline import live_lag_stats
from run_multi import seq_name
from tools.simulate_rtsp import ffmpeg_cmd, parse

FPS = 25


def test_flat_lag_keeps_up():
    lags = [0.3 + 0.01 * (i % 3) for i in range(FPS * 30)]          # jitter only
    s = live_lag_stats(lags, FPS, expected_frames=FPS * 30)
    assert s["keeps_up"] and abs(s["lag_slope"]) < 0.01 and s["realtime_factor"] == pytest.approx(1, abs=0.01)


def test_startup_backlog_then_catching_up_keeps_up():
    # 3 s of lag builds up during warm-up, is drained by t=10 s, then stays flat
    lags = [min(i / FPS, 3.0) if i < 3 * FPS else max(0.2, 3.0 - (i - 3 * FPS) / FPS * 0.43)
            for i in range(FPS * 40)]
    s = live_lag_stats(lags, FPS, expected_frames=FPS * 40)
    assert s["keeps_up"] and s["lag_max_s"] == pytest.approx(3.0)


def test_growing_lag_falls_behind():
    # processing at 20 fps on a 25 fps stream: each frame arrives 1/20 s after the previous
    lags = [i / 20 - i / FPS for i in range(FPS * 30)]
    s = live_lag_stats(lags, FPS, expected_frames=FPS * 30)
    assert not s["keeps_up"] and s["realtime_factor"] == pytest.approx(20 / 25, abs=0.01)


def test_stream_that_ends_early_does_not_keep_up():
    s = live_lag_stats([0.1] * 61, 30, expected_frames=1800)
    assert s["ended_early"] and not s["keeps_up"]


def test_ffmpeg_cmd_for_video_and_image_folder(tmp_path):
    v = ffmpeg_cmd("cam1", "clip.mpg")
    assert v[v.index("-i") + 1] == "clip.mpg" and v[-1] == "rtsp://127.0.0.1:8554/cam1"
    assert "-re" in v and v[v.index("-stream_loop") + 1] == "-1"
    (tmp_path / "000001.jpg").write_bytes(b"")
    f = ffmpeg_cmd("cam2", f"{tmp_path}@30")
    assert f[f.index("-framerate") + 1] == "30" and f[f.index("-i") + 1].endswith("%06d.jpg")
    assert f[f.index("-g") + 1] == "30"          # one keyframe per second


def test_parse_and_seq_name():
    assert parse(["a=x.mpg", "b=dir@25"]) == {"a": "x.mpg", "b": "dir@25"}
    with pytest.raises(SystemExit):
        parse(["nonsense"])
    assert seq_name("data/caviar/TwoEnterShop1front/TwoEnterShop1front.mpg") == "TwoEnterShop1front"
