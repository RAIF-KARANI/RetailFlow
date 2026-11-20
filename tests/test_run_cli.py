"""run.py command-line options."""

from pathlib import Path

import pytest

import run
from retailflow.sources import FrameSource

CLIP = Path("data/caviar/TwoEnterShop1front/TwoEnterShop1front.mpg")


def test_duration_option_parses():
    args = run.build_parser().parse_args(["--source", "x.mp4", "--scene", "s.json", "--duration", "2.5"])
    assert args.duration == 2.5


@pytest.mark.skipif(not CLIP.exists(), reason="CAVIAR data not downloaded")
def test_duration_uses_source_fps():
    src = FrameSource(str(CLIP))
    run.apply_duration(src, 2)                      # CAVIAR is 25 fps
    assert src.max_frames == 50
    assert sum(1 for _ in src) == 50
    capped = FrameSource(str(CLIP), max_frames=10)
    run.apply_duration(capped, 2)                   # the smaller limit wins
    assert capped.max_frames == 10
