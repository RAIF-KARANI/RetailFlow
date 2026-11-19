"""Unit tests for the common ground-truth loader, using tiny hand-written files."""

from pathlib import Path

import pytest

from retailflow.groundtruth import GT_COLUMNS, load_caviar_gt, load_mot_gt

CAVIAR_XML = """<?xml version="1.0" encoding="UTF-8"?><dataset name="test">
  <frame number="0"><objectlist>
    <object id="3"><box h="80" w="30" xc="100" yc="60"/>
      <hypothesislist><hypothesis><role evaluation="1.0">walker</role></hypothesis></hypothesislist>
    </object>
  </objectlist>
  <grouplist><group id="0"><box h="80" w="70" xc="120" yc="60"/></group></grouplist>
  </frame>
  <frame number="1"><objectlist/></frame>
</dataset>"""

# frame,id,left,top,w,h,consider,class,visibility
MOT_TXT = "\n".join([
    "1,7,10,20,30,60,1,1,1.0",   # kept: pedestrian, consider=1
    "2,7,12,20,30,60,1,1,0.9",   # kept
    "1,8,50,50,10,20,0,1,1.0",   # dropped: consider flag 0
    "1,9,50,50,10,20,1,7,1.0",   # dropped: class 7 (static person)
])


def test_caviar_converts_centre_box_and_ignores_groups(tmp_path: Path):
    xml = tmp_path / "gt.xml"
    xml.write_text(CAVIAR_XML)
    df = load_caviar_gt(xml)
    assert list(df.columns) == GT_COLUMNS
    assert len(df) == 1                       # group box not included
    r = df.iloc[0]
    assert (r.frame, r.track_id, r.role) == (0, 3, "walker")
    assert (r.x1, r.y1, r.x2, r.y2) == (85, 20, 115, 100)  # xc±w/2, yc±h/2
    assert (r.feet_x, r.feet_y) == (100, 100)              # bottom-centre


def test_mot_filters_and_makes_frames_zero_based(tmp_path: Path):
    txt = tmp_path / "gt.txt"
    txt.write_text(MOT_TXT)
    df = load_mot_gt(txt)
    assert df["track_id"].unique().tolist() == [7]
    assert df["frame"].tolist() == [0, 1]
    r = df.iloc[0]
    assert (r.x1, r.y1, r.x2, r.y2) == (10, 20, 40, 80)
    assert (r.feet_x, r.feet_y) == pytest.approx((25, 80))
