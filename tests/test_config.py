"""Tests for scene config loading/saving and setup_scene helpers."""

import json
import sys
from pathlib import Path

import pytest

from retailflow.config import load_scene, save_scene, scene_from_dict, scene_path_for_sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from setup_scene import default_scene, parse_floor_points  # noqa: E402

SCENE = {
    "camera_id": "cam1", "frame_size": [384, 288],
    "line": {"p1": [0, 100], "p2": [200, 100], "in_point": [100, 150], "band_px": 5},
    "zones": [{"name": "entrance", "polygon": [[0, 0], [10, 0], [10, 10]]}],
    "homography": {"image_points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                   "floor_points": [[0, 0], [5, 0], [5, 5], [0, 5]]},
}


def test_round_trip(tmp_path: Path):
    s = scene_from_dict(SCENE)
    save_scene(s, tmp_path / "s.json")
    assert load_scene(tmp_path / "s.json") == s


def test_loads_file_with_utf8_bom(tmp_path: Path):
    p = tmp_path / "bom.json"
    p.write_bytes(b"\xef\xbb\xbf" + json.dumps(SCENE).encode())  # what PowerShell 5.1 writes
    assert load_scene(p).camera_id == "cam1"


def test_rejects_bad_geometry():
    bad_line = {**SCENE, "line": {**SCENE["line"], "p2": [0, 100]}}
    with pytest.raises(ValueError):
        scene_from_dict(bad_line)
    bad_zone = {**SCENE, "zones": [{"name": "z", "polygon": [[0, 0], [1, 1]]}]}
    with pytest.raises(ValueError):
        scene_from_dict(bad_zone)


def test_sequence_to_scene_mapping():
    assert scene_path_for_sequence("OneLeaveShop1cor").name == "caviar_cor.json"
    assert scene_path_for_sequence("OneLeaveShop1front").name == "caviar_front.json"
    assert scene_path_for_sequence("MOT17-04-FRCNN").name == "MOT17-04.json"
    assert scene_path_for_sequence("MOT20-02").name == "MOT20-02.json"


def test_repo_scenes_are_valid():
    scenes = list(Path("scenes").glob("*.json"))
    assert scenes, "no scene files found"
    for p in scenes:
        load_scene(p)


def test_setup_helpers():
    assert parse_floor_points("0,0; 5,0; 5,10; 0,10") == [(0, 0), (5, 0), (5, 10), (0, 10)]
    with pytest.raises(ValueError):
        parse_floor_points("0,0; 5,0")
    s = default_scene("x", 300, 300)
    assert s.line.p1[1] == s.line.p2[1] == 200
