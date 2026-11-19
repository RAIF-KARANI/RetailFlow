"""Tests for retailflow.profiles and the scene 'profile' field."""

import pytest

from retailflow.config import load_scene, scene_from_dict, scene_to_dict
from retailflow.profiles import CROWD_MIN_PEOPLE, PROFILES, get_profile, suggest_profile

BASE = {"camera_id": "cam", "frame_size": [100, 100],
        "line": {"p1": [0, 50], "p2": [100, 50], "in_point": [50, 80]}}


def test_run_names_match_benchmark_format():
    assert PROFILES["standard"].run_name() == "m960_botsort_nogmc"
    assert PROFILES["crowd"].run_name(suffix="_trt") == "m1920_botsort_nogmc_trt"
    assert PROFILES["standard"].run_name("botsort") == "m960_botsort"      # Phase 3 runs


def test_backend_selects_weights(tmp_path, monkeypatch):
    import retailflow.profiles as P
    monkeypatch.setattr(P, "MODELS_DIR", tmp_path)
    prof = P.PROFILES["standard"]
    assert prof.weights("auto") == "yolo26m.pt"                  # no engine built yet
    with pytest.raises(FileNotFoundError):
        prof.weights("trt")
    (tmp_path / "yolo26m_960.engine").write_bytes(b"")
    assert prof.weights("auto").endswith("yolo26m_960.engine") and prof.weights("pt") == "yolo26m.pt"


def test_suggest_profile_threshold():
    assert suggest_profile(CROWD_MIN_PEOPLE - 0.1) == "standard"
    assert suggest_profile(CROWD_MIN_PEOPLE) == "crowd"


def test_scene_profile_default_and_round_trip():
    assert scene_from_dict(BASE).profile == "standard"
    s = scene_from_dict({**BASE, "profile": "crowd"})
    assert scene_from_dict(scene_to_dict(s)).profile == "crowd"


def test_unknown_profile_rejected():
    with pytest.raises(ValueError):
        scene_from_dict({**BASE, "profile": "turbo"})
    with pytest.raises(ValueError):
        get_profile("turbo")


def test_repo_scene_profiles():
    """Scenes were assigned by the people-per-frame rule (docs/04 section 5.5)."""
    assert load_scene("scenes/caviar_front.json").profile == "standard"
    assert load_scene("scenes/MOT17-04.json").profile == "crowd"
    assert load_scene("scenes/MOT20-05.json").profile == "crowd"
