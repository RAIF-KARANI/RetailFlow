"""Tests for benchmark config naming and the automated selection rule."""

import pytest

from benchmark import parse_config, pick_best
from retailflow.evaluate import split_for, sweep_variants


def test_parse_config_names_and_alias():
    assert parse_config("n1280_botsort_reid") == {"model": "yolo26n.pt", "imgsz": 1280,
                                                  "tracker": "botsort_reid", "conf": 0.1}
    assert parse_config("baseline") == parse_config("n640_bytetrack")
    with pytest.raises(ValueError):
        parse_config("bytetrack")


def test_parse_config_exported_backends():
    assert parse_config("m960_botsort_trt")["model"] == "models/yolo26m_960.engine"
    assert parse_config("m1920_botsort_onnx")["model"] == "models/yolo26m_1920.onnx"
    assert parse_config("m960_botsort_reid")["tracker"] == "botsort_reid"   # not mistaken for a backend
    c = parse_config("m1920_botsort_nogmc_trt")
    assert (c["tracker"], c["model"]) == ("botsort_nogmc", "models/yolo26m_1920.engine")


def test_pick_best_prefers_fewest_errors():
    scores = [{"config": "a", "dev_abs_err": 30, "ms_per_frame": 10},
              {"config": "b", "dev_abs_err": 20, "ms_per_frame": 40}]
    assert pick_best(scores) == "b"


def test_pick_best_tie_goes_to_faster():
    scores = [{"config": "slow", "dev_abs_err": 20, "ms_per_frame": 40},
              {"config": "fast", "dev_abs_err": 21, "ms_per_frame": 12}]  # within 1 error
    assert pick_best(scores) == "fast"


def test_split_and_sweep():
    assert split_for("caviar_cor") == "dev" and split_for("MOT17-04") == "dev"
    assert split_for("caviar_front") == "test" and split_for("MOT20-02") == "test"
    v = sweep_variants()
    assert len(v) == 4 * 4 * 2 and "feet_b1_age0" in v and "feet_b2_age10_once" in v
