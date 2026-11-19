"""Tests for evaluation helpers (IoU distance, tracking metrics, density, summary)."""

import numpy as np
import pandas as pd
import pytest

from retailflow.evaluate import density_group, iou_distance, summarise, tracking_metrics


def test_iou_distance_values_and_threshold():
    gt = np.array([[0, 0, 10, 10]], float)
    pr = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [50, 50, 60, 60]], float)
    d = iou_distance(gt, pr)
    assert d[0, 0] == pytest.approx(0.0)       # identical boxes
    assert np.isnan(d[0, 1])                   # IoU 1/3 < 0.5 -> not matchable
    assert np.isnan(d[0, 2])                   # no overlap


def test_tracking_metrics_perfect_and_id_switch():
    box = dict(x1=0, y1=0, x2=10, y2=10)
    gt = pd.DataFrame([{"frame": f, "track_id": 1, **box} for f in range(4)])
    perfect = gt.copy()
    m = tracking_metrics(gt, perfect, 4)
    assert m["idf1"] == 1.0 and m["num_switches"] == 0
    switched = gt.copy()
    switched.loc[switched.frame >= 2, "track_id"] = 2  # same person, new ID halfway
    m2 = tracking_metrics(gt, switched, 4)
    assert m2["num_switches"] == 1 and m2["idf1"] == pytest.approx(0.5)


def test_density_groups():
    assert density_group(1.0) == "quiet"
    assert density_group(8.0) == "medium"
    assert density_group(45.0) == "busy"


def test_summarise_wape_does_not_cancel_errors():
    res = pd.DataFrame([
        # 3 missed INs and 3 extra OUTs: net total is right, but WAPE must be 60%.
        dict(config="c", variant="v", camera="cam", sequence="s1", true_in=5, true_out=5,
             pred_in=2, pred_out=8, abs_err=6, idf1=0.8, avg_proc_ms=10.0),
    ])
    s = summarise(res, ["camera"])
    assert s.loc[0, "wape_pct"] == 60.0
    assert s.loc[0, "exact_seqs"] == 0
