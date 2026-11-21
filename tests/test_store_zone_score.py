"""Scoring store zones: identical inputs score perfectly, a split ID shows up only in visits."""

import pandas as pd

from evaluate_store_zones import score
from retailflow.config import Zone

ZONES = [Zone("a", ((-5.0, -5.0), (0.0, -5.0), (0.0, 5.0), (-5.0, 5.0))),
         Zone("b", ((0.0, -5.0), (5.0, -5.0), (5.0, 5.0), (0.0, 5.0)))]


def walk(person, secs, x):
    return pd.DataFrame({"sec": list(secs), "person": person, "xw": x, "yw": 0.0})


def test_truth_against_itself_is_perfect():
    t = pd.concat([walk(1, range(0, 20), -2.5), walk(1, range(20, 40), 2.5)])
    r = score(t, t, ZONES, 0, 40)["total"]
    assert r["occupancy_mae_mean"] == 0 and r["person_s_wape_pct"] == 0
    assert r["visits_wape_pct"] == 0 and r["moves_wape_pct"] == 0 and r["moves_true"] == 1


def test_one_person_split_into_two_ids_keeps_time_but_adds_visits():
    t = walk(1, range(0, 40), -2.5)                                   # 40 s in zone a, one visit
    p = pd.concat([walk(1, range(0, 20), -2.5), walk(2, range(20, 40), -2.5)])   # same, two IDs
    r = score(t, p, ZONES, 0, 40)["total"]
    assert r["occupancy_mae_mean"] <= 0.1 and r["person_s_wape_pct"] <= 5
    assert r["visits_true"] == 1 and r["visits_pred"] == 2
