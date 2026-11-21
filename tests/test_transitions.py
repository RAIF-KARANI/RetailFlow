"""Zone-to-zone transitions: paths, matrix, events round trip, backend agrees."""

import importlib.util
from pathlib import Path

import pandas as pd

from retailflow.transitions import (END, START, transition_counts, transition_events,
                                    transition_matrix, zone_paths)

_spec = importlib.util.spec_from_file_location("backend_analytics", Path("backend/app/analytics.py"))
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

VISITS = pd.DataFrame([
    # track 1: entrance -> walkway -> shop, with a repeat of walkway merged
    (1, "entrance", 0, 50), (1, "walkway", 60, 120), (1, "walkway", 130, 150), (1, "shop", 160, 400),
    # track 2: shop -> entrance
    (2, "shop", 10, 90), (2, "entrance", 100, 140),
    # track 3: one zone only, no move
    (3, "walkway", 5, 300),
], columns=["track_id", "zone", "start_frame", "end_frame"])


def test_paths_merge_repeats_and_keep_order():
    p = zone_paths(VISITS.sample(frac=1, random_state=0))      # input order must not matter
    assert p == {1: ["entrance", "walkway", "shop"], 2: ["shop", "entrance"], 3: ["walkway"]}


def test_counts_and_matrix():
    c = transition_counts(VISITS)
    assert c == {("entrance", "walkway"): 1, ("walkway", "shop"): 1, ("shop", "entrance"): 1}
    m = transition_matrix(c, ["entrance", "walkway", "shop"])
    assert m.loc["entrance", "walkway"] == 1 and m.loc["walkway", "entrance"] == 0
    assert int(m.values.sum()) == 3
    e = transition_counts(VISITS, ends=True)
    assert e[(START, "walkway")] == 1 and e[("shop", END)] == 1 and e[(START, "entrance")] == 1


def test_visits_rebuilt_from_events_csv_rows():
    ev = pd.DataFrame([
        ("zone_enter", 1, "entrance", 0), ("zone_exit", 1, "entrance", 50),
        ("zone_enter", 1, "walkway", 60), ("line_cross", 1, "", 70), ("zone_exit", 1, "walkway", 120),
        ("zone_enter", 1, "shop", 160),                       # still inside at the end
    ], columns=["type", "track_id", "zone", "frame"])
    assert transition_counts(transition_events(ev)) == {("entrance", "walkway"): 1, ("walkway", "shop"): 1}


def test_backend_copy_gives_the_same_matrix():
    zones = ["entrance", "walkway", "shop"]
    api_visits = [{"track_id": r.track_id, "zone": r.zone, "start": r.start_frame / 25, "end": r.end_frame / 25}
                  for r in VISITS.itertuples()]
    t = B.transition_table(api_visits, zones)
    m = transition_matrix(transition_counts(VISITS), zones)
    assert t["matrix"] == m.values.tolist()
    assert t["total_steps"] == 3 and t["people_with_paths"] == 3 and t["people_moving_between_zones"] == 2
    assert t["steps"][0]["n"] == 1
