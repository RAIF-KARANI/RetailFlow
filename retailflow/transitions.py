"""Zone-to-zone movement: which zone people go to next.

Input is a table of zone visits (one row per visit, as produced by
`zones.ZoneTracker` / `zones.zone_analysis`): track_id, zone, start_frame,
end_frame. Each track's visits, in order of their start, give its **path**
(e.g. entrance_walkway -> window_walkway -> shop_inside). Repeats of the same
zone in a row are merged (a person who leaves a zone for a moment and comes
back has not moved anywhere). Every step A -> B of a path counts once in the
**transition matrix**.

Visits already went through the zone debounce (2 s to enter, 2 s to leave), so
a person walking along a zone border does not produce A -> B -> A -> B.

Why it can be wrong: a path only exists within one track. If the tracker loses
a person between two zones and gives them a new ID, the step A -> B is lost (two
one-zone paths instead); an ID switch between two people can create a step
nobody made. `evaluate_transitions.py` measures this against ground truth.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

START = "(appears)"   # optional pseudo-zones: where paths begin and end
END = "(leaves)"


def zone_paths(visits: pd.DataFrame) -> dict[int, list[str]]:
    """track_id -> zones in the order they were entered, consecutive repeats merged."""
    paths: dict[int, list[str]] = {}
    if visits.empty:
        return paths
    v = visits.sort_values(["track_id", "start_frame", "end_frame"])
    for tid, g in v.groupby("track_id", sort=False):
        path: list[str] = []
        for z in g.zone:
            if not path or path[-1] != z:
                path.append(z)
        paths[int(tid)] = path
    return paths


def transition_counts(visits: pd.DataFrame, ends: bool = False) -> Counter:
    """Counter {(from_zone, to_zone): n}. With `ends`, also (START, first) and (last, END)."""
    c: Counter = Counter()
    for path in zone_paths(visits).values():
        p = [START, *path, END] if ends else path
        for a, b in zip(p, p[1:]):
            c[(a, b)] += 1
    return c


def transition_matrix(counts: Counter, zones: list[str], ends: bool = False) -> pd.DataFrame:
    """Square table: rows = from, columns = to, in the scene's zone order."""
    rows = ([START] if ends else []) + list(zones)
    cols = list(zones) + ([END] if ends else [])
    m = pd.DataFrame(0, index=pd.Index(rows, name="from"), columns=pd.Index(cols, name="to"))
    for (a, b), n in counts.items():
        if a in m.index and b in m.columns:
            m.loc[a, b] += n
    return m


def transition_events(events: pd.DataFrame) -> pd.DataFrame:
    """Visits from an events.csv table (zone_enter / zone_exit rows), for `transition_counts`.

    zone_enter rows carry the visit's true start frame and zone_exit rows its end
    frame (pipeline.py), so pairing them per (track, zone) in order gives the visits.
    """
    e = events[events["type"].isin(["zone_enter", "zone_exit"])]
    rows = []
    open_: dict[tuple[int, str], int] = {}
    for r in e.sort_values("frame", kind="stable").itertuples(index=False):
        key = (int(r.track_id), str(r.zone))
        if r.type == "zone_enter":
            open_[key] = int(r.frame)
        elif key in open_:
            rows.append((key[0], key[1], open_.pop(key), int(r.frame)))
    for (tid, z), start in open_.items():          # still inside when the run ended
        rows.append((tid, z, start, start))
    return pd.DataFrame(rows, columns=["track_id", "zone", "start_frame", "end_frame"])
