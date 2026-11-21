"""Pure-Python analytics on event rows (no database, no web framework).

Kept separate from the API so it can be unit-tested directly
(tests/test_backend_analytics.py).
"""

from __future__ import annotations

from collections import defaultdict
from statistics import median


def counts_series(crossings: list[tuple[float, str]], bucket_s: float, end_s: float) -> list[dict]:
    """IN/OUT per time bucket. crossings: (ts, direction). Buckets cover 0..end_s."""
    n = max(1, int(end_s // bucket_s) + 1)
    rows = [{"t": round(i * bucket_s, 3), "in": 0, "out": 0} for i in range(n)]
    for ts, d in crossings:
        i = min(int(ts // bucket_s), n - 1)
        rows[i][d] += 1
    return rows


def pair_visits(zone_events: list[dict], end_s: float) -> list[dict]:
    """Turn zone_enter / zone_exit events into visits {track_id, zone, start, end, dwell_s, open}.

    Events must be sorted by ts. An enter without an exit is a visit still in
    progress (open), ending at `end_s` for occupancy purposes.
    """
    open_: dict[tuple, float] = {}
    visits = []
    for e in zone_events:
        key = (e["track_id"], e["zone"])
        if e["type"] == "zone_enter":
            open_[key] = e["ts"]
        elif key in open_:
            start = open_.pop(key)
            visits.append({"track_id": key[0], "zone": key[1], "start": start, "end": e["ts"],
                           "dwell_s": e["value"], "open": False})
    for (tid, zone), start in open_.items():
        visits.append({"track_id": tid, "zone": zone, "start": start, "end": end_s,
                       "dwell_s": round(end_s - start, 3), "open": True})
    return visits


def occupancy_series(visits: list[dict], zones: list[str], bucket_s: float, end_s: float) -> list[dict]:
    """Average number of people in each zone per time bucket (time-weighted)."""
    n = max(1, int(end_s // bucket_s) + 1)
    acc = {z: [0.0] * n for z in zones}
    for v in visits:
        if v["zone"] not in acc:
            continue
        s, e = v["start"], max(v["end"], v["start"])
        i = int(s // bucket_s)
        while i < n and i * bucket_s <= e:
            lo, hi = max(s, i * bucket_s), min(e, (i + 1) * bucket_s)
            if hi > lo:
                acc[v["zone"]][i] += (hi - lo) / bucket_s
            i += 1
    return [{"t": round(i * bucket_s, 3), **{z: round(acc[z][i], 3) for z in zones}} for i in range(n)]


def dwell_stats(visits: list[dict], zones: list[str]) -> list[dict]:
    """Per zone: number of finished visits, mean / median / max dwell, people in it now."""
    by = defaultdict(list)
    now = defaultdict(int)
    for v in visits:
        if v["open"]:
            now[v["zone"]] += 1
        else:
            by[v["zone"]].append(v["dwell_s"])
    out = []
    for z in zones:
        d = by.get(z, [])
        out.append({"zone": z, "visits": len(d),
                    "mean_dwell_s": round(sum(d) / len(d), 1) if d else None,
                    "median_dwell_s": round(median(d), 1) if d else None,
                    "max_dwell_s": round(max(d), 1) if d else None,
                    "occupancy_now": now.get(z, 0)})
    return out


def transition_table(visits: list[dict], zones: list[str]) -> dict:
    """Zone-to-zone movement: how often people went from one zone to another.

    Same rule as retailflow/transitions.py (kept in step by a test): each track's
    visits in order of start give its path, repeats of a zone in a row are merged,
    and every step A -> B counts once. Visits already passed the 2 s zone debounce.
    """
    by_track: dict = {}
    for v in sorted(visits, key=lambda v: (v["track_id"], v["start"], v["end"])):
        path = by_track.setdefault(v["track_id"], [])
        if not path or path[-1] != v["zone"]:
            path.append(v["zone"])
    idx = {z: i for i, z in enumerate(zones)}
    matrix = [[0] * len(zones) for _ in zones]
    for path in by_track.values():
        for a, b in zip(path, path[1:]):
            if a in idx and b in idx:
                matrix[idx[a]][idx[b]] += 1
    steps = [{"from": a, "to": b, "n": matrix[idx[a]][idx[b]]}
             for a in zones for b in zones if a != b and matrix[idx[a]][idx[b]]]
    steps.sort(key=lambda s: -s["n"])
    multi = sum(1 for p in by_track.values() if len(p) > 1)
    return {"zones": zones, "matrix": matrix, "steps": steps,
            "total_steps": sum(s["n"] for s in steps),
            "people_with_paths": len(by_track), "people_moving_between_zones": multi}
