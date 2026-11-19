"""Multi-camera: several cameras watching the same entrance, on one clock.

A **site** groups cameras of one store. Each camera keeps its own scene file;
the site file (scenes/sites/<site>.json) says which counting line on which
camera watches which shared entrance, and how to put the cameras on one clock:

    {
      "site_id": "caviar_promod",
      "entrance": "promod_door",
      "match_window_s": 1.0,
      "cameras": {
        "caviar_front": {"line": "scene"},                       # the scene's own line
        "caviar_cor":   {"line": {"p1": [70, 130], "p2": [70, 215],
                                  "in_point": [40, 170], "band_px": 5}}
      }
    }

Fusion of two cameras' crossings of the same door (`fuse`): events with the
same direction whose times differ by at most `match_window_s` are the same
person and count once. What happens to unmatched events is the policy:
- "union":        count them (a camera that missed someone is covered by the other);
- "intersection": drop them (only count what both cameras saw);
- a single camera alone is the baseline.
No appearance or identity is used: only direction and time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from retailflow.config import LineConfig, _pt


@dataclass(frozen=True)
class SiteCamera:
    camera_id: str
    line: LineConfig | None      # None = use the camera's scene line


@dataclass(frozen=True)
class Site:
    site_id: str
    entrance: str
    cameras: tuple[SiteCamera, ...]
    match_window_s: float = 1.0


def load_site(path: str | Path) -> Site:
    d = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    cams = []
    for cid, c in d["cameras"].items():
        ln = c.get("line", "scene")
        line = None if ln == "scene" else LineConfig(_pt(ln["p1"]), _pt(ln["p2"]), _pt(ln["in_point"]),
                                                     float(ln.get("band_px", 0)))
        cams.append(SiteCamera(cid, line))
    return Site(d["site_id"], d["entrance"], tuple(cams), float(d.get("match_window_s", 1.0)))


def fuse(a: list[tuple[float, str]], b: list[tuple[float, str]], window_s: float) -> dict:
    """Match two cameras' (time_s, direction) crossings of the same door.

    Greedy in time order: each event of `a` takes the closest unused event of `b`
    with the same direction within `window_s`. Returns per-direction counts for
    each policy plus the matched pairs' time differences.
    """
    used = set()
    pairs = []
    for ta, da in sorted(a):
        best, bj = None, None
        for j, (tb, db) in enumerate(b):
            if j in used or db != da:
                continue
            dt = abs(tb - ta)
            if dt <= window_s and (best is None or dt < best):
                best, bj = dt, j
        if bj is not None:
            used.add(bj)
            pairs.append((ta, b[bj][0], da))
    out = {}
    for d in ("in", "out"):
        na = sum(1 for _, x in a if x == d)
        nb = sum(1 for _, x in b if x == d)
        m = sum(1 for p in pairs if p[2] == d)
        out[d] = {"a": na, "b": nb, "matched": m, "union": na + nb - m, "intersection": m}
    out["dt"] = [round(tb - ta, 3) for ta, tb, _ in pairs]
    return out
