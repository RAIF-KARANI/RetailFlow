"""Static-object filter: drop "people" that never move (mannequins, posters, statues).

Shop windows are full of mannequins, and a person detector sees them as people.
They never cross the counting line, but they would sit in zones and heatmaps
for hours. A real person waiting in one place still **arrives and leaves**,
so over the whole life of their track their feet move.

Rule, applied to a **finished** track (its whole life):

    static  if  life >= min_seconds
            and the feet point's x-range and y-range are both
                <= max_move_frac x the track's median box height

Box height is used as the yardstick so the rule works for near (big) and far
(small) objects alike. Defaults (10 s, 0.5 x height) were set from this
reasoning before measuring; their effect is reported in docs/05.

It is switched on **per camera** (`"static_filter": true` in the scene file),
because people who really stand still for their whole visible life (a guard,
a cashier, people chatting on a street) would be removed too. Measured on
ground truth: 26% of real person-seconds on MOT17-04 (people standing on a
street), 0-3% on the shopping-centre cameras (docs/05).

Limitation: if a waiting person's track is broken by an ID switch, the
fragment that only covers the standing part can look static and be dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_SECONDS = 10.0
MAX_MOVE_FRAC = 0.5


def is_static(feet: np.ndarray, heights: np.ndarray, fps: float,
              min_seconds: float = MIN_SECONDS, max_move_frac: float = MAX_MOVE_FRAC) -> bool:
    """feet: (N, 2) feet points of one track over its life; heights: (N,) box heights."""
    if len(feet) < min_seconds * fps:
        return False
    span = np.ptp(np.asarray(feet, float), axis=0)          # (x range, y range)
    return bool((span <= max_move_frac * float(np.median(heights))).all())


def static_tracks(table: pd.DataFrame, fps: float, min_seconds: float = MIN_SECONDS,
                  max_move_frac: float = MAX_MOVE_FRAC) -> set[int]:
    """Track IDs in a tracks/GT table that the rule marks as static objects."""
    if table.empty:
        return set()
    t = table.assign(fx=(table.x1 + table.x2) / 2, fy=table.y2, h=table.y2 - table.y1)
    out = set()
    for tid, g in t.groupby("track_id"):
        if is_static(g[["fx", "fy"]].to_numpy(), g.h.to_numpy(), fps, min_seconds, max_move_frac):
            out.add(int(tid))
    return out
