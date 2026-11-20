"""Staff filtering: exclude employees from customer counts, dwell and heatmaps.

Rule (per camera, in the scene file):

    "staff": {"zones": ["shop_inside"], "min_seconds": 45}

A track is marked **staff** once it has spent `min_seconds` in total inside any
of the listed zones (e.g. behind a till, or inside the shop seen from the mall).
Customers pass through these areas; employees stay there.

Why a rule and not a classifier: none of the public datasets used here label
who is staff, so a learned model could not be trained or tested. The rule only
uses positions and time, which keeps it anonymous (no appearance is stored).

Known weaknesses (docs/06_STAFF_AND_PRIVACY.md):
- the decision comes only after `min_seconds`, so a staff member's earlier
  events are flagged retroactively (the pipeline emits a `staff_flag` event,
  and offline analysis drops all of that track's rows);
- an ID switch splits a staff member into shorter tracks, which may each stay
  under the threshold;
- a customer who stays longer than `min_seconds` in the zone is misclassified.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from retailflow.config import StaffRule  # defined with the other scene settings
from retailflow.zones import points_in_polygon

__all__ = ["StaffRule", "StaffFilter", "staff_tracks"]


class StaffFilter:
    """Online: accumulates time in staff zones per track and flags staff tracks."""

    def __init__(self, rule: StaffRule | None, zones, fps: float):
        self.rule = rule
        self.fps = fps
        self.polygons = [z.polygon for z in zones if rule and z.name in rule.zones]
        self.frames_in: dict[int, int] = defaultdict(int)
        self.staff: set[int] = set()

    def update(self, points: dict[int, tuple[float, float]], step: int = 1) -> list[int]:
        """Feed this frame's feet points; returns track IDs newly flagged as staff.

        `step`: frames this update stands for (> 1 when a live stream skipped frames).
        """
        if not self.polygons or not points:
            return []
        ids = list(points)
        pts = np.array([points[i] for i in ids], dtype=float)
        inside = np.zeros(len(ids), dtype=bool)
        for poly in self.polygons:
            inside |= points_in_polygon(pts, poly)
        new = []
        need = self.rule.min_seconds * self.fps
        for tid, ok in zip(ids, inside):
            if ok and tid not in self.staff:
                self.frames_in[tid] += step
                if self.frames_in[tid] >= need:
                    self.staff.add(tid)
                    new.append(tid)
        return new

    def is_staff(self, track_id: int) -> bool:
        return track_id in self.staff


def staff_tracks(table: pd.DataFrame, rule: StaffRule | None, zones, fps: float) -> set[int]:
    """Offline: the set of track IDs a whole tracks/GT table marks as staff."""
    if rule is None or table.empty:
        return set()
    t = table if "feet_x" in table else table.assign(feet_x=(table.x1 + table.x2) / 2, feet_y=table.y2)
    inside = np.zeros(len(t), dtype=bool)
    for z in zones:
        if z.name in rule.zones:
            inside |= points_in_polygon(t[["feet_x", "feet_y"]].to_numpy(), z.polygon)
    secs = pd.Series(inside, index=t.index).groupby(t.track_id.values).sum() / fps
    return set(secs[secs >= rule.min_seconds].index.astype(int))
