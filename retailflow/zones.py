"""Zones: who is inside each floor area, for how long (dwell), and how many at once (occupancy).

A zone is a polygon on the floor, drawn per camera in the scene file. A person
is "in" a zone when their **feet point** is inside the polygon (same reason as
for counting: in an angled view only the feet are on the floor).

Debouncing (the zone version of the counting band)
--------------------------------------------------
The feet point jitters, and tracks briefly lose a person behind others. Without
smoothing, one person standing on a zone edge produces many tiny visits. So:

- a visit **starts** after `enter_s` seconds of consecutive frames inside, and
  is back-dated to the first of those frames;
- a visit **ends** after `exit_s` seconds of consecutive frames outside *or
  missing*, and its end is the last frame the person was actually inside.

    dwell_s = (end_frame - start_frame + 1) / fps

Visits still open when the video ends are closed at the last frame and marked
`truncated` (their real dwell is at least that long).

The same class runs online in the pipeline (emitting zone_enter / zone_exit
events) and offline on saved tracks or on ground truth, so predicted and true
dwell are computed by identical rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from retailflow.config import Zone

Point = tuple[float, float]


@dataclass(frozen=True)
class ZoneConfig:
    """Debounce times in seconds (converted to frames with the video fps).

    Defaults chosen on dev data (docs/05 section 3): a visit must last at least
    2 s, and leaving for less than 2 s does not end it.
    """

    enter_s: float = 2.0     # time inside before a visit counts
    exit_s: float = 2.0      # time outside/missing before a visit ends

    def frames(self, fps: float) -> tuple[int, int]:
        return max(1, round(self.enter_s * fps)), max(1, round(self.exit_s * fps))


def points_in_polygon(points: np.ndarray, polygon) -> np.ndarray:
    """Vectorised ray casting. points: (N, 2) array -> (N,) bool array.

    A horizontal ray from each point is cast to the right; the point is inside
    if the ray crosses the polygon's edges an odd number of times.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    poly = np.asarray(polygon, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    xj, yj = poly[-1]
    for xi, yi in poly:
        crosses = (yi > y) != (yj > y)            # edge spans the point's y
        with np.errstate(divide="ignore", invalid="ignore"):
            x_edge = xi + (y - yi) * (xj - xi) / (yj - yi)
        inside ^= crosses & (x < x_edge)
        xj, yj = xi, yi
    return inside


@dataclass
class _State:
    run_in: int = 0          # consecutive frames inside (while not yet in a visit)
    run_out: int = 0         # consecutive frames outside/missing (while in a visit)
    first_in: int = -1       # first frame of the current inside run
    last_in: int = -1        # last frame the person was inside
    visiting: bool = False


@dataclass(frozen=True)
class Visit:
    track_id: int
    zone: str
    start_frame: int
    end_frame: int
    dwell_s: float
    truncated: bool = False


class ZoneTracker:
    """Online zone membership for many tracks and zones.

    Call `update(frame, {track_id: feet_point})` once per frame with ALL people
    seen in that frame. Returns ("enter"|"exit", track_id, zone, Visit|None)
    tuples. Call `finish(last_frame)` at the end to close open visits.
    """

    def __init__(self, zones: tuple[Zone, ...] | list[Zone], fps: float, cfg: ZoneConfig = ZoneConfig()):
        self.zones = list(zones)
        self.fps = fps
        self.cfg = cfg
        self.enter_frames, self.exit_frames = cfg.frames(fps)
        self.states: dict[tuple[int, str], _State] = {}
        self.visits: list[Visit] = []

    def _close(self, key, st: _State, truncated=False) -> Visit:
        v = Visit(key[0], key[1], st.first_in, st.last_in,
                  round((st.last_in - st.first_in + 1) / self.fps, 3), truncated)
        self.visits.append(v)
        st.visiting, st.run_in, st.run_out = False, 0, 0
        return v

    def update(self, frame: int, points: dict[int, Point]) -> list[tuple]:
        out = []
        ids = list(points)
        pts = np.array([points[i] for i in ids], dtype=float).reshape(-1, 2)
        for zone in self.zones:
            inside = points_in_polygon(pts, zone.polygon) if ids else np.zeros(0, bool)
            now_in = {tid for tid, ok in zip(ids, inside) if ok}
            # people inside this frame
            for tid in now_in:
                st = self.states.setdefault((tid, zone.name), _State())
                if st.visiting:
                    st.run_out = 0
                else:
                    if st.run_in == 0:
                        st.first_in = frame
                    st.run_in += 1
                    if st.run_in >= self.enter_frames:
                        st.visiting = True
                        st.run_out = 0
                        out.append(("enter", tid, zone.name, None))
                st.last_in = frame
            # everyone else known in this zone: outside or missing
            idle = []
            for (tid, zname), st in self.states.items():
                if zname != zone.name or tid in now_in:
                    continue
                if st.visiting:
                    st.run_out += 1
                    if st.run_out >= self.exit_frames:
                        out.append(("exit", tid, zname, self._close((tid, zname), st)))
                if not st.visiting:
                    idle.append((tid, zname))   # nothing in progress: forget it
            for key in idle:
                del self.states[key]
        return out

    def occupancy(self) -> dict[str, int]:
        """People currently in each zone (visits in progress)."""
        occ = {z.name: 0 for z in self.zones}
        for (_, zname), st in self.states.items():
            if st.visiting:
                occ[zname] += 1
        return occ

    def finish(self) -> list[tuple]:
        """Close every open visit at the last frame its person was inside."""
        out = []
        for key, st in self.states.items():
            if st.visiting:
                out.append(("exit", key[0], key[1], self._close(key, st, truncated=True)))
        return out


def zone_analysis(table: pd.DataFrame, zones, fps: float, n_frames: int,
                  cfg: ZoneConfig = ZoneConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Offline zone analysis of a tracks or ground-truth table.

    `table` needs frame, track_id and either feet_x/feet_y or x1/x2/y2.
    Returns (visits, occupancy): visits has one row per Visit; occupancy has
    one row per frame in 0..n_frames-1 and one column per zone.
    """
    t = table
    if "feet_x" not in t:
        t = t.assign(feet_x=(t.x1 + t.x2) / 2, feet_y=t.y2)
    zt = ZoneTracker(zones, fps, cfg)
    occ = np.zeros((n_frames, len(zt.zones)), dtype=int)
    by_frame = {f: g for f, g in t.groupby("frame")}
    for f in range(n_frames):
        g = by_frame.get(f)
        pts = {} if g is None else dict(zip(g.track_id.astype(int), zip(g.feet_x, g.feet_y)))
        zt.update(f, pts)
        o = zt.occupancy()
        occ[f] = [o[z.name] for z in zt.zones]
    zt.finish()
    visits = pd.DataFrame([v.__dict__ for v in zt.visits],
                          columns=["track_id", "zone", "start_frame", "end_frame", "dwell_s", "truncated"])
    occupancy = pd.DataFrame(occ, columns=[z.name for z in zt.zones])
    occupancy.index.name = "frame"
    return visits, occupancy
