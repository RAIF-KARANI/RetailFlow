"""Multi-camera tracking: one global ID per person across calibrated, synchronised cameras.

Input: each camera's own tracks (from tools/track_multicam.py) and a floor
homography per camera (floor metres -> image pixels). Steps:

1. **Floor positions.** Every box's feet point (bottom centre) is mapped to the
   floor with the inverse homography. Boxes touching the bottom image edge are
   not used for linking: their bottom is not the feet.
2. **Overlap links.** Two tracklets of *different* cameras that are on the floor
   at the same time and stay close (median distance <= `link_m` over at least
   `min_common` shared frames) are the same person seen twice. Links are merged
   closest first; a merge is refused if the joined group would contain two
   tracklets of one camera at the same moment (one person cannot be in a view twice).
3. **Hand-offs.** A group whose path ends, followed by another group whose path
   starts within `gap_s`, at a floor distance a walking person can cover
   (`walk_mps` * gap + `handoff_m`), is the same person leaving one view (or a
   broken track) and appearing again. Linked best-first, each end and start once.

No appearance (ReID) is used: on synchronised cameras with a shared floor,
position alone decides. Appearance would help when views do not overlap and
several people hand off at once; that is listed as a limit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MTMCConfig:
    link_m: float = 0.5          # overlap link: median floor distance (metres); chosen on dev minutes 0-2
    min_common: int = 10         # overlap link: shared (sampled) frames needed
    sample_every: int = 3        # use every n-th frame for overlap distances (speed)
    gap_s: float = 4.0           # hand-off: max time between end and start; chosen on dev minutes 0-2
    handoff_m: float = 1.0       # hand-off: distance allowed at zero gap
    walk_mps: float = 2.0        # hand-off: walking speed allowance (m/s)
    edge_px: float = 4.0         # boxes this close to the bottom edge have no usable feet
    close_frac: float | None = None  # overlap link by share of shared frames within link_m
                                     # (robust to legs hidden behind shelves); None = median rule


def floor_points(tracks: pd.DataFrame, H: np.ndarray, frame_h: int, cfg: MTMCConfig) -> pd.DataFrame:
    """Add xw, yw (floor metres) and `usable` (feet visible) to one camera's tracks."""
    t = tracks.copy()
    fx, fy = (t.x1 + t.x2) / 2, t.y2
    p = np.c_[fx, fy, np.ones(len(t))] @ np.linalg.inv(H).T
    t["xw"], t["yw"] = p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]
    t["usable"] = t.y2 < frame_h - cfg.edge_px
    return t


class _Groups:
    """Union-find over tracklets that keeps, per group, each camera's time spans."""

    def __init__(self, spans: dict):
        self.parent = {k: k for k in spans}
        self.spans = {k: {k[0]: [v]} for k, v in spans.items()}   # root -> cam -> [(start, end)]

    def find(self, k):
        while self.parent[k] != k:
            self.parent[k] = self.parent[self.parent[k]]
            k = self.parent[k]
        return k

    def can_merge(self, a, b) -> bool:
        sa, sb = self.spans[a], self.spans[b]
        for cam, la in sa.items():
            for (s1, e1) in la:
                for (s2, e2) in sb.get(cam, []):
                    if s1 <= e2 and s2 <= e1:      # same camera, same time: two different people
                        return False
        return True

    def merge(self, a, b) -> bool:
        a, b = self.find(a), self.find(b)
        if a == b or not self.can_merge(a, b):
            return False
        self.parent[b] = a
        for cam, lst in self.spans.pop(b).items():
            self.spans[a].setdefault(cam, []).extend(lst)
        return True


def link(cams: dict[str, pd.DataFrame], fps: float, cfg: MTMCConfig = MTMCConfig()) -> pd.DataFrame:
    """Global IDs. `cams`: camera -> tracks with floor points (see `floor_points`).

    Returns one row per tracklet: camera, track_id, global_id, start, end, n.
    """
    rows = []
    for cam, t in cams.items():
        rows.append(t.assign(camera=cam))
    allt = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if allt.empty:
        return pd.DataFrame(columns=["camera", "track_id", "global_id", "start", "end", "n"])
    allt["key"] = list(zip(allt.camera, allt.track_id))
    info = allt.groupby("key").agg(start=("frame", "min"), end=("frame", "max"), n=("frame", "size"))
    groups = _Groups({k: (r.start, r.end) for k, r in info.iterrows()})

    # ---- 2. overlap links: per sampled frame, distances between every pair of cameras' people
    u = allt[allt.usable & (allt.frame % cfg.sample_every == 0)]
    dsum: dict = {}
    for _, g in u.groupby("frame"):
        if g.camera.nunique() < 2:
            continue
        xy = g[["xw", "yw"]].to_numpy()
        keys = g.key.tolist()
        camv = g.camera.to_numpy()
        d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        ii, jj = np.where((d <= 3 * cfg.link_m) & (camv[:, None] < camv[None, :]))
        for i, j in zip(ii, jj):
            dsum.setdefault((keys[i], keys[j]), []).append(d[i, j])
    # Common frames counted over ALL shared frames (also those where they were far apart):
    common = _shared_frames(u)
    cand = []
    for pair, ds in dsum.items():
        n_common = common.get(pair, 0)
        if n_common < cfg.min_common:
            continue
        # median over all shared frames: frames where they were > 3*link_m apart count as far
        ds = np.asarray(ds)
        if cfg.close_frac is not None:
            close = int((ds <= cfg.link_m).sum())
            if close >= cfg.min_common and close / n_common >= cfg.close_frac:
                cand.append((1.0 - close / n_common, pair))     # most consistently close first
            continue
        far = n_common - len(ds)
        med = np.median(np.r_[ds, np.full(far, np.inf)]) if far else np.median(ds)
        if med <= cfg.link_m:
            cand.append((med, pair))
    for _, (a, b) in sorted(cand):
        groups.merge(a, b)

    # ---- 3. hand-offs between groups (also re-joins a camera's own broken tracks)
    gid = {k: groups.find(k) for k in info.index}
    allt["group"] = allt.key.map(gid)
    ends, starts = {}, {}
    for grp, g in allt[allt.usable].groupby("group"):
        g = g.sort_values("frame")
        head, tail = g.head(5), g.tail(5)
        starts[grp] = (int(g.frame.iloc[0]), head.xw.median(), head.yw.median())
        ends[grp] = (int(g.frame.iloc[-1]), tail.xw.median(), tail.yw.median())
    gap = cfg.gap_s * fps
    hand = []
    for a, (fe, xe, ye) in ends.items():
        for b, (fs, xs, ys) in starts.items():
            if a == b or not (0 < fs - fe <= gap):
                continue
            dist = float(np.hypot(xs - xe, ys - ye))
            if dist <= cfg.handoff_m + cfg.walk_mps * (fs - fe) / fps:
                hand.append((dist + (fs - fe) / fps, a, b))
    used_end, used_start = set(), set()
    for _, a, b in sorted(hand):
        if a in used_end or b in used_start:
            continue
        if groups.merge(a, b):
            used_end.add(a)
            used_start.add(b)

    # ---- numbering
    roots = {k: groups.find(k) for k in info.index}
    first: dict = {}
    for k, r in roots.items():                     # number people in order of first appearance
        first[r] = min(first.get(r, info.at[k, "start"]), info.at[k, "start"])
    number = {r: i + 1 for i, r in enumerate(sorted(first, key=lambda r: (first[r], str(r))))}
    out = info.reset_index()
    out["camera"] = [k[0] for k in out.key]
    out["track_id"] = [k[1] for k in out.key]
    out["global_id"] = [number[roots[k]] for k in out.key]
    return out[["camera", "track_id", "global_id", "start", "end", "n"]]


def _shared_frames(u: pd.DataFrame) -> dict:
    """(tracklet a, tracklet b) -> number of sampled frames both are on the floor (different cameras)."""
    spans = u.groupby("key").frame.agg(["min", "max"])
    frames = u.groupby("key").frame.apply(lambda s: set(s.tolist()))
    keys = list(spans.index)
    out = {}
    mins, maxs = spans["min"].to_numpy(), spans["max"].to_numpy()
    for i, a in enumerate(keys):
        for j in range(len(keys)):
            b = keys[j]
            if a[0] >= b[0] or maxs[i] < mins[j] or maxs[j] < mins[i]:
                continue
            out[(a, b)] = len(frames[a] & frames[b])
    return out
