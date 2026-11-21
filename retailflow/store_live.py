"""Live cross-camera state of a whole store (run_site.py -> API -> dashboard).

`StoreLinker` collects every camera's tracks as they arrive (feet on the floor,
colour descriptors in memory only), re-runs the cross-camera linking
(`retailflow/mtmc.py`) every few seconds, and keeps person numbers **stable**
between re-links: a new group of tracks takes over the number most of its
tracks already had. `state()` summarises the store for the dashboard:
people now (floor position, which cameras see them), 10 s trails, unique people,
a people table and camera-to-camera moves.

Times are seconds of the store's video timeline (frame / fps): the dataset has no
real clock, so none is invented.
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from retailflow.appearance import MIN_H_PX, box_descriptor
from retailflow.config import Zone
from retailflow.mtmc import MTMCConfig, link
from retailflow.transitions import transition_counts
from retailflow.zones import ZoneConfig, ZoneTracker

DESC_EVERY_S = 1.0      # one colour sample per track per second at most
DESC_MAX = 6            # ... and at most this many per track
MIN_SEEN_S = 2.0        # a person counts as a visitor after this many seconds seen
TRAIL_S = 10            # trail length shown on the floor map
WINDOW_S = 120          # only tracks seen in the last WINDOW_S are re-linked; older people keep their number
HEAT_CELL_M = 0.5       # floor heatmap cell size (metres)
# Linking settings for a store: never-moving tracks and tracks shorter than 5 s get no
# store-wide ID (chosen on dev minutes 0-2 of the synthetic store: zone visit error
# 114% -> 35%, time in zones 31.7% -> 16.9%; global IDF1 0.509 -> 0.500; docs/11 section 8).
STORE_LINKING = MTMCConfig(drop_static=True, min_track_s=5.0)


@dataclass
class StoreConfig:
    store_id: str
    description: str
    scene_dir: Path
    cameras: list[str]
    homographies: dict[str, np.ndarray]
    camera_xy: dict[str, tuple[float, float]]     # camera position on the floor (from calibration)
    floor_bounds: tuple[float, float, float, float]
    frame_h: int = 1080
    fps: float = 30.0
    raw: dict = field(default_factory=dict)
    zones: tuple = ()                               # store zones on the floor (metres), config "zones"

    def to_api(self) -> dict:
        """What the dashboard needs to draw the store (no video paths)."""
        return {"store_id": self.store_id, "description": self.description, "cameras": self.cameras,
                "camera_xy": {c: [round(x, 2), round(y, 2)] for c, (x, y) in self.camera_xy.items()},
                "floor_bounds": list(self.floor_bounds),
                "zones": [{"name": z.name, "polygon": [list(pt) for pt in z.polygon]} for z in self.zones]}


def camera_floor_position(P: np.ndarray) -> tuple[float, float]:
    """Where a camera stands on the floor: its centre C solves P @ [C, 1] = 0."""
    M, p4 = P[:, :3], P[:, 3]
    c = -np.linalg.solve(M, p4)
    return float(c[0]), float(c[1])


def load_store(path: str | Path) -> StoreConfig:
    d = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    scene = Path(d["scene_dir"])
    cams = d.get("cameras") or sorted(p.parent.name for p in scene.glob("camera_*/video.mp4")
                                      if p.stat().st_size > 1_000_000)
    H, pos = {}, {}
    for c in cams:
        cal = json.loads((scene / c / "calibration.json").read_text())
        H[c] = np.array(cal["homography matrix"], float)
        pos[c] = camera_floor_position(np.array(cal["camera projection matrix"], float))
    zones = tuple(Zone(z["name"], tuple(tuple(map(float, pt)) for pt in z["polygon"])) for z in d.get("zones") or [])
    return StoreConfig(d["store_id"], d.get("description", ""), scene, cams, H, pos,
                       tuple(d["floor_bounds"]), d.get("frame_h", 1080), d.get("fps", 30.0), d, zones)


class StoreLinker:
    """Collects live tracks of all cameras and keeps stable store-wide person numbers."""

    def __init__(self, store: StoreConfig, cfg: MTMCConfig = STORE_LINKING, appearance: bool = True,
                 window_s: float = WINDOW_S):
        self.store, self.cfg, self.use_app, self.window_s = store, cfg, appearance, window_s
        self.rows: dict[str, list] = {c: [] for c in store.cameras}
        self.desc: dict[tuple, list] = {}          # (camera, track) -> [sum vector, count, last time]
        self.person_of: dict[tuple, int] = {}      # (camera, track) -> stable person number
        self.next_person = 1
        self.now_frame = 0
        # add() runs in the detection loop, relink()/state() may run in another thread:
        # the lock guards rows and colours; linking works on a copy taken under the lock.
        self.lock = threading.Lock()
        self._snap: tuple | None = None

    # ---- collect
    def add(self, cam: str, frame: int, people, image: np.ndarray | None = None) -> None:
        if not people:
            with self.lock:
                self.now_frame = max(self.now_frame, frame)
            return
        b = np.array([(p.x1, p.y1, p.x2, p.y2) for p in people], float)
        Hi = np.linalg.inv(self.store.homographies[cam])
        q = np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3], np.ones(len(b))] @ Hi.T
        xw, yw = q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]
        usable = b[:, 3] < self.store.frame_h - self.cfg.edge_px
        rows = [(frame, p.track_id, *bb, float(x), float(y), bool(u))
                for p, bb, x, y, u in zip(people, b.tolist(), xw, yw, usable)]
        with self.lock:
            self.now_frame = max(self.now_frame, frame)
            self.rows[cam].extend(rows)
            if self.use_app and image is not None:
                self._sample_colours(cam, frame, people, image)

    def _sample_colours(self, cam, frame, people, image) -> None:
        boxes = np.array([[p.x1, p.y1, p.x2, p.y2] for p in people])
        for i, p in enumerate(people):
            if p.y2 - p.y1 < MIN_H_PX:
                continue
            key = (cam, p.track_id)
            d = self.desc.get(key)
            if d and (d[1] >= DESC_MAX or frame - d[2] < DESC_EVERY_S * self.store.fps):
                continue
            # skip views where someone else covers more than 20% of the box (colours would mix)
            ix = np.clip(np.minimum(boxes[:, 2], p.x2) - np.maximum(boxes[:, 0], p.x1), 0, None)
            iy = np.clip(np.minimum(boxes[:, 3], p.y2) - np.maximum(boxes[:, 1], p.y1), 0, None)
            inter = ix * iy
            inter[i] = 0
            if inter.max(initial=0) > 0.2 * (p.x2 - p.x1) * (p.y2 - p.y1):
                continue
            v = box_descriptor(image, p.x1, p.y1, p.x2, p.y2)
            if v is None:
                continue
            if d:
                d[0] += v
                d[1] += 1
                d[2] = frame
            else:
                self.desc[key] = [v.copy(), 1, frame]

    # ---- link
    COLS = ["frame", "track_id", "x1", "y1", "x2", "y2", "xw", "yw", "usable"]

    def _snapshot(self) -> tuple[dict, dict, int]:
        """Copies of rows, colour sums and the current frame, taken under the lock."""
        with self.lock:
            rows = {c: list(r) for c, r in self.rows.items() if r}
            desc = {k: v[0].copy() for k, v in self.desc.items()}
            now = self.now_frame
        return {c: pd.DataFrame(r, columns=self.COLS) for c, r in rows.items()}, desc, now

    def tracks(self) -> dict[str, pd.DataFrame]:
        return self._snapshot()[0]

    def relink(self) -> pd.DataFrame:
        """Re-link the tracks seen in the last `window_s`; returns camera, track_id, person for all tracks.

        Linking cost grows with the number of tracks, so a live run only re-links recent ones
        (`window_s` is longer than the longest hand-off gap). Tracks that ended before the
        window keep the person number they had.
        """
        all_cams, desc, now = self._snapshot()
        self._snap = (all_cams, now)
        cut = now - self.window_s * self.store.fps
        cams, frozen = {}, []
        for c, t in all_cams.items():
            last = t.groupby("track_id").frame.max()
            recent = last[last >= cut].index
            cams[c] = t[t.track_id.isin(recent)]
            old = t[~t.track_id.isin(recent)].groupby("track_id").frame.agg(["min", "max", "size"])
            for tid, r in old.iterrows():
                if (c, tid) in self.person_of:
                    frozen.append((c, tid, self.person_of[(c, tid)], r["min"], r["max"], r["size"]))
        cams = {c: t for c, t in cams.items() if len(t)}
        app = None
        if self.use_app:
            app = {k: v / (np.linalg.norm(v) or 1.0) for k, v in desc.items()}
        links = link(cams, self.store.fps, self.cfg, appearance=app) if cams else pd.DataFrame(
            columns=["camera", "track_id", "global_id", "start", "end", "n"])
        frozen_df = pd.DataFrame(frozen, columns=["camera", "track_id", "person", "start", "end", "n"])
        if links.empty:
            return frozen_df
        # Stable numbers: each new group takes the number most of its tracks (by boxes) already had.
        votes = []
        for g, rows in links.groupby("global_id"):
            c = Counter()
            for r in rows.itertuples():
                old = self.person_of.get((r.camera, r.track_id))
                if old is not None:
                    c[old] += int(r.n)
            for old, w in c.items():
                votes.append((w, g, old))
        number, taken = {}, set()
        for w, g, old in sorted(votes, key=lambda v: (-v[0], v[1], v[2])):
            if g not in number and old not in taken:
                number[g] = old
                taken.add(old)
        first_start = links.groupby("global_id").start.min()
        for g in first_start.sort_values(kind="stable").index:
            if g not in number:
                number[g] = self.next_person
                self.next_person += 1
        links["person"] = links.global_id.map(number)
        for r in links.itertuples():
            self.person_of[(r.camera, r.track_id)] = int(r.person)
        return pd.concat([links.drop(columns="global_id"), frozen_df], ignore_index=True)

    # ---- summarise
    def state(self, links: pd.DataFrame | None = None) -> dict:
        """Store summary for the dashboard, from the same snapshot as the last relink()."""
        if links is None or self._snap is None:
            links = self.relink()
        cams, now = self._snap
        fps = self.store.fps
        if not cams or links.empty:
            return {"t": round(now / fps, 1), "people_now": [], "unique_people": 0, "people": [],
                    "camera_people_now": {}, "camera_moves": {"cameras": self.store.cameras, "matrix": []}}
        allr = pd.concat([t.assign(camera=c) for c, t in cams.items()], ignore_index=True)
        allr = allr.merge(links[["camera", "track_id", "person"]], on=["camera", "track_id"])
        allr["sec"] = (allr.frame // fps).astype(int)

        # people table: seen time, first / last time, cameras in order of first sight
        people = []
        for pid, g in allr.groupby("person"):
            seen_s = g.sec.nunique()
            if seen_s < MIN_SEEN_S:
                continue
            first = g.groupby("camera").frame.min().sort_values()
            people.append({"id": int(pid), "first_s": round(g.frame.min() / fps, 1),
                           "last_s": round(g.frame.max() / fps, 1), "seen_s": int(seen_s),
                           "cameras": [c.replace("camera_", "") for c in first.index]})
        people.sort(key=lambda p: p["first_s"])
        visitors = {p["id"] for p in people}

        # now: seen in the last second, position = median of the cameras that see the feet
        recent = allr[(allr.frame >= now - fps) & allr.person.isin(visitors)]
        now_list = []
        for pid, g in recent.groupby("person"):
            u = g[g.usable] if g.usable.any() else g
            now_list.append({"id": int(pid), "x": round(float(u.xw.median()), 2), "y": round(float(u.yw.median()), 2),
                             "cameras": sorted(c.replace("camera_", "") for c in g.camera.unique())})
        trails = {}
        tr = allr[(allr.frame >= now - TRAIL_S * fps) & allr.usable & allr.person.isin([p["id"] for p in now_list])]
        for pid, g in tr.groupby("person"):
            pts = g.groupby("sec")[["xw", "yw"]].median()
            trails[int(pid)] = [[round(float(x), 2), round(float(y), 2)] for x, y in pts.to_numpy()]

        # camera moves: for each visitor, the order in which cameras first saw them
        names = self.store.cameras
        idx = {c: i for i, c in enumerate(names)}
        m = [[0] * len(names) for _ in names]
        for p in people:
            seq = ["camera_" + c for c in p["cameras"]]
            for a, b in zip(seq, seq[1:]):
                m[idx[a]][idx[b]] += 1
        cam_now = {c.replace("camera_", ""): int(g.person.nunique())
                   for c, g in recent.groupby("camera")}
        return {"t": round(now / fps, 1), "people_now": now_list, "trails": trails,
                "unique_people": len(people), "people": people, "camera_people_now": cam_now,
                "camera_moves": {"cameras": [c.replace("camera_", "") for c in names], "matrix": m},
                "heatmap": self._heatmap(allr), **self._zone_stats(allr)}

    def _zone_stats(self, allr: pd.DataFrame) -> dict:
        """Store zones: people now, visits, dwell and zone-to-zone moves of store-wide people
        (each person's floor position once per second, median of the cameras that see their feet)."""
        if not self.store.zones:
            return {}
        u = allr[allr.usable]
        pos = u.groupby(["sec", "person"])[["xw", "yw"]].median().reset_index()
        z = store_zone_analysis(pos, list(self.store.zones))
        return {"zones": z["rows"], "zone_moves": {"zones": z["names"], "matrix": z["moves"]}}

    def _heatmap(self, allr: pd.DataFrame) -> dict:
        """Person-seconds on the floor, counted once per store-wide person per second.

        Each person's floor position in each second is the median of all cameras that saw
        their feet, so someone seen by three cameras is not counted three times. Cells of
        HEAT_CELL_M; sent sparse as [row, col, seconds].
        """
        x0, y0, x1, y1 = self.store.floor_bounds
        cell = HEAT_CELL_M
        cols, rows = int(np.ceil((x1 - x0) / cell)), int(np.ceil((y1 - y0) / cell))
        u = allr[allr.usable]
        pos = u.groupby(["person", "sec"])[["xw", "yw"]].median()
        ci = np.floor((pos.xw.to_numpy() - x0) / cell).astype(int)
        ri = np.floor((pos.yw.to_numpy() - y0) / cell).astype(int)
        ok = (ci >= 0) & (ci < cols) & (ri >= 0) & (ri < rows)
        grid = np.zeros((rows, cols), int)
        np.add.at(grid, (ri[ok], ci[ok]), 1)
        nz = np.argwhere(grid > 0)
        return {"cell": cell, "x0": x0, "y0": y0, "rows": rows, "cols": cols,
                "total_s": int(grid.sum()), "cells": [[int(r), int(c), int(grid[r, c])] for r, c in nz]}


def store_zone_analysis(pos: pd.DataFrame, zones: list) -> dict:
    """Zones of a store from people's floor positions once per second.

    `pos`: columns sec, person, xw, yw (one row per person per second). Uses the same
    2 s enter / 2 s leave rule as camera zones (zones.py), at 1 update per second.
    Shared by the live store state and by evaluate_store_zones.py (ground truth goes
    through exactly the same code). Returns names, rows (now / visits / stays per zone),
    moves (zone-to-zone matrix), visits (table) and occupancy (people per zone per second).
    """
    names = [z.name for z in zones]
    zt = ZoneTracker(zones, 1.0, ZoneConfig())
    occ_rows = []
    if len(pos):
        by_sec = {int(sec): g for sec, g in pos.groupby("sec")}
        for sec in range(min(by_sec), max(by_sec) + 1):
            g = by_sec.get(sec)
            pts = {} if g is None else {int(p): (float(x), float(y)) for p, x, y in zip(g.person, g.xw, g.yw)}
            zt.update(sec, pts)
            o = zt.occupancy()
            occ_rows.append([sec] + [o[n] for n in names])
    occupancy_now = zt.occupancy()                # visits still open = people in the zone now
    zt.finish()
    visits = pd.DataFrame([v.__dict__ for v in zt.visits],
                          columns=["track_id", "zone", "start_frame", "end_frame", "dwell_s", "truncated"])
    rows = []
    for n in names:
        v = visits[visits.zone == n]
        rows.append({"zone": n, "now": int(occupancy_now.get(n, 0)), "visits": int(len(v)),
                     "mean_dwell_s": round(float(v.dwell_s.mean()), 1) if len(v) else None,
                     "median_dwell_s": round(float(v.dwell_s.median()), 1) if len(v) else None})
    idx = {n: i for i, n in enumerate(names)}
    m = [[0] * len(names) for _ in names]
    for (a, b), k in transition_counts(visits).items():
        m[idx[a]][idx[b]] += k
    occupancy = pd.DataFrame(occ_rows, columns=["sec", *names]).set_index("sec")
    return {"names": names, "rows": rows, "moves": m, "visits": visits, "occupancy": occupancy}


# ---- linking in its own process (run_store.py) ---------------------------------
# A linking thread shares the GIL with detection: its pandas work starved the detector
# (NMS time-outs, video falling behind). In a separate process it cannot.

def take_delta(lk: "StoreLinker", sent: dict) -> tuple[dict, dict, int]:
    """New rows per camera since the last call (positions in `sent`), all colour sums, current frame."""
    with lk.lock:
        delta = {}
        for c, r in lk.rows.items():
            k = sent.get(c, 0)
            if len(r) > k:
                delta[c] = r[k:]
                sent[c] = len(r)
        desc = {k: (v[0].copy(), v[1], v[2]) for k, v in lk.desc.items()}
        return delta, desc, lk.now_frame


def ingest(lk: "StoreLinker", delta: dict, desc: dict, now: int) -> None:
    with lk.lock:
        for c, r in delta.items():
            lk.rows.setdefault(c, []).extend(r)
        lk.desc = {k: [v[0], v[1], v[2]] for k, v in desc.items()}
        lk.now_frame = max(lk.now_frame, now)


def link_worker(store_path: str, cameras: list[str], appearance: bool, api_url: str | None,
                run_id: str | None, inbox, outbox) -> None:
    """Process: receive deltas, re-link, post the store state; on 'final' return links and state.

    After every re-link it also sends ("numbers", {(camera, track): person}) for tracks seen in
    the last 30 s, so the detection process can label boxes in the live camera preview.
    """
    import queue
    import time

    from retailflow.sender import StoreApiClient

    store = load_store(store_path)
    store.cameras = cameras
    lk = StoreLinker(store, appearance=appearance)
    api = StoreApiClient.attach(api_url, store.store_id, run_id) if api_url and run_id else None
    link_ms: list[float] = []
    while True:
        msgs = [inbox.get()]
        while True:                                   # catch up: merge everything queued
            try:
                msgs.append(inbox.get_nowait())
            except queue.Empty:
                break
        final = False
        for kind, delta, desc, now in msgs:
            ingest(lk, delta, desc, now)
            final = final or kind == "final"
        t = time.perf_counter()
        links = lk.relink()
        state = lk.state(links)
        link_ms.append((time.perf_counter() - t) * 1000)
        if final:
            outbox.put(("final", {"state": state, "links": links[["camera", "track_id", "person", "start", "end", "n"]],
                                  "link_ms": link_ms}))
            return
        recent = links[links.end >= lk.now_frame - 30 * store.fps]
        outbox.put(("numbers", {(r.camera, int(r.track_id)): int(r.person) for r in recent.itertuples()}))
        if api:
            api.state(state)
        print(f"t={state['t']:6.1f}s  people now {len(state['people_now']):3d}  "
              f"visitors {state['unique_people']:4d}  link {link_ms[-1]:.0f} ms", flush=True)
