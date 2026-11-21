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
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from retailflow.appearance import MIN_H_PX, box_descriptor
from retailflow.mtmc import MTMCConfig, floor_points, link

DESC_EVERY_S = 1.0      # one colour sample per track per second at most
DESC_MAX = 6            # ... and at most this many per track
MIN_SEEN_S = 2.0        # a person counts as a visitor after this many seconds seen
TRAIL_S = 10            # trail length shown on the floor map
WINDOW_S = 120          # only tracks seen in the last WINDOW_S are re-linked; older people keep their number


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

    def to_api(self) -> dict:
        """What the dashboard needs to draw the store (no video paths)."""
        return {"store_id": self.store_id, "description": self.description, "cameras": self.cameras,
                "camera_xy": {c: [round(x, 2), round(y, 2)] for c, (x, y) in self.camera_xy.items()},
                "floor_bounds": list(self.floor_bounds)}


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
    return StoreConfig(d["store_id"], d.get("description", ""), scene, cams, H, pos,
                       tuple(d["floor_bounds"]), d.get("frame_h", 1080), d.get("fps", 30.0), d)


class StoreLinker:
    """Collects live tracks of all cameras and keeps stable store-wide person numbers."""

    def __init__(self, store: StoreConfig, cfg: MTMCConfig = MTMCConfig(), appearance: bool = True,
                 window_s: float = WINDOW_S):
        self.store, self.cfg, self.use_app, self.window_s = store, cfg, appearance, window_s
        self.rows: dict[str, list] = {c: [] for c in store.cameras}
        self.desc: dict[tuple, list] = {}          # (camera, track) -> [sum vector, count, last time]
        self.person_of: dict[tuple, int] = {}      # (camera, track) -> stable person number
        self.next_person = 1
        self.now_frame = 0

    # ---- collect
    def add(self, cam: str, frame: int, people, image: np.ndarray | None = None) -> None:
        self.now_frame = max(self.now_frame, frame)
        if not people:
            return
        t = pd.DataFrame([(frame, p.track_id, p.x1, p.y1, p.x2, p.y2) for p in people],
                         columns=["frame", "track_id", "x1", "y1", "x2", "y2"])
        t = floor_points(t, self.store.homographies[cam], self.store.frame_h, self.cfg)
        self.rows[cam].extend(t[["frame", "track_id", "x1", "y1", "x2", "y2", "xw", "yw", "usable"]]
                              .itertuples(index=False, name=None))
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
    def tracks(self) -> dict[str, pd.DataFrame]:
        cols = ["frame", "track_id", "x1", "y1", "x2", "y2", "xw", "yw", "usable"]
        return {c: pd.DataFrame(r, columns=cols) for c, r in self.rows.items() if r}

    def relink(self) -> pd.DataFrame:
        """Re-link the tracks seen in the last `window_s`; returns camera, track_id, person for all tracks.

        Linking cost grows with the number of tracks, so a live run only re-links recent ones
        (`window_s` is longer than the longest hand-off gap). Tracks that ended before the
        window keep the person number they had.
        """
        all_cams = self.tracks()
        cut = self.now_frame - self.window_s * self.store.fps
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
            app = {k: v[0] / (np.linalg.norm(v[0]) or 1.0) for k, v in self.desc.items()}
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
        fps, now = self.store.fps, self.now_frame
        cams = self.tracks()
        if links is None:
            links = self.relink()
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
                "camera_moves": {"cameras": [c.replace("camera_", "") for c in names], "matrix": m}}
