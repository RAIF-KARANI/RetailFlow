"""Why are cross-camera IDs wrong? Splits (one person, many IDs) vs mixes (one ID, many people).

    python tools/diagnose_mtmc.py --tracks outputs/mtmc/scene_071/tracks --to-s 120

Matches every predicted box to a ground-truth box of the same camera and frame (IoU >= 0.5,
greedy), then per predicted store-wide ID and per true person:
- purity: share of an ID's matched boxes that belong to its main true person (1 = clean);
- splits: how many IDs a true person owns (IDs mostly matching them; 1 = perfect);
- ghost IDs: IDs with under half their boxes matching anyone (false detections, signs);
- for split persons: are the pieces apart in time (hand-off missed) or at the same time
  (two cameras' views not joined)?
Colour descriptors are computed in memory only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_mtmc import appearance_for, load_gt, load_tracks   # noqa: E402
from retailflow.mtmc import MTMCConfig, link                     # noqa: E402

COLS = ["x1", "y1", "x2", "y2"]


def match(gt: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """pred rows with the matched true person (obj) or -1."""
    out = []
    gb = dict(tuple(gt.groupby(["camera", "frame"])))
    for key, p in pred.groupby(["camera", "frame"]):
        g = gb.get(key)
        obj = np.full(len(p), -1)
        if g is not None:
            a, b = p[COLS].to_numpy(float), g[COLS].to_numpy(float)
            ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
            iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
            inter = ix * iy
            area = lambda q: (q[:, 2] - q[:, 0]) * (q[:, 3] - q[:, 1])
            iou = inter / (area(a)[:, None] + area(b)[None, :] - inter)
            used = set()
            for i, j in sorted(((i, j) for i in range(len(a)) for j in range(len(b)) if iou[i, j] >= 0.5),
                               key=lambda ij: -iou[ij]):
                if obj[i] == -1 and j not in used:
                    obj[i] = g.obj.iloc[j]
                    used.add(j)
        out.append(p.assign(obj=obj))
    return pd.concat(out, ignore_index=True)


def diagnose(m: pd.DataFrame, fps: float) -> dict:
    mm = m[m.obj >= 0]
    counts = mm.groupby(["person", "obj"]).size()
    main = counts.groupby(level=0).max()                 # boxes of each ID's main true person
    purity = float(main.sum() / counts.sum())
    # IDs OWNED per true person (an ID belongs to the person most of its boxes match); counting every
    # ID that touches a person would also count other people's IDs after a tracker mix-up
    owner = counts.reset_index(name="n").sort_values("n", ascending=False).drop_duplicates("person")
    matched = m.assign(ok=m.obj >= 0).groupby("person").ok.mean()
    owner = owner[owner.person.map(matched) >= 0.5]
    splits = owner.obj.value_counts()
    # pieces of each true person: same time or apart?
    same_time = apart = 0
    for obj, g in mm.groupby("obj"):
        spans = g.groupby("person").frame.agg(["min", "max"]).sort_values("min").to_numpy()
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                if spans[i][0] <= spans[j][1] and spans[j][0] <= spans[i][1]:
                    same_time += 1
                else:
                    apart += 1
    return {"boxes_matched_pct": round(100 * len(mm) / len(m), 1), "ids": int(m.person.nunique()),
            "true_people": int(mm.obj.nunique()), "purity_box_weighted": round(purity, 3),
            "ids_owned_per_true_person": round(float(splits.mean()), 1),
            "ghost_ids": int((matched < 0.5).sum()),
            "piece_pairs_same_time": same_time, "piece_pairs_apart": apart}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scene-dir", default="data/smartspaces/scene_071")
    p.add_argument("--tracks", default="outputs/mtmc/scene_071/tracks")
    p.add_argument("--to-s", type=float, default=120)
    p.add_argument("--no-colour", action="store_true")
    args = p.parse_args()
    scene = Path(args.scene_dir)
    cams, frames, fps = load_tracks(scene, Path(args.tracks), MTMCConfig())
    hi = int(args.to_s * fps)
    cams = {c: t[t.frame < hi] for c, t in cams.items()}
    app = None if args.no_colour else appearance_for(scene, cams)
    links = link(cams, fps, MTMCConfig(), appearance=app)
    pred = pd.concat([t.assign(camera=c) for c, t in cams.items()]).merge(
        links[["camera", "track_id", "global_id"]], on=["camera", "track_id"]).rename(columns={"global_id": "person"})
    gt = load_gt(scene)
    gt = gt[(gt.frame < hi) & gt.camera.isin(cams)]
    print(diagnose(match(gt, pred), fps))


if __name__ == "__main__":
    main()
