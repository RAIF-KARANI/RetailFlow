"""Evaluate two-camera counting of the Promod door (docs/11_MULTI_CAMERA.md).

    python tools/sync_caviar_views.py      # once: clock time of frame 0, both views
    python evaluate_multicam.py            # -> outputs/multicam_results.csv

Per clip, both views are put on the recording clock (outputs/caviar_sync.csv) and
only the time span covered by BOTH views is scored.
Truth: the front camera's ground-truth crossings of the shop threshold (the same
truth as in Phases 2-6). Predictions come from the final system's saved tracks:
  front only        front camera, shop threshold line
  corridor only     corridor camera, door line from the site file
  union             both, matched pairs counted once, unmatched kept
  intersection      only crossings seen by both cameras
"""

from __future__ import annotations

import argparse

import pandas as pd

from evaluate import RUNS_DIR, VIRTUAL, FINAL
from retailflow.config import load_scene
from retailflow.counting import CountingConfig, count_tracks
from retailflow.evaluate import TRUTH_CFG
from retailflow.groundtruth import caviar_sequence
from retailflow.multicam import fuse, load_site
from retailflow.profiles import get_profile

POLICIES = ["front", "corridor", "union", "intersection"]


def crossings(table, line, cfg, offset_s: float, f0: int, f1: int, fps: float = 25.0):
    """(clock-relative time, direction) of crossings inside front frames [f0, f1]."""
    ev = count_tracks(table, line, cfg)
    t = ev.frame / fps + offset_s
    keep = (t >= f0 / fps) & (t <= f1 / fps)
    return list(zip(t[keep].round(3), ev.direction[keep]))


def run_dir(camera: str, clip: str, config: str):
    tracker, suffix = VIRTUAL[config]
    return RUNS_DIR / get_profile(load_scene(f"scenes/{camera}.json").profile).run_name(tracker, suffix) / clip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=FINAL, choices=list(VIRTUAL))
    p.add_argument("--window", type=float, default=None, help="override the site's match window (s)")
    args = p.parse_args()

    site = load_site("scenes/sites/caviar_promod.json")
    window = args.window or site.match_window_s
    front_line = load_scene("scenes/caviar_front.json").line
    door_line = next(c.line for c in site.cameras if c.camera_id == "caviar_cor")
    sync = pd.read_csv("outputs/caviar_sync.csv")
    pred_cfg = CountingConfig()
    rows, dts = [], []
    for r in sync.itertuples():
        gf = caviar_sequence(f"data/caviar/{r.clip}front").load_gt()
        # Common time span, in front-view frames: cor frame k = front frame k + offset.
        f0 = max(0, -r.offset_frames)
        f1 = min(r.front_frames - 1, r.cor_frames - 1 - r.offset_frames, int(gf.frame.max()))
        off_cor = -r.offset_frames / 25.0          # corridor times expressed on the front timeline
        truth = crossings(gf, front_line, TRUTH_CFG, 0.0, f0, f1)
        tf = pd.read_csv(run_dir("caviar_front", f"{r.clip}front", args.config) / "tracks.csv")
        tc = pd.read_csv(run_dir("caviar_cor", f"{r.clip}cor", args.config) / "tracks.csv")
        pf = crossings(tf, front_line, pred_cfg, 0.0, f0, f1)
        pc = crossings(tc, door_line, pred_cfg, off_cor, f0, f1)
        fz = fuse(pf, pc, window)
        dts += fz["dt"]
        true = {d: sum(1 for _, x in truth if x == d) for d in ("in", "out")}
        row = {"clip": r.clip, "true_in": true["in"], "true_out": true["out"]}
        for pol, key in [("front", "a"), ("corridor", "b"), ("union", "union"), ("intersection", "intersection")]:
            pi, po = fz["in"][key], fz["out"][key]
            row[f"{pol}_in"], row[f"{pol}_out"] = pi, po
            row[f"{pol}_err"] = abs(pi - true["in"]) + abs(po - true["out"])
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv("outputs/multicam_results.csv", index=False)
    total = d.true_in.sum() + d.true_out.sum()
    print(f"{len(d)} clips, {total} true crossings ({d.true_in.sum()} in / {d.true_out.sum()} out), "
          f"config {args.config}, window {window} s")
    for pol in POLICIES:
        err = d[f"{pol}_err"].sum()
        exact = (d[f"{pol}_err"] == 0).sum()
        print(f"  {pol:13s} pred {d[f'{pol}_in'].sum():3d} in / {d[f'{pol}_out'].sum():3d} out   "
              f"abs err {err:3d}   WAPE {100 * err / total:5.1f}%   exact clips {exact}/{len(d)}")
    if dts:
        s = pd.Series(dts)
        print(f"  matched pairs {len(s)}: time difference corridor - front median {s.median():+.2f} s, "
              f"p90 |dt| {s.abs().quantile(0.9):.2f} s")


if __name__ == "__main__":
    main()
