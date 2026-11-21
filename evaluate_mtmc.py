"""Score multi-camera tracking (global IDs across cameras) against ground truth.

Uses each camera's saved tracks (tools/track_multicam.py), links them across
cameras with retailflow/mtmc.py, and compares with the dataset's ground truth,
where every person has one ID in all cameras.

Metrics (boxes matched per camera and frame at IoU >= 0.5):
- **global IDF1**: IDF1 over all cameras and frames together, so a person only
  counts as correctly identified if they keep ONE ID across cameras
  (the AI City Challenge multi-camera measure);
- the same with **no linking** (every camera's track is its own person), to show
  what linking adds;
- **people**: predicted global IDs vs true people; per-camera IDF1 for reference.

  python evaluate_mtmc.py                                    # SmartSpaces scene_071
  python evaluate_mtmc.py --sweep                            # try linking settings
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import motmetrics as mm
import numpy as np
import pandas as pd

from retailflow.appearance import track_descriptors
from retailflow.evaluate import iou_distance, tracking_metrics
from retailflow.mtmc import MTMCConfig, floor_points, link

COLS = ["x1", "y1", "x2", "y2"]


def load_gt(scene: Path) -> pd.DataFrame:
    g = pd.read_csv(scene / "ground_truth.txt", sep=" ", header=None,
                    names=["cam", "obj", "frame", "x", "y", "w", "h", "xw", "yw"])
    g["camera"] = g.cam.map(lambda c: f"camera_{c:04d}")
    return g.assign(x1=g.x, y1=g.y, x2=g.x + g.w, y2=g.y + g.h, track_id=g.obj)


def load_tracks(scene: Path, track_dir: Path, cfg: MTMCConfig) -> tuple[dict, dict, float]:
    cams, frames, fps = {}, {}, 30.0
    for d in sorted(track_dir.glob("camera_*")):
        if not (d / "tracks.csv").exists():
            continue
        s = json.loads((d / "summary.json").read_text())
        frames[d.name], fps = s["frames"], s["fps"]
        H = np.array(json.loads((scene / d.name / "calibration.json").read_text())["homography matrix"])
        cams[d.name] = floor_points(pd.read_csv(d / "tracks.csv"), H, 1080, cfg)
    return cams, frames, fps


def global_idf1(gt: pd.DataFrame, pred: pd.DataFrame, id_col: str, frames: dict, f0: int = 0) -> dict:
    """IDF1 etc. over every (camera, frame) at once, frames f0..n-1; ids must be global."""
    acc = mm.MOTAccumulator(auto_id=True)
    gb = {k: v for k, v in gt.groupby(["camera", "frame"])}
    pb = {k: v for k, v in pred.groupby(["camera", "frame"])}
    empty = pd.DataFrame(columns=["obj", id_col, *COLS])
    for cam, n in frames.items():
        for f in range(f0, n):
            g, p = gb.get((cam, f), empty), pb.get((cam, f), empty)
            acc.update(g["obj"].tolist(), p[id_col].tolist(),
                       iou_distance(g[COLS].to_numpy(float), p[COLS].to_numpy(float)))
    s = mm.metrics.create().compute(acc, metrics=["idf1", "idp", "idr", "recall", "precision", "num_switches"], name="x")
    return {k: round(float(s[k].iloc[0]), 4) for k in s.columns}


def appearance_for(scene: Path, cams: dict) -> dict:
    """{(camera, track_id): colour descriptor}, computed now and kept in memory only (never saved)."""
    app = {}
    for cam, t in cams.items():
        for tid, d in track_descriptors(str(scene / cam / "video.mp4"), t).items():
            app[(cam, tid)] = d
    return app


def evaluate(scene: Path, track_dir: Path, cfg: MTMCConfig, per_camera: bool = True,
             from_s: float = 0.0, to_s: float | None = None, app: dict | None = None,
             baseline: bool = True) -> dict:
    """Link on everything tracked (as a live system would), score only [from_s, to_s)."""
    cams, frames, fps = load_tracks(scene, track_dir, cfg)
    f0 = int(round(from_s * fps))
    if to_s is not None:
        frames = {c: min(n, int(round(to_s * fps))) for c, n in frames.items()}
    gt = load_gt(scene)
    gt = gt[gt.camera.isin(frames) & (gt.frame < gt.camera.map(frames)) & (gt.frame >= f0)]
    links = link(cams, fps, cfg, appearance=app)
    pred = pd.concat([t.assign(camera=c) for c, t in cams.items()], ignore_index=True)
    pred = pred.merge(links[["camera", "track_id", "global_id"]], on=["camera", "track_id"])
    pred["local_id"] = pred.camera.str[-4:].astype(int) * 100000 + pred.track_id   # no linking
    pred = pred[(pred.frame >= f0) & (pred.frame < pred.camera.map(frames))]
    seen = pred.groupby("global_id").frame.size()     # boxes per person in the window
    out = {"cameras": len(frames), "window_s": [round(f0 / fps, 1), round(min(frames.values()) / fps, 1)],
           "true_people": int(gt.obj.nunique()), "tracklets": len(links),
           "pred_people": int(pred.global_id.nunique()),
           "pred_people_10s": int(seen.ge(10 * fps).sum()),
           "linked": global_idf1(gt, pred, "global_id", frames, f0),
           "no_linking": global_idf1(gt, pred, "local_id", frames, f0) if baseline else None,
           "appearance": app is not None, "config": asdict(cfg)}
    if per_camera:
        per = []
        for cam, n in frames.items():
            g = gt[gt.camera == cam].assign(frame=lambda d: d.frame - f0)
            t = pred[pred.camera == cam].assign(frame=lambda d: d.frame - f0)
            m = tracking_metrics(g, t[["frame", "track_id", *COLS]], n - f0)
            per.append({"camera": cam, **m})
        out["per_camera"] = per
        out["mean_camera_idf1"] = round(float(np.mean([p["idf1"] for p in per])), 4)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scene-dir", default="data/smartspaces/scene_071")
    p.add_argument("--tracks", default=None, help="default outputs/mtmc/<scene>/tracks")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--from-s", type=float, default=0.0, help="score from this second (dev: 0-120, test: 120-)")
    p.add_argument("--to-s", type=float, default=None)
    p.add_argument("--name", default="mtmc_results", help="results file name in the tracks' parent folder")
    p.add_argument("--link-m", type=float, default=None)
    p.add_argument("--gap-s", type=float, default=None)
    p.add_argument("--appearance", action="store_true", help="also use colour descriptors (in memory only)")
    p.add_argument("--app-thr", type=float, default=None)
    p.add_argument("--app-veto", type=float, default=None)
    p.add_argument("--long-gap-s", type=float, default=None)
    args = p.parse_args()
    scene = Path(args.scene_dir)
    track_dir = Path(args.tracks or f"outputs/mtmc/{scene.name}/tracks")
    out_dir = track_dir.parent

    app = None
    if args.appearance:
        cams, _, _ = load_tracks(scene, track_dir, MTMCConfig())
        app = appearance_for(scene, cams)
        print(f"colour descriptors for {len(app)} camera tracks (in memory)", flush=True)

    if args.sweep and app is not None:
        rows = []
        for thr in (0.3, 0.35, 0.4):
            for veto in (None, 0.6):
                for lg in (10.0, 30.0):
                    cfg = replace(MTMCConfig(), app_thr=thr, app_veto=veto, long_gap_s=lg)
                    r = evaluate(scene, track_dir, cfg, per_camera=False, app=app, baseline=False)
                    rows.append({"app_thr": thr, "app_veto": veto, "long_gap_s": lg, "idf1": r["linked"]["idf1"],
                                 "people": r["pred_people"], "people_10s": r["pred_people_10s"]})
                    print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(out_dir / "mtmc_sweep_appearance.csv", index=False)
        return

    if args.sweep:
        rows = []
        for link_m in (0.5, 1.0, 1.5):
            for gap_s in (0.0, 2.0, 4.0):
                cfg = replace(MTMCConfig(), link_m=link_m, gap_s=gap_s)
                r = evaluate(scene, track_dir, cfg, per_camera=False)
                rows.append({"link_m": link_m, "gap_s": gap_s, "idf1": r["linked"]["idf1"],
                             "people": r["pred_people"], "people_10s": r["pred_people_10s"]})
                print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(out_dir / "mtmc_sweep.csv", index=False)
        return

    cfg = MTMCConfig()
    if args.link_m is not None:
        cfg = replace(cfg, link_m=args.link_m)
    if args.gap_s is not None:
        cfg = replace(cfg, gap_s=args.gap_s)
    for k in ("app_thr", "app_veto", "long_gap_s"):
        if getattr(args, k) is not None:
            cfg = replace(cfg, **{k: getattr(args, k)})
    r = evaluate(scene, track_dir, cfg, from_s=args.from_s, to_s=args.to_s, app=app)
    (out_dir / f"{args.name}.json").write_text(json.dumps(r, indent=2))
    print(json.dumps({k: v for k, v in r.items() if k != "per_camera"}, indent=2))
    print(pd.DataFrame(r["per_camera"]).to_string(index=False))


if __name__ == "__main__":
    main()
