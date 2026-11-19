"""Ground-truth evaluation: true vs predicted IN/OUT counts, and tracking quality.

TRUE counts
-----------
Ground truth has no "IN/OUT" labels, only boxes per person per frame. We get
the true counts by running the SAME line logic on the ground-truth feet points,
with the scene's band (TRUTH_CFG). This definition never changes, so every
experiment is compared with the same truth.

PREDICTED counts
----------------
Re-counted offline from the pipeline's saved tracks.csv for each counting
variant (centre vs feet point, simple line vs band, ...). No detector re-run needed.

Error
-----
Per sequence: abs_err = |pred_in - true_in| + |pred_out - true_out|.
Per group: WAPE = sum(abs_err) / sum(true_in + true_out). Unlike a net total,
this does not let 3 missed INs cancel 3 extra INs.

Tracking quality
----------------
IDF1 (how consistently each person keeps one ID), MOTA, and ID switches, via
motmetrics, with IoU >= 0.5 as a match.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import motmetrics as mm
import numpy as np
import pandas as pd

from retailflow.config import load_scene, scene_path_for_sequence
from retailflow.counting import CountingConfig, count_tracks
from retailflow.groundtruth import Sequence

# The fixed definition of truth: GT feet point + scene band, every crossing counts.
TRUTH_CFG = CountingConfig(point="feet", use_band=True, min_track_age=0, once_per_direction=False)

# Counting variants applied to the SAME saved tracks.
VARIANTS: dict[str, CountingConfig] = {
    "centre_simple": CountingConfig(point="centre", use_band=False),   # classic baseline
    "feet_simple": CountingConfig(point="feet", use_band=False),
    "feet_band": CountingConfig(point="feet", use_band=True),
    "feet_band_age": CountingConfig(point="feet", use_band=True, min_track_age=10),
    "feet_band_age_once": CountingConfig(point="feet", use_band=True, min_track_age=10,
                                         once_per_direction=True),
}

# Density groups by mean people per frame (from ground truth).
DENSITY_BINS = [(0, 2.5, "quiet"), (2.5, 15, "medium"), (15, float("inf"), "busy")]


def density_group(mean_people: float) -> str:
    for lo, hi, name in DENSITY_BINS:
        if lo <= mean_people < hi:
            return name
    return "busy"


# ------------------------------------------------------------- tracking metrics
def iou_distance(gt: np.ndarray, pr: np.ndarray, min_iou: float = 0.5) -> np.ndarray:
    """1 - IoU between every GT and predicted box (x1,y1,x2,y2); NaN = not matchable.

    Written here because motmetrics' own iou_matrix uses np.asfarray, which
    was removed in NumPy 2.0.
    """
    if len(gt) == 0 or len(pr) == 0:
        return np.empty((len(gt), len(pr)))
    ix1 = np.maximum(gt[:, None, 0], pr[None, :, 0])
    iy1 = np.maximum(gt[:, None, 1], pr[None, :, 1])
    ix2 = np.minimum(gt[:, None, 2], pr[None, :, 2])
    iy2 = np.minimum(gt[:, None, 3], pr[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_g = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    area_p = (pr[:, 2] - pr[:, 0]) * (pr[:, 3] - pr[:, 1])
    iou = inter / (area_g[:, None] + area_p[None, :] - inter)
    dist = 1.0 - iou
    dist[iou < min_iou] = np.nan
    return dist


def tracking_metrics(gt: pd.DataFrame, tracks: pd.DataFrame, n_frames: int) -> dict:
    """IDF1 / MOTA / ID switches over all frames of one sequence."""
    acc = mm.MOTAccumulator(auto_id=True)
    cols = ["x1", "y1", "x2", "y2"]
    gt_by = dict(tuple(gt.groupby("frame")))
    pr_by = dict(tuple(tracks.groupby("frame")))
    empty = pd.DataFrame(columns=["track_id", *cols])
    for f in range(n_frames):
        g = gt_by.get(f, empty)
        p = pr_by.get(f, empty)
        acc.update(g["track_id"].tolist(), p["track_id"].tolist(),
                   iou_distance(g[cols].to_numpy(float), p[cols].to_numpy(float)))
    mh = mm.metrics.create()
    s = mh.compute(acc, metrics=["idf1", "mota", "num_switches", "precision", "recall"], name="x")
    return {k: round(float(s[k].iloc[0]), 4) for k in s.columns}


# ------------------------------------------------------------- per-run evaluation
def load_tracks(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "tracks.csv")


def evaluate_run(run_dir: Path, seq: Sequence, variants: dict[str, CountingConfig]) -> list[dict]:
    """Evaluate one (config, sequence) run under every counting variant."""
    summary = json.loads((run_dir / "summary.json").read_text())
    scene = load_scene(scene_path_for_sequence(seq.name))
    gt = seq.load_gt()
    tracks = load_tracks(run_dir)
    n_frames = summary["frames"]

    truth = count_tracks(gt, scene.line, TRUTH_CFG)
    true_in = int((truth["direction"] == "in").sum())
    true_out = int((truth["direction"] == "out").sum())

    # Tracking metrics do not depend on the counting variant: compute once, cache.
    cache = run_dir / "tracking_metrics.json"
    if cache.exists():
        tm = json.loads(cache.read_text())
    else:
        tm = tracking_metrics(gt, tracks, n_frames)
        cache.write_text(json.dumps(tm, indent=2))

    mean_people = len(gt) / max(n_frames, 1)
    rows = []
    for vname, vcfg in variants.items():
        pred = count_tracks(tracks, scene.line, vcfg)
        pred_in = int((pred["direction"] == "in").sum())
        pred_out = int((pred["direction"] == "out").sum())
        abs_err = abs(pred_in - true_in) + abs(pred_out - true_out)
        true_total = true_in + true_out
        rows.append({
            "config": summary.get("config", run_dir.parent.name),
            "variant": vname,
            "sequence": seq.name,
            "dataset": seq.dataset,
            "camera": scene.camera_id,
            "model": summary.get("model"), "tracker": summary.get("tracker"), "imgsz": summary.get("imgsz"),
            "frames": n_frames,
            "people": int(gt["track_id"].nunique()),
            "mean_people": round(mean_people, 2),
            "density": density_group(mean_people),
            "true_in": true_in, "true_out": true_out,
            "pred_in": pred_in, "pred_out": pred_out,
            "abs_err": abs_err,
            "pct_err": round(100 * abs_err / true_total, 1) if true_total else None,
            "pred_ids": int(tracks["track_id"].nunique()),
            **tm,
            "avg_proc_ms": summary.get("avg_proc_ms"),
            "throughput_fps": summary.get("throughput_fps"),
            "counting": json.dumps(asdict(vcfg)),
        })
    return rows


def summarise(results: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Group results: total true/pred counts, WAPE, exact-count rate, mean IDF1."""
    g = results.assign(true_total=results.true_in + results.true_out,
                       exact=(results.abs_err == 0)).groupby(by)
    out = g.agg(sequences=("sequence", "count"), true_crossings=("true_total", "sum"),
                abs_err=("abs_err", "sum"), exact_seqs=("exact", "sum"),
                idf1=("idf1", "mean"), ms_per_frame=("avg_proc_ms", "mean"))
    out["wape_pct"] = (100 * out.abs_err / out.true_crossings.where(out.true_crossings > 0)).round(1)
    out["idf1"] = out["idf1"].round(3)
    out["ms_per_frame"] = out["ms_per_frame"].round(1)
    return out.reset_index()
