"""Evaluation of zones (occupancy, visits, dwell), heatmaps and the staff rule.

Ground truth is treated exactly like a tracker output: the same ZoneTracker and
Heatmap code runs on the labelled feet points, so any difference comes from
detection and tracking, not from different definitions.

Frames are limited to the labelled range (some CAVIAR front clips are only
partly labelled), so unlabelled frames are never scored.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from retailflow.config import Scene
from retailflow.groundtruth import Sequence
from retailflow.heatmap import heatmap_from_table
from retailflow.pipeline import heatmap_cell_px
from retailflow.staff import staff_tracks
from retailflow.static_filter import static_tracks
from retailflow.zones import ZoneConfig, zone_analysis


def load_run(run_dir: Path, seq: Sequence) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """(gt, tracks, n_frames) restricted to the frames that are both decoded and labelled."""
    gt = seq.load_gt()
    tracks = pd.read_csv(run_dir / "tracks.csv")
    frames = json.loads((run_dir / "summary.json").read_text())["frames"]
    n = min(frames, int(gt.frame.max()) + 1) if len(gt) else frames
    return gt[gt.frame < n], tracks[tracks.frame < n], n


def apply_filters(table: pd.DataFrame, seq: Sequence, scene: Scene, staff: bool, static: bool) -> pd.DataFrame:
    """Drop staff and/or static tracks. Used on predictions AND ground truth alike,
    so both sides describe the same population ("customers")."""
    drop: set[int] = set()
    if staff and scene.staff:
        drop |= staff_tracks(table, scene.staff, scene.zones, seq.fps)
    if static:
        drop |= static_tracks(table, seq.fps)
    return table[~table.track_id.isin(drop)] if drop else table


def zone_rows(gt, tracks, n, seq: Sequence, scene: Scene, cfg: ZoneConfig,
              staff: bool = False, static: bool = False) -> list[dict]:
    """One row per zone: occupancy MAE, visits and dwell, predicted vs true."""
    gt = apply_filters(gt, seq, scene, staff, static=False)        # GT has no mannequins
    tracks = apply_filters(tracks, seq, scene, staff, static)
    tv, to = zone_analysis(gt, scene.zones, seq.fps, n, cfg)
    pv, po = zone_analysis(tracks, scene.zones, seq.fps, n, cfg)
    rows = []
    for z in scene.zones:
        t, p = tv[tv.zone == z.name], pv[pv.zone == z.name]
        rows.append({
            "sequence": seq.name, "camera": scene.camera_id, "zone": z.name, "frames": n,
            "true_occ_mean": round(to[z.name].mean(), 3), "pred_occ_mean": round(po[z.name].mean(), 3),
            "occ_mae": round((po[z.name] - to[z.name]).abs().mean(), 4),
            "true_visits": len(t), "pred_visits": len(p),
            "true_person_s": round(t.dwell_s.sum(), 1), "pred_person_s": round(p.dwell_s.sum(), 1),
            "true_mean_dwell": round(t.dwell_s.mean(), 2) if len(t) else np.nan,
            "pred_mean_dwell": round(p.dwell_s.mean(), 2) if len(p) else np.nan,
        })
    return rows


def summarise_zones(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Group totals. Visit and person-second errors use the WAPE idea (no cancelling)."""
    d = df.assign(visit_abs=(df.pred_visits - df.true_visits).abs(),
                  ps_abs=(df.pred_person_s - df.true_person_s).abs(),
                  occ_err_frames=df.occ_mae * df.frames)
    g = d.groupby(by).agg(zone_seqs=("zone", "size"), frames=("frames", "sum"),
                          occ_err_frames=("occ_err_frames", "sum"),
                          true_visits=("true_visits", "sum"), pred_visits=("pred_visits", "sum"),
                          visit_abs=("visit_abs", "sum"),
                          true_person_s=("true_person_s", "sum"), pred_person_s=("pred_person_s", "sum"),
                          ps_abs=("ps_abs", "sum")).reset_index()
    g["occ_mae"] = (g.occ_err_frames / g.frames).round(3)
    g["visit_wape_pct"] = (100 * g.visit_abs / g.true_visits.clip(lower=1)).round(1)
    g["person_s_wape_pct"] = (100 * g.ps_abs / g.true_person_s.clip(lower=1e-9)).round(1)
    g["true_mean_dwell"] = (g.true_person_s / g.true_visits.clip(lower=1)).round(1)
    g["pred_mean_dwell"] = (g.pred_person_s / g.pred_visits.clip(lower=1)).round(1)
    return g.drop(columns=["occ_err_frames"])


def heatmaps(gt, tracks, seq: Sequence, scene: Scene, staff: bool = False, static: bool = False):
    """(true_grid, pred_grid) person-seconds on the scene's grid for one sequence."""
    gt = apply_filters(gt, seq, scene, staff, static=False)
    tracks = apply_filters(tracks, seq, scene, staff, static)
    size, cell = (seq.width, seq.height), heatmap_cell_px(seq.width)
    hg = heatmap_from_table(gt, size, seq.fps, cell, scene.homography)
    hp = heatmap_from_table(tracks, size, seq.fps, cell, scene.homography)
    return hg, hp
