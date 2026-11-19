"""Evaluate zones (occupancy / visits / dwell), heatmaps and the staff rule.

Uses the saved tracks of the final system (each camera's profile run), so no
detector is re-run.

Examples:
  python evaluate_zones.py --sweep           # choose zone debounce on dev (caviar_cor, MOT17-09)
  python evaluate_zones.py                   # zones + heatmaps with the chosen settings
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from evaluate import profile_run_dirs
from retailflow.analytics_eval import heatmaps, load_run, summarise_zones, zone_rows
from retailflow.config import load_scene, scene_path_for_sequence
from retailflow.evaluate import split_for
from retailflow.groundtruth import discover_sequences
from retailflow.heatmap import median_background, similarity
from retailflow.sources import FrameSource
from retailflow.staff import staff_tracks
from retailflow.static_filter import static_tracks
from retailflow.zones import ZoneConfig

OUT = Path("outputs")
FIGS = Path("docs/img")
CHOSEN = ZoneConfig()   # 2 s / 2 s, chosen by the dev sweep (docs/05 section 3)
FILTERS = ("none", "static", "static+staff")
ZONES_CONFIG = "final_phase3"   # see evaluate.VIRTUAL


def runs():
    """(run_dir, sequence, scene) for every sequence with a finished final-profile run."""
    seqs = {s.name: s for s in discover_sequences()}
    # Phase 4 numbers (docs/05) were measured on the Phase 3 runs (PyTorch, BoT-SORT with GMC).
    for run_dir in profile_run_dirs(seqs, ZONES_CONFIG):
        if (run_dir / "summary.json").exists():
            yield run_dir, seqs[run_dir.name], load_scene(scene_path_for_sequence(run_dir.name))


def background(seq, n=15, step=50) -> np.ndarray:
    """People-free background from the median of n frames, one every `step` frames."""
    frames = []
    for fr in FrameSource(str(seq.media), fps=seq.fps):
        if fr.index % step == 0:
            frames.append(fr.image)
        if len(frames) >= n:
            break
    return median_background(frames)


def sweep(cache) -> None:
    rows = []
    # Values in frames at CAVIAR's 25 fps (the sweep was first run in frames);
    # stored in seconds so MOT17-09 (30 fps) uses the same durations.
    for ef in (1, 3, 5, 10, 25, 50):
        for xf in (1, 6, 12, 25, 50, 100):
            cfg = ZoneConfig(ef / 25, xf / 25)
            for gt, tracks, n, seq, scene in cache:
                for r in zone_rows(gt, tracks, n, seq, scene, cfg):
                    rows.append({**r, "enter_frames": ef, "exit_frames": xf, "split": split_for(scene.camera_id)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "zones_sweep.csv", index=False)
    s = summarise_zones(df, ["enter_frames", "exit_frames", "split"])
    wide = s.pivot(index=["enter_frames", "exit_frames"], columns="split",
                   values=["visit_wape_pct", "person_s_wape_pct", "occ_mae"])
    wide.columns = [f"{sp}_{m}" for m, sp in wide.columns]
    print(wide.sort_values("dev_visit_wape_pct").reset_index().to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()

    cache = []
    for run_dir, seq, scene in runs():
        gt, tracks, n = load_run(run_dir, seq)
        cache.append((gt, tracks, n, seq, scene))
    zoned = [c for c in cache if c[4].zones]
    if args.sweep:
        sweep(zoned)
        return

    # ---- zones: no filter / static filter / static + staff rule
    rows = []
    for gt, tracks, n, seq, scene in zoned:
        for filt in FILTERS:
            if filt == "static+staff" and not scene.staff:
                continue
            for r in zone_rows(gt, tracks, n, seq, scene, CHOSEN,
                               staff="staff" in filt, static="static" in filt):
                rows.append({**r, "filter": filt, "split": split_for(scene.camera_id)})
    z = pd.DataFrame(rows)
    z.to_csv(OUT / "zones_results.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summarise_zones(z, ["split", "camera", "filter", "zone"]).to_string(index=False))
        print()
        print(summarise_zones(z, ["split", "camera", "filter"]).to_string(index=False))

    # ---- static filter safety check: how much REAL (ground-truth) presence would it remove?
    print("\nstatic filter applied to ground truth (real people wrongly removed):")
    lost = defaultdict(lambda: [0.0, 0.0])
    for gt, tracks, n, seq, scene in cache:
        drop = static_tracks(gt, seq.fps)
        lost[scene.camera_id][0] += gt.track_id.isin(drop).sum() / seq.fps
        lost[scene.camera_id][1] += len(gt) / seq.fps
    for cam, (a, b) in sorted(lost.items()):
        print(f"  {cam}: {a:.1f} of {b:.1f} person-seconds ({100 * a / b:.2f}%)")

    # ---- staff rule: what it flags
    print("\nstaff rule:")
    for gt, tracks, n, seq, scene in zoned:
        if scene.staff:
            pred = staff_tracks(tracks, scene.staff, scene.zones, seq.fps)
            true = staff_tracks(gt, scene.staff, scene.zones, seq.fps)
            if pred or true:
                print(f"  {seq.name}: GT ids flagged {sorted(true)}, predicted ids flagged {sorted(pred)}")

    # ---- heatmaps per camera (all clips of a camera summed), without and with filters
    hrows = []
    FIGS.mkdir(parents=True, exist_ok=True)
    for filt in ("none", "static+staff"):
        acc = defaultdict(lambda: [None, None, None])
        for gt, tracks, n, seq, scene in cache:
            hg, hp = heatmaps(gt, tracks, seq, scene, staff="staff" in filt, static="static" in filt)
            a = acc[scene.camera_id]
            if a[0] is None:
                a[0], a[1], a[2] = hg, hp, seq
            else:
                a[0].grid += hg.grid
                a[1].grid += hp.grid
        for cam, (hg, hp, seq) in sorted(acc.items()):
            sim = similarity(hp.smoothed(), hg.smoothed())
            hrows.append({"camera": cam, "split": split_for(cam), "filter": filt, **sim})
            if filt == "none" or cam not in ("caviar_front", "caviar_cor", "MOT20-02"):
                continue
            bg = background(seq)
            fig = np.hstack([hg.render(bg), hp.render(bg)])
            cv2.putText(fig, "ground truth", (8, fig.shape[0] - 10), 0, fig.shape[1] / 1500, (255, 255, 255), 2)
            cv2.putText(fig, "RetailFlow", (fig.shape[1] // 2 + 8, fig.shape[0] - 10), 0, fig.shape[1] / 1500,
                        (255, 255, 255), 2)
            if fig.shape[1] > 1600:
                fig = cv2.resize(fig, None, fx=1600 / fig.shape[1], fy=1600 / fig.shape[1])
            cv2.imwrite(str(FIGS / f"heatmap_{cam}.png"), fig)
    h = pd.DataFrame(hrows)
    h.to_csv(OUT / "heatmap_results.csv", index=False)
    print("\nheatmaps (prediction vs ground truth, per camera):")
    print(h.to_string(index=False))


if __name__ == "__main__":
    main()
