"""Score 'people in view' and 'total people' (retailflow/people.py) against ground truth.

    python evaluate_people.py [--run outputs/runs/m960_botsort_nogmc_trt]

Uses the saved tracks of a run (tracks.csv per sequence); no detector needed.
- people in view: GT boxes per frame vs tracked boxes per frame (static objects left out),
  both averaged per second of video; MAE over seconds.
- total people: GT person IDs vs track IDs lasting >= N s, for several N. N is chosen on DEV;
  TEST is reported only. Writes outputs/people_results.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from retailflow.config import load_scene, scene_path_for_sequence
from retailflow.evaluate import split_for
from retailflow.groundtruth import discover_sequences

MINS = [0, 1, 2, 3, 5]


def score_sequence(seq, run_dir: Path) -> dict:
    cam = load_scene(scene_path_for_sequence(seq.name)).camera_id
    gt, tr = seq.load_gt(), pd.read_csv(run_dir / "tracks.csv")
    static = set(json.loads((run_dir / "summary.json").read_text()).get("static_track_ids", []))
    tr = tr[~tr.track_id.isin(static)]
    n = int(max(gt.frame.max(), tr.frame.max())) + 1
    per_s = lambda s: s.groupby((s.index.to_numpy() // seq.fps).astype(int)).mean()
    g_now = per_s(gt.groupby("frame").size().reindex(range(n), fill_value=0))
    p_now = per_s(tr.groupby("frame").size().reindex(range(n), fill_value=0))
    r = {"seq": seq.name, "cam": cam, "split": split_for(cam), "secs": len(g_now),
         "now_true_mean": round(float(g_now.mean()), 2), "now_mae": round(float((p_now - g_now).abs().mean()), 2),
         "total_true": int(gt.track_id.nunique())}
    span = tr.groupby("track_id").frame.agg(lambda f: (f.max() - f.min()) / seq.fps)
    for m in MINS:
        r[f"total_ge{m}s"] = int((span >= m).sum())
    return r


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", default="outputs/runs/m960_botsort_nogmc_trt")
    args = p.parse_args()
    rows = [score_sequence(s, Path(args.run) / s.name) for s in discover_sequences("data")
            if (Path(args.run) / s.name / "tracks.csv").exists()]
    df = pd.DataFrame(rows)
    df.to_csv("outputs/people_results.csv", index=False)
    with pd.option_context("display.width", 250):
        print(df.to_string(index=False))
    for split, g in df.groupby("split"):
        out = {"seqs": len(g), "people_in_view_true_mean": round(float(g.now_true_mean.mean()), 2),
               "people_in_view_mae": round(float(g.now_mae.mean()), 2), "total_true": int(g.total_true.sum())}
        for m in MINS:
            pred = g[f"total_ge{m}s"]
            out[f"ge{m}s"] = f"{int(pred.sum())} ({100 * (pred - g.total_true).abs().sum() / g.total_true.sum():.1f}% WAPE)"
        print(split, json.dumps(out))


if __name__ == "__main__":
    main()
