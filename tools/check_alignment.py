"""Check that ground-truth frame i matches decoded video frame i.

For each sequence, compare GT boxes at frame f with the system's tracked boxes
at frame f + k for k in -3..+3, and report the mean best-IoU per GT box. If the
labels are aligned with the decoded frames, the peak is at k = 0.

Run after a benchmark:  python tools/check_alignment.py --config baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retailflow.evaluate import iou_distance  # noqa: E402
from retailflow.groundtruth import discover_sequences  # noqa: E402

OFFSETS = range(-3, 4)
COLS = ["x1", "y1", "x2", "y2"]


def mean_best_iou(gt: pd.DataFrame, pr: pd.DataFrame, k: int) -> float:
    shifted = pr.assign(frame=pr["frame"] - k)  # predicted frame f+k is compared with GT frame f
    pr_by = dict(tuple(shifted.groupby("frame")))
    best = []
    for f, g in gt.groupby("frame"):
        p = pr_by.get(f)
        if p is None:
            continue
        d = iou_distance(g[COLS].to_numpy(float), p[COLS].to_numpy(float), min_iou=0.0)
        best.extend(1.0 - np.nanmin(d, axis=1))
    return float(np.mean(best)) if best else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="baseline")
    args = p.parse_args()
    rows = []
    for seq in discover_sequences():
        tracks_csv = Path("outputs/runs") / args.config / seq.name / "tracks.csv"
        if not tracks_csv.exists():
            continue
        gt, pr = seq.load_gt(), pd.read_csv(tracks_csv)
        scores = {k: mean_best_iou(gt, pr, k) for k in OFFSETS}
        rows.append({"sequence": seq.name, **{f"k={k}": round(v, 3) for k, v in scores.items()},
                     "peak_offset": max(scores, key=lambda k: scores[k])})
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(df.to_string(index=False))
    print("\npeak offset counts:", df["peak_offset"].value_counts().to_dict())


if __name__ == "__main__":
    main()
