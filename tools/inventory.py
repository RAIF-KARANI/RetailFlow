"""Dataset inventory: frames, ground-truth coverage and crowd density per sequence.

For every discovered sequence this decodes the media to count REAL frames
(MPEG headers are often wrong), loads the ground truth with the common loader,
and reports how many people appear and how crowded it is. Output:
printed table + outputs/dataset_inventory.csv.

Run:  python tools/inventory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retailflow.groundtruth import Sequence, discover_sequences  # noqa: E402


def count_frames(seq: Sequence) -> int:
    """Number of frames actually readable from the sequence media."""
    if seq.media.is_dir():
        return len(list(seq.media.glob("*.jpg")))
    cap = cv2.VideoCapture(str(seq.media))
    n = 0
    while cap.grab():  # grab() decodes headers only, faster than read()
        n += 1
    cap.release()
    return n


def main() -> None:
    rows = []
    for seq in discover_sequences():
        gt = seq.load_gt()
        per_frame = gt.groupby("frame").size()
        rows.append({
            "sequence": seq.name,
            "dataset": seq.dataset,
            "resolution": f"{seq.width}x{seq.height}",
            "fps": seq.fps,
            "video_frames": count_frames(seq),
            "gt_last_frame": int(gt["frame"].max()) if len(gt) else -1,
            "people": gt["track_id"].nunique(),
            # mean over ALL video frames (frames with nobody count as 0)
            "mean_per_frame": round(len(gt) / max(count_frames(seq), 1), 2),
            "max_per_frame": int(per_frame.max()) if len(per_frame) else 0,
        })
    df = pd.DataFrame(rows)
    out = Path("outputs") / "dataset_inventory.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(df.to_string(index=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
