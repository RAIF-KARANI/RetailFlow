"""Put the two CAVIAR views of every clip on one timeline (docs/11_MULTI_CAMERA.md).

    python tools/sync_caviar_views.py        # -> outputs/caviar_sync.csv

For each clip and view: read the on-screen clock of every frame
(retailflow/osd_clock.py), fit the clock time of frame 0, and report
    offset_frames = round((start_front - start_cor) * fps)
i.e. corridor frame k shows the same moment as front frame k - offset_frames.

Check without any ground truth: the reading must be consistent inside each clip
(few outliers, small residuals).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow "import retailflow"

from retailflow.osd_clock import fit_start, fmt_clock, load_templates, read_clock  # noqa: E402
from retailflow.sources import FrameSource  # noqa: E402

FPS = 25.0


def clip_start(path: Path, templates) -> dict:
    clock = [read_clock(f.image, templates) for f in FrameSource(str(path))]
    return fit_start(clock, FPS)


def main() -> None:
    templates = load_templates()
    clips = sorted({p.name[:-3] for p in Path("data/caviar").glob("*cor")})
    rows = []
    for clip in clips:
        cor = clip_start(Path(f"data/caviar/{clip}cor/{clip}cor.mpg"), templates)
        front = clip_start(Path(f"data/caviar/{clip}front/{clip}front.mpg"), templates)
        off = round((front["start_s"] - cor["start_s"]) * FPS)
        row = {"clip": clip, "cor_start": fmt_clock(cor["start_s"]), "front_start": fmt_clock(front["start_s"]),
               "offset_s": round(front["start_s"] - cor["start_s"], 3), "offset_frames": off,
               "cor_frames": cor["frames"], "front_frames": front["frames"],
               "cor_outliers": cor["outliers"], "front_outliers": front["outliers"],
               "cor_resid_p95": cor["resid_p95_s"], "front_resid_p95": front["resid_p95_s"],
               "cor_start_s": cor["start_s"], "front_start_s": front["start_s"]}
        print(row, flush=True)
        rows.append(row)
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/caviar_sync.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("-> outputs/caviar_sync.csv")


if __name__ == "__main__":
    main()
