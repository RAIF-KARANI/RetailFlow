"""Run detection + tracking configurations over every dataset sequence.

Each (config, sequence) run writes outputs/runs/<config>/<sequence>/ with
tracks.csv + summary.json. Runs that already have a summary.json are SKIPPED,
so the benchmark can be stopped and resumed (e.g. overnight).

Counting variants (centre vs feet point, band, min track age...) are NOT part
of this grid: they are applied offline by evaluate.py on the saved tracks,
which takes seconds instead of re-running the detector.

Examples:
  python benchmark.py --configs baseline
  python benchmark.py --configs baseline --datasets mot17 --max-frames 100
  python benchmark.py --list
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from retailflow.config import load_scene, scene_path_for_sequence
from retailflow.counting import CountingConfig
from retailflow.detector import PersonTracker
from retailflow.groundtruth import discover_sequences
from retailflow.pipeline import Pipeline
from retailflow.sources import FrameSource

RUNS_DIR = Path("outputs/runs")

# name -> detector/tracker settings. Phase 3 adds the model x imgsz x tracker grid.
CONFIGS: dict[str, dict] = {
    "baseline": {"model": "yolo26n.pt", "tracker": "bytetrack", "imgsz": 640, "conf": 0.1},
}

DATASET_ORDER = {"mot17": 0, "caviar": 1, "mot20": 2}


def run_config(name: str, cfg: dict, datasets: set[str] | None, only: set[str] | None,
               max_frames: int | None) -> None:
    seqs = [s for s in discover_sequences()
            if (datasets is None or s.dataset in datasets) and (only is None or s.name in only)]
    # MOT first, then CAVIAR corridor, then CAVIAR front.
    seqs.sort(key=lambda s: (DATASET_ORDER.get(s.dataset, 9), s.name.endswith("front"), s.name))
    print(f"== {name}: {len(seqs)} sequences, {cfg}")
    for i, seq in enumerate(seqs, 1):
        out = RUNS_DIR / name / seq.name
        if (out / "summary.json").exists():
            print(f"[{i}/{len(seqs)}] skip {seq.name} (done)")
            continue
        t0 = time.perf_counter()
        scene = load_scene(scene_path_for_sequence(seq.name))
        tracker = PersonTracker(cfg["model"], cfg["tracker"], cfg["imgsz"], cfg["conf"])  # fresh state
        summary = Pipeline(scene, tracker, CountingConfig(), out,
                           run_info={**cfg, "config": name, "sequence": seq.name, "dataset": seq.dataset}
                           ).run(FrameSource(seq.media, fps=seq.fps, max_frames=max_frames))
        print(f"[{i}/{len(seqs)}] {seq.name}: {summary['frames']} frames, "
              f"{summary['avg_proc_ms']} ms/frame, {time.perf_counter() - t0:.0f}s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--configs", nargs="+", default=["baseline"])
    p.add_argument("--datasets", nargs="+", default=None, help="caviar mot17 mot20")
    p.add_argument("--sequences", nargs="+", default=None)
    p.add_argument("--max-frames", type=int, default=None, help="for quick tests only")
    p.add_argument("--list", action="store_true", help="list configs and exit")
    args = p.parse_args()
    if args.list:
        for k, v in CONFIGS.items():
            print(k, v)
        return
    for name in args.configs:
        run_config(name, CONFIGS[name], set(args.datasets) if args.datasets else None,
                   set(args.sequences) if args.sequences else None, args.max_frames)


if __name__ == "__main__":
    main()
