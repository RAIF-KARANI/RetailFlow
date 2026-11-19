"""Run detection + tracking configurations over every dataset sequence.

Each (config, sequence) run writes outputs/runs/<config>/<sequence>/ with
tracks.csv + summary.json. Runs that already have a summary.json are SKIPPED,
so the benchmark can be stopped and resumed (e.g. overnight).

Counting variants (centre vs feet point, band, min track age...) are NOT part
of this grid: they are applied offline by evaluate.py on the saved tracks,
which takes seconds instead of re-running the detector.

Config names: <model size><imgsz>_<tracker>, e.g. n1280_bytetrack, s960_botsort_reid.

Examples:
  python benchmark.py --configs baseline
  python benchmark.py --configs n960_bytetrack s1280_botsort_reid
  python benchmark.py --plan onefactor          # automated 3-stage grid (Phase 3)
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

from retailflow.config import load_scene, scene_path_for_sequence
from retailflow.counting import CountingConfig
from retailflow.detector import PersonTracker
from retailflow.evaluate import VARIANTS, evaluate_run, split_for, summarise
from retailflow.groundtruth import discover_sequences
from retailflow.pipeline import Pipeline
from retailflow.sources import FrameSource

RUNS_DIR = Path("outputs/runs")
SELECTION_FILE = Path("outputs/grid_selection.json")
DATASET_ORDER = {"mot17": 0, "caviar": 1, "mot20": 2}

# The Phase 2 baseline keeps its own name so its results stay reproducible.
ALIASES = {"baseline": "n640_bytetrack"}

# Selection rule for the automated grid (fixed BEFORE seeing results):
SELECT_VARIANT = "feet_band"   # counting method used to score detector/tracker configs
TIE_ERRORS = 1                 # configs within this many dev counting errors are "tied" -> faster wins


def parse_config(name: str) -> dict:
    """'n1280_botsort_reid' -> {'model': 'yolo26n.pt', 'imgsz': 1280, 'tracker': 'botsort_reid', 'conf': 0.1}."""
    m = re.fullmatch(r"([nsmlx])(\d+)_(\w+)", ALIASES.get(name, name))
    if not m:
        raise ValueError(f"bad config name {name!r}; expected e.g. n640_bytetrack")
    size, imgsz, tracker = m.groups()
    return {"model": f"yolo26{size}.pt", "imgsz": int(imgsz), "tracker": tracker, "conf": 0.1}


def run_config(name: str, datasets: set[str] | None = None, only: set[str] | None = None,
               max_frames: int | None = None) -> None:
    cfg = parse_config(name)
    seqs = [s for s in discover_sequences()
            if (datasets is None or s.dataset in datasets) and (only is None or s.name in only)]
    # MOT17 first, then CAVIAR corridor, CAVIAR front, then MOT20.
    seqs.sort(key=lambda s: (DATASET_ORDER.get(s.dataset, 9), s.name.endswith("front"), s.name))
    print(f"== {name}: {len(seqs)} sequences, {cfg}", flush=True)
    for i, seq in enumerate(seqs, 1):
        out = RUNS_DIR / name / seq.name
        if (out / "summary.json").exists():
            print(f"[{i}/{len(seqs)}] skip {seq.name} (done)", flush=True)
            continue
        t0 = time.perf_counter()
        scene = load_scene(scene_path_for_sequence(seq.name))
        tracker = PersonTracker(cfg["model"], cfg["tracker"], cfg["imgsz"], cfg["conf"])  # fresh state
        summary = Pipeline(scene, tracker, CountingConfig(), out,
                           run_info={**cfg, "config": name, "sequence": seq.name, "dataset": seq.dataset}
                           ).run(FrameSource(seq.media, fps=seq.fps, max_frames=max_frames))
        print(f"[{i}/{len(seqs)}] {seq.name}: {summary['frames']} frames, "
              f"{summary['avg_proc_ms']} ms/frame, {time.perf_counter() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ automated grid
def dev_score(name: str) -> dict:
    """Score one finished config on the DEV split with the selection counting variant."""
    seqs = {s.name: s for s in discover_sequences()}
    rows = []
    for run_dir in sorted((RUNS_DIR / name).iterdir()):
        if run_dir.name in seqs and (run_dir / "summary.json").exists():
            rows.extend(evaluate_run(run_dir, seqs[run_dir.name], {SELECT_VARIANT: VARIANTS[SELECT_VARIANT]}))
    df = pd.DataFrame(rows)
    dev = df[df["split"] == "dev"]
    s = summarise(dev, ["config"]).iloc[0]
    return {"config": name, "dev_wape": float(s.wape_pct), "dev_abs_err": int(s.abs_err),
            "dev_idf1": float(s.idf1), "ms_per_frame": float(s.ms_per_frame)}


def pick_best(scores: list[dict]) -> str:
    """Lowest dev counting error; among configs within TIE_ERRORS of the best, the fastest."""
    best_err = min(s["dev_abs_err"] for s in scores)
    tied = [s for s in scores if s["dev_abs_err"] <= best_err + TIE_ERRORS]
    return min(tied, key=lambda s: s["ms_per_frame"])["config"]


def run_stage(stage: str, names: list[str], log: dict, **kw) -> str:
    for n in names:
        run_config(n, **kw)
    scores = [dev_score(n) for n in names]
    best = pick_best(scores)
    log[stage] = {"scores": scores, "chosen": best}
    SELECTION_FILE.write_text(json.dumps(log, indent=2))
    print(f"\n>> {stage}: chose {best}\n{pd.DataFrame(scores).to_string(index=False)}\n", flush=True)
    return best


def plan_onefactor(**kw) -> None:
    """Stage 1 image size -> stage 2 model -> stage 3 tracker, one factor at a time."""
    log: dict = {"rule": f"min dev abs counting error ({SELECT_VARIANT}); "
                         f"within {TIE_ERRORS} error(s) of best -> fastest"}
    best = run_stage("1_imgsz", ["n640_bytetrack", "n960_bytetrack", "n1280_bytetrack"], log, **kw)
    imgsz = parse_config(best)["imgsz"]
    best = run_stage("2_model", [best, f"s{imgsz}_bytetrack", f"m{imgsz}_bytetrack"], log, **kw)
    stem = best.split("_")[0]  # e.g. "s1280"
    best = run_stage("3_tracker", [best, f"{stem}_botsort", f"{stem}_botsort_reid"], log, **kw)
    log["final"] = best
    SELECTION_FILE.write_text(json.dumps(log, indent=2))
    print(f"== FINAL CONFIG: {best}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--configs", nargs="+", default=None)
    p.add_argument("--plan", choices=["onefactor"], default=None)
    p.add_argument("--datasets", nargs="+", default=None, help="caviar mot17 mot20")
    p.add_argument("--sequences", nargs="+", default=None)
    p.add_argument("--max-frames", type=int, default=None, help="for quick tests only")
    args = p.parse_args()
    kw = dict(datasets=set(args.datasets) if args.datasets else None,
              only=set(args.sequences) if args.sequences else None, max_frames=args.max_frames)
    if args.plan == "onefactor":
        plan_onefactor(**kw)
    for name in args.configs or ([] if args.plan else ["baseline"]):
        run_config(name, **kw)


if __name__ == "__main__":
    main()
