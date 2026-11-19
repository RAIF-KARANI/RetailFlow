"""Evaluate benchmark runs against ground truth and write outputs/results.csv.

Examples:
  python evaluate.py                                  # all configs found in outputs/runs
  python evaluate.py --configs baseline --variants centre_simple feet_simple
  python evaluate.py --group dataset density          # choose summary grouping
  python evaluate.py --configs n640_bytetrack final   # baseline vs final (per-scene profile)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from retailflow.evaluate import VARIANTS, evaluate_run, summarise, sweep_variants
from retailflow.config import load_scene, scene_path_for_sequence
from retailflow.groundtruth import discover_sequences
from retailflow.profiles import get_profile

RUNS_DIR = Path("outputs/runs")
FINAL = "final"  # virtual config: each camera uses the run of its scene's profile


def profile_run_dirs(seqs: dict) -> list[Path]:
    """For every sequence, the run folder of the profile its scene file selects."""
    return [RUNS_DIR / get_profile(load_scene(scene_path_for_sequence(name)).profile).run_name / name
            for name in sorted(seqs)]


def sweep_table(results: pd.DataFrame) -> pd.DataFrame:
    """One row per counting variant: dev and test WAPE side by side, sorted by dev."""
    s = summarise(results, ["variant", "split"])
    wide = s.pivot(index="variant", columns="split", values=["wape_pct", "abs_err", "exact_seqs"])
    wide.columns = [f"{split}_{metric}" for metric, split in wide.columns]
    return wide.sort_values(["dev_wape_pct", "dev_abs_err"]).reset_index()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--configs", nargs="+", default=None)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--group", nargs="+", default=["camera"], help="summary grouping columns")
    p.add_argument("--out", default="outputs/results.csv")
    p.add_argument("--sweep", action="store_true",
                   help="tune band scale / min age / once-per-direction (feet point); "
                        "writes outputs/counting_sweep_<config>.csv")
    args = p.parse_args()

    seqs = {s.name: s for s in discover_sequences()}
    configs = args.configs or sorted(d.name for d in RUNS_DIR.iterdir() if d.is_dir())
    variants = sweep_variants() if args.sweep else {k: VARIANTS[k] for k in args.variants}
    if args.sweep:
        args.out = f"outputs/counting_sweep_{'_'.join(configs)}.csv"
    rows = []
    for cfg in configs:
        run_dirs = profile_run_dirs(seqs) if cfg == FINAL else sorted((RUNS_DIR / cfg).iterdir())
        for run_dir in run_dirs:
            if (run_dir / "summary.json").exists() and run_dir.name in seqs:
                new = evaluate_run(run_dir, seqs[run_dir.name], variants)
                if cfg == FINAL:
                    for r in new:
                        r["config"] = FINAL
                rows.extend(new)
    if not rows:
        raise SystemExit("no finished runs found")
    results = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.max_rows", 500):
        if args.sweep:
            print(sweep_table(results).to_string(index=False))
        else:
            print(summarise(results, ["config", "variant", *args.group]).to_string(index=False))
    print(f"\n{len(results)} rows -> {args.out}")


if __name__ == "__main__":
    main()
