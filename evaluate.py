"""Evaluate benchmark runs against ground truth and write outputs/results.csv.

Examples:
  python evaluate.py                                  # all configs found in outputs/runs
  python evaluate.py --configs baseline --variants centre_simple feet_simple
  python evaluate.py --group dataset density          # choose summary grouping
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from retailflow.evaluate import VARIANTS, evaluate_run, summarise
from retailflow.groundtruth import discover_sequences

RUNS_DIR = Path("outputs/runs")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--configs", nargs="+", default=None)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--group", nargs="+", default=["camera"], help="summary grouping columns")
    p.add_argument("--out", default="outputs/results.csv")
    args = p.parse_args()

    seqs = {s.name: s for s in discover_sequences()}
    configs = args.configs or sorted(d.name for d in RUNS_DIR.iterdir() if d.is_dir())
    variants = {k: VARIANTS[k] for k in args.variants}
    rows = []
    for cfg in configs:
        for run_dir in sorted((RUNS_DIR / cfg).iterdir()):
            if (run_dir / "summary.json").exists() and run_dir.name in seqs:
                rows.extend(evaluate_run(run_dir, seqs[run_dir.name], variants))
    if not rows:
        raise SystemExit("no finished runs found")
    results = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.max_rows", 500):
        print(summarise(results, ["config", "variant", *args.group]).to_string(index=False))
    print(f"\n{len(results)} rows -> {args.out}")


if __name__ == "__main__":
    main()
