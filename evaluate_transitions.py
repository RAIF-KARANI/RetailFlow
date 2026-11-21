"""Score zone-to-zone transitions (which zone people go to next) against ground truth.

Ground-truth tracks and the system's saved tracks go through the same zone rules
(2 s enter / 2 s leave), then through `retailflow.transitions`. Errors use the
WAPE idea: sum over sequences and zone pairs of |predicted - true| steps, divided
by the true steps (an extra A->B cannot cancel a missed B->A).

Examples:
  python evaluate_transitions.py                       # deployed system vs the YOLO26n baseline
  python evaluate_transitions.py --configs final final_phase3 n640_bytetrack
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluate import FINAL, RUNS_DIR, VIRTUAL, profile_run_dirs
from retailflow.analytics_eval import apply_filters, load_run
from retailflow.config import load_scene, scene_path_for_sequence
from retailflow.evaluate import split_for
from retailflow.groundtruth import discover_sequences
from retailflow.transitions import transition_counts, transition_matrix
from retailflow.zones import ZoneConfig, zone_analysis

OUT = Path("outputs")
CHOSEN = ZoneConfig()      # same debounce as the zone results (docs/05)


def run_dirs(cfg: str, seqs: dict) -> list[Path]:
    dirs = profile_run_dirs(seqs, cfg) if cfg in VIRTUAL else sorted((RUNS_DIR / cfg).iterdir())
    return [d for d in dirs if (d / "summary.json").exists() and d.name in seqs]


def rows_for(cfg: str, seqs: dict) -> tuple[list[dict], dict]:
    rows, matrices = [], {}
    for d in run_dirs(cfg, seqs):
        seq = seqs[d.name]
        scene = load_scene(scene_path_for_sequence(d.name))
        if len(scene.zones) < 2:
            continue
        gt, tracks, n = load_run(d, seq)
        # Same filters as the deployed system: mannequin filter where the scene enables it.
        tracks = apply_filters(tracks, seq, scene, staff=False, static=scene.static_filter)
        tv, _ = zone_analysis(gt, scene.zones, seq.fps, n, CHOSEN)
        pv, _ = zone_analysis(tracks, scene.zones, seq.fps, n, CHOSEN)
        tc, pc = transition_counts(tv), transition_counts(pv)
        names = [z.name for z in scene.zones]
        for a in names:
            for b in names:
                if a != b:
                    rows.append({"config": cfg, "sequence": d.name, "camera": scene.camera_id,
                                 "split": split_for(scene.camera_id), "from": a, "to": b,
                                 "true": tc.get((a, b), 0), "pred": pc.get((a, b), 0)})
        m = matrices.setdefault(scene.camera_id, [None, None, names])
        tm, pm = transition_matrix(tc, names), transition_matrix(pc, names)
        m[0] = tm if m[0] is None else m[0] + tm
        m[1] = pm if m[1] is None else m[1] + pm
    return rows, matrices


def summarise(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    d = df.assign(abs_err=(df.pred - df.true).abs())
    g = d.groupby(by).agg(sequences=("sequence", "nunique"), true_steps=("true", "sum"),
                          pred_steps=("pred", "sum"), abs_err=("abs_err", "sum")).reset_index()
    g["wape_pct"] = (100 * g.abs_err / g.true_steps.clip(lower=1)).round(1)
    return g


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--configs", nargs="+", default=[FINAL, "n640_bytetrack"])
    args = p.parse_args()
    seqs = {s.name: s for s in discover_sequences()}

    all_rows = []
    for cfg in args.configs:
        rows, matrices = rows_for(cfg, seqs)
        all_rows += rows
        print(f"\n=== {cfg}: transition matrices, all clips of a camera summed (true | predicted)")
        for cam, (tm, pm, _) in sorted(matrices.items()):
            print(f"\n{cam}  true:\n{tm.to_string()}\n{cam}  predicted:\n{pm.to_string()}")
    df = pd.DataFrame(all_rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "transitions_results.csv", index=False)
    with pd.option_context("display.width", 200):
        print("\nper camera:")
        print(summarise(df, ["config", "split", "camera"]).to_string(index=False))
        print("\nper camera and zone pair (final system):")
        f = df[df.config == args.configs[0]]
        print(summarise(f, ["camera", "from", "to"]).query("true_steps > 0 or pred_steps > 0")
              .to_string(index=False))


if __name__ == "__main__":
    main()
