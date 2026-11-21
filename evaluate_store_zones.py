"""Score the store zones against ground truth (SmartSpaces floor positions).

Ground truth gives every person's floor position in every camera frame. The true
store-wide people (one ID across cameras), once per second at their median position,
go through exactly the same zone code as the system (store_live.store_zone_analysis:
2 s enter / 2 s leave). The system side is its own store-wide people, either from a
live run (run_store.py output) or from offline linking of saved tracks.

Measures per zone, over the scoring window:
- occupancy MAE: |predicted - true people in the zone|, averaged over seconds;
- time in zone (person-seconds): WAPE;                     } not hurt by one person
                                                            } getting several IDs
- visits and mean stay: hurt by ID splits (one visit becomes several shorter ones);
- zone-to-zone moves: WAPE over the move matrix (also hurt by ID splits).

  python evaluate_store_zones.py --live-run outputs/store/smartspaces_071/live_420s_batch --from-s 120
  python evaluate_store_zones.py --offline outputs/mtmc/scene_071/tracks_7min --from-s 120
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_mtmc import appearance_for, load_gt, load_tracks
from retailflow.mtmc import MTMCConfig, link
from retailflow.store_live import load_store, store_zone_analysis


def true_positions(scene: Path, fps: float) -> pd.DataFrame:
    g = load_gt(scene)
    g["sec"] = (g.frame // fps).astype(int)
    return g.groupby(["sec", "obj"])[["xw", "yw"]].median().reset_index().rename(columns={"obj": "person"})


def project(t: pd.DataFrame, H: np.ndarray, frame_h: int = 1080, edge_px: float = 4.0) -> pd.DataFrame:
    q = np.c_[(t.x1 + t.x2) / 2, t.y2, np.ones(len(t))] @ np.linalg.inv(H).T
    return t.assign(xw=q[:, 0] / q[:, 2], yw=q[:, 1] / q[:, 2], usable=t.y2 < frame_h - edge_px)


def live_positions(run: Path, store) -> pd.DataFrame:
    links = pd.read_csv(run / "links.csv")
    parts = []
    for d in sorted(run.glob("camera_*")):
        t = project(pd.read_csv(d / "tracks.csv"), store.homographies[d.name]).assign(camera=d.name)
        parts.append(t)
    allr = pd.concat(parts, ignore_index=True).merge(links[["camera", "track_id", "person"]],
                                                     on=["camera", "track_id"])
    allr = allr[allr.usable]
    allr["sec"] = (allr.frame // store.fps).astype(int)
    return allr.groupby(["sec", "person"])[["xw", "yw"]].median().reset_index()


def offline_positions(tracks: Path, store, appearance: bool) -> pd.DataFrame:
    cams, _, fps = load_tracks(store.scene_dir, tracks, MTMCConfig())
    app = appearance_for(store.scene_dir, cams) if appearance else None     # in memory only
    links = link(cams, fps, MTMCConfig(), appearance=app)
    allr = pd.concat([t.assign(camera=c) for c, t in cams.items()], ignore_index=True)
    allr = allr.merge(links[["camera", "track_id", "global_id"]], on=["camera", "track_id"]).rename(
        columns={"global_id": "person"})
    allr = allr[allr.usable]
    allr["sec"] = (allr.frame // fps).astype(int)
    return allr.groupby(["sec", "person"])[["xw", "yw"]].median().reset_index()


def score(true_pos: pd.DataFrame, pred_pos: pd.DataFrame, zones: list, lo: int, hi: int) -> dict:
    t = store_zone_analysis(true_pos[(true_pos.sec >= lo) & (true_pos.sec < hi)], zones)
    p = store_zone_analysis(pred_pos[(pred_pos.sec >= lo) & (pred_pos.sec < hi)], zones)
    secs = range(lo, hi)
    to = t["occupancy"].reindex(secs, fill_value=0)
    po = p["occupancy"].reindex(secs, fill_value=0)
    rows = []
    for n in t["names"]:
        tv, pv = t["visits"][t["visits"].zone == n], p["visits"][p["visits"].zone == n]
        rows.append({"zone": n,
                     "true_people_mean": round(float(to[n].mean()), 2),
                     "occupancy_mae": round(float((po[n] - to[n]).abs().mean()), 2),
                     "true_person_s": int(to[n].sum()), "pred_person_s": int(po[n].sum()),
                     "true_visits": len(tv), "pred_visits": len(pv),
                     "true_mean_stay_s": round(float(tv.dwell_s.mean()), 1) if len(tv) else None,
                     "pred_mean_stay_s": round(float(pv.dwell_s.mean()), 1) if len(pv) else None})
    df = pd.DataFrame(rows)
    tm, pm = np.array(t["moves"]), np.array(p["moves"])
    total = {
        "occupancy_mae_mean": round(float(df.occupancy_mae.mean()), 2),
        "true_people_per_zone_mean": round(float(df.true_people_mean.mean()), 2),
        "person_s_wape_pct": round(100 * float((df.pred_person_s - df.true_person_s).abs().sum())
                                   / max(1, df.true_person_s.sum()), 1),
        "visits_true": int(df.true_visits.sum()), "visits_pred": int(df.pred_visits.sum()),
        "visits_wape_pct": round(100 * float((df.pred_visits - df.true_visits).abs().sum())
                                 / max(1, df.true_visits.sum()), 1),
        "moves_true": int(tm.sum()), "moves_pred": int(pm.sum()),
        "moves_wape_pct": round(100 * float(np.abs(pm - tm).sum()) / max(1, tm.sum()), 1),
    }
    return {"window_s": [lo, hi], "total": total, "zones": rows,
            "moves": {"zones": t["names"], "true": tm.tolist(), "pred": pm.tolist()}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--store", default="scenes/stores/smartspaces_071.json")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--live-run", help="run_store.py output folder")
    src.add_argument("--offline", help="tracks folder (tools/track_multicam.py); linked here")
    p.add_argument("--no-colour", action="store_true", help="offline: link on floor position only")
    p.add_argument("--from-s", type=float, default=120.0)
    p.add_argument("--to-s", type=float, default=None)
    args = p.parse_args()

    store = load_store(args.store)
    zones = list(store.zones)
    true_pos = true_positions(store.scene_dir, store.fps)
    if args.live_run:
        pred_pos, out = live_positions(Path(args.live_run), store), Path(args.live_run)
        name = "store_zone_score.json"
    else:
        pred_pos, out = offline_positions(Path(args.offline), store, not args.no_colour), Path(args.offline).parent
        name = f"store_zone_score_offline{'_nocolour' if args.no_colour else ''}.json"
    hi = int(args.to_s) if args.to_s else int(pred_pos.sec.max()) + 1
    r = score(true_pos, pred_pos, zones, int(args.from_s), hi)
    (out / name).write_text(json.dumps(r, indent=2))
    with pd.option_context("display.width", 200):
        print(pd.DataFrame(r["zones"]).to_string(index=False))
    print(json.dumps(r["total"], indent=2))


if __name__ == "__main__":
    main()
