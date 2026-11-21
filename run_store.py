"""Run a whole store live: every camera at once, one store-wide ID per person.

    python run_store.py --store scenes/stores/smartspaces_071.json --seconds 300 --api http://127.0.0.1:8000
    python run_store.py --store ... --cams camera_0639 camera_0643 camera_0644 camera_0645   # fewer cameras

What it does:
  1. every camera's video is replayed at real-time speed in its own thread, and the
     pipeline always takes the NEWEST frame (as from a live camera; frames it has no
     time for are skipped, and counted);
  2. each camera is analysed up to `--analysis-fps` times per second: one shared
     detector, then each camera's own tracker (retailflow/detector.py MultiCameraTracker);
  3. every `--link-every` seconds the cameras' tracks are linked into store-wide
     people (floor position + clothing colours, colours in memory only), with stable
     person numbers, and the store state goes to the API / dashboard.

No RTSP here: re-encoding 15 full-HD streams would need the CPU the pipeline needs
(single cameras over RTSP: run.py / demo.ps1). Outputs in outputs/store/<store>/<tag>/:
per-camera tracks.csv (video frame numbers, so they can be scored), links.csv
(camera, track, person) and summary.json. No colour descriptors are written.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from retailflow.detector import MultiCameraTracker
from retailflow.profiles import get_profile
from retailflow.sender import StoreApiClient
from retailflow.sources import LatestFrameReader
from retailflow.store_live import StoreLinker, load_store


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--store", required=True, help="scenes/stores/<store>.json")
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--analysis-fps", type=float, default=5.0, help="analyses per camera per second (target)")
    p.add_argument("--link-every", type=float, default=3.0, help="seconds between store-wide re-links")
    p.add_argument("--cams", nargs="*", default=None, help="only these cameras")
    p.add_argument("--no-colour", action="store_true", help="link on floor position only")
    p.add_argument("--cpu-decode", action="store_true", help="decode videos on the CPU (default: GPU if available)")
    p.add_argument("--api", default=None, metavar="URL")
    p.add_argument("--profile", default="standard")
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    store = load_store(args.store)
    cams = [c for c in store.cameras if not args.cams or c in args.cams]
    store.cameras = cams
    tag = args.tag or f"{len(cams)}cams_{int(args.seconds)}s_{args.analysis_fps:g}fps"
    out = Path("outputs/store") / store.store_id / tag
    prof = get_profile(args.profile)
    weights = prof.weights("auto")
    mct = MultiCameraTracker(weights, prof.tracker, prof.imgsz, prof.conf, 0, cams)
    linker = StoreLinker(store, appearance=not args.no_colour)

    # Warm up with a full batch before the cameras start (TensorRT context, CUDA kernels).
    blank = np.zeros((store.frame_h, 1920, 3), np.uint8)
    for _ in range(3):
        mct.model.predict(blank, imgsz=prof.imgsz, device=0, verbose=False)

    run_info = {"cameras": len(cams), "decode": "cpu" if args.cpu_decode else "hardware if available", "analysis_fps_target": args.analysis_fps, "model": weights,
                "tracker": prof.tracker, "imgsz": prof.imgsz, "colour": not args.no_colour,
                "source": "video files replayed at real-time speed"}
    api = StoreApiClient(args.api, store.to_api(), run_info) if args.api else None
    if api:
        print(f"API store run {api.run_id}", flush=True)

    max_frames = int(args.seconds * store.fps)
    readers = {}
    for c in cams:
        # Hardware video decoding (GPU) when available: 15 full-HD software decoders would take
        # the CPU the detector needs (measured: 25 ms instead of 10 ms per analysed frame).
        hw = [] if args.cpu_decode else [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY]
        cap = cv2.VideoCapture(str(store.scene_dir / c / "video.mp4"), cv2.CAP_FFMPEG, hw)
        # Only every n-th frame becomes an image: at most ~2x the analysis rate is ever needed.
        keep = max(1, int(store.fps / (2 * args.analysis_fps)))
        readers[c] = LatestFrameReader(cap, max_frames, pace_fps=store.fps, keep_every=keep)

    files = {}
    for c in cams:
        (out / c).mkdir(parents=True, exist_ok=True)
        files[c] = open(out / c / "tracks.csv", "w", newline="")
    writers = {c: csv.writer(f) for c, f in files.items()}
    for w in writers.values():
        w.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "conf"])

    interval = 1.0 / args.analysis_fps
    due = {c: 0.0 for c in cams}
    analysed = {c: 0 for c in cams}
    analysed_frames = {c: [] for c in cams}
    batch_ms, link_ms, batches = [], [], 0
    t0 = time.perf_counter()
    last_link = t0
    links = None
    try:
        while not all(r.finished for r in readers.values()):
            now = time.perf_counter()
            frames, index = {}, {}
            for c in cams:
                if now >= due[c]:
                    got = readers[c].newest()
                    if got is not None:
                        index[c], frames[c] = got
                        due[c] = max(due[c] + interval, now)     # keep the rate, never bunch up
            if not frames:
                time.sleep(0.003)
                continue
            tb = time.perf_counter()
            results = mct.update(frames)
            batch_ms.append((time.perf_counter() - tb) * 1000)
            batches += 1
            for c, people in results.items():
                analysed[c] += 1
                analysed_frames[c].append(index[c])
                for q in people:
                    writers[c].writerow([index[c], q.track_id, f"{q.x1:.1f}", f"{q.y1:.1f}", f"{q.x2:.1f}",
                                         f"{q.y2:.1f}", f"{q.conf:.3f}"])
                linker.add(c, index[c], people, frames[c])
            if time.perf_counter() - last_link >= args.link_every:
                tl = time.perf_counter()
                links = linker.relink()
                state = linker.state(links)
                link_ms.append((time.perf_counter() - tl) * 1000)
                last_link = time.perf_counter()
                if api:
                    api.state(state)
                print(f"t={state['t']:6.1f}s  people now {len(state['people_now']):3d}  "
                      f"visitors {state['unique_people']:4d}  link {link_ms[-1]:.0f} ms", flush=True)
    finally:
        for r in readers.values():
            r.stop()
        for f in files.values():
            f.close()

    wall = time.perf_counter() - t0
    links = linker.relink()
    state = linker.state(links)
    links[["camera", "track_id", "person", "start", "end", "n"]].to_csv(out / "links.csv", index=False)
    per_cam = {c: {"analysed": analysed[c], "analysed_fps": round(analysed[c] / wall, 2),
                   "frames_skipped_by_reader": readers[c].skipped} for c in cams}
    summary = {**run_info, "seconds": args.seconds, "wall_s": round(wall, 1),
               "analysed_fps_mean": round(float(np.mean([v["analysed_fps"] for v in per_cam.values()])), 2),
               "batch_ms_mean": round(float(np.mean(batch_ms)), 1) if batch_ms else None,
               "link_ms_mean": round(float(np.mean(link_ms)), 1) if link_ms else None,
               "link_ms_max": round(float(np.max(link_ms)), 1) if link_ms else None,
               "people_now_end": len(state["people_now"]), "unique_people": state["unique_people"],
               "per_camera": per_cam}
    for c in cams:
        (out / c / "summary.json").write_text(json.dumps({"frames": max_frames, "fps": store.fps,
                                                          "analysed": analysed[c], "live": True}))
        (out / c / "analysed_frames.txt").write_text(" ".join(map(str, analysed_frames[c])))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if api:
        api.finish(state, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_camera"}, indent=2))
    print(f"Outputs in {out}")


if __name__ == "__main__":
    main()
