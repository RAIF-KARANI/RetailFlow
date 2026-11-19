"""Run several simulated CCTV streams at once and check whether they keep up in real time.

    docker compose --profile streaming up -d mediamtx
    python run_multi.py --n 4 --seconds 60 --cams data/caviar/TwoEnterShop1front/TwoEnterShop1front.mpg
    python run_multi.py --n 2 --cams data/MOT17/train/MOT17-09-FRCNN/img1@30 --model models/yolo26m_960.engine

For each of the N streams this:
  1. publishes a looped copy of a dataset video to MediaMTX (tools/simulate_rtsp.py),
  2. starts one pipeline PROCESS reading rtsp://127.0.0.1:8554/<stream> (all share the GPU),
  3. after --seconds, collects each summary.json: processing ms/frame, throughput, the
     lag check (does lag grow?) and MediaMTX's "reader too slow, discarding" warnings for
     that stream. A stream keeps up only if lag does not grow AND nothing was dropped.

Streams are assigned round-robin from --cams. The scene (line, zones, profile) is
the camera's scene file, exactly as for run.py. Results: outputs/multi/<tag>/.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from retailflow.config import load_scene, scene_path_for_sequence
from tools.simulate_rtsp import RTSP, start, stop


def seq_name(source: str) -> str:
    p = Path(source.split("@")[0])
    return p.parent.name if p.is_dir() else p.stem


def worker(name: str, scene_path: str, out_dir: str, seconds: float, model: str | None,
           backend: str, api: str | None, ready: mp.Barrier) -> None:
    # Imports inside the worker: each process loads its own model and tracker.
    from retailflow.counting import CountingConfig
    from retailflow.detector import PersonTracker
    from retailflow.pipeline import Pipeline
    from retailflow.profiles import get_profile
    from retailflow.sender import ApiEventSender
    from retailflow.sources import FrameSource

    scene = load_scene(scene_path)
    prof = get_profile(scene.profile)
    weights = model or prof.weights(backend)
    tracker = PersonTracker(weights, prof.tracker, prof.imgsz, prof.conf, 0)
    run_info = {"profile": scene.profile, "model": weights, "tracker": prof.tracker,
                "imgsz": prof.imgsz, "stream": name}
    # Warm up BEFORE opening the stream: the first inference builds the TensorRT
    # context / CUDA kernels and takes seconds. If the stream were already open,
    # MediaMTX's queue for this reader would overflow and it would drop frames.
    import numpy as np
    for _ in range(3):
        tracker.model.predict(np.zeros((scene.frame_size[1], scene.frame_size[0], 3), np.uint8),
                              imgsz=prof.imgsz, device=0, verbose=False)
    ready.wait()                                  # all models loaded and warm: start reading together
    src = FrameSource(f"{RTSP}/{name}")
    src.max_frames = int(seconds * src.fps)
    sinks = [ApiEventSender(api, scene, f"{RTSP}/{name}", src.fps, run_info)] if api else []
    Pipeline(scene, tracker, CountingConfig(), out_dir, run_info=run_info, sinks=sinks).run(src)


def server_drops(since_utc: str) -> dict[str, int]:
    """MediaMTX 'reader is too slow, discarding N frames' warnings per stream since a time.

    N counts RTP packets in MediaMTX's queue, not video frames, so this shows WHETHER
    a reader fell behind (and roughly how badly), not an exact number of lost frames.
    Returns {} if the docker logs cannot be read.
    """
    try:
        log = subprocess.run(["docker", "compose", "logs", "mediamtx", "--since", since_utc, "--no-log-prefix"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return {}
    sessions, drops = {}, {}
    for line in log.splitlines():
        m = re.search(r"\[session (\w+)\] is reading from path '([\w-]+)'", line)
        if m:
            sessions[m.group(1)] = m.group(2)
        m = re.search(r"\[session (\w+)\] reader is too slow, discarding (\d+) frames", line)
        if m:
            path = sessions.get(m.group(1), m.group(1))
            drops[path] = drops.get(path, 0) + int(m.group(2))
    return drops


def gpu_sampler(stop_evt: threading.Event, samples: list) -> None:
    """Sample GPU utilisation and memory once a second with nvidia-smi."""
    while not stop_evt.is_set():
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,clocks.gr",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            u, m, c = (float(v) for v in out.stdout.strip().split(","))
            samples.append((u, m, c))
        except Exception:
            pass
        stop_evt.wait(1.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cams", nargs="+", required=True, help="video file or image_folder@fps")
    p.add_argument("--n", type=int, default=1, help="number of simultaneous streams")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--backend", choices=["auto", "pt", "trt"], default="auto",
                   help="auto = TensorRT engine if built, else PyTorch")
    p.add_argument("--model", default=None, help="override the profile's model file")
    p.add_argument("--api", default=None, help="also send events, e.g. http://127.0.0.1:8000")
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    names, scenes, sources = [], [], {}
    for i in range(args.n):
        src = args.cams[i % len(args.cams)]
        seq = seq_name(src)
        name = f"{seq.lower().replace('-', '_')}_{i}"
        names.append(name)
        scenes.append(str(scene_path_for_sequence(seq)))
        sources[name] = src
    model_tag = Path(args.model).stem if args.model else args.backend
    tag = args.tag or f"{seq_name(args.cams[0])}_n{args.n}_{model_tag}"
    out = Path("outputs/multi") / tag

    since = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    procs = start(sources)
    time.sleep(3)                                 # let MediaMTX receive the first keyframes
    ctx = mp.get_context("spawn")
    ready = ctx.Barrier(args.n)
    workers = [ctx.Process(target=worker, args=(n, s, str(out / n), args.seconds, args.model, args.backend,
                                                args.api, ready))
               for n, s in zip(names, scenes)]
    samples: list = []
    stop_evt = threading.Event()
    try:
        for w in workers:
            w.start()
        threading.Thread(target=gpu_sampler, args=(stop_evt, samples), daemon=True).start()
        for w in workers:
            w.join()
    finally:
        stop_evt.set()
        stop(procs)

    drops = server_drops(since)
    rows = []
    for n in names:
        f = out / n / "summary.json"
        if not f.exists():
            rows.append({"stream": n, "error": "no summary (worker failed)"})
            continue
        s = json.loads(f.read_text())
        live = s.get("live", {})
        rows.append({"stream": n, "model": Path(s["model"]).name, "imgsz": s["imgsz"], "frames": s["frames"],
                     "video_fps": s["video_fps"], "proc_ms": s["avg_proc_ms"], "throughput_fps": s["throughput_fps"],
                     "lag_max_s": live.get("lag_max_s"), "lag_slope": live.get("lag_slope"),
                     "realtime_factor": live.get("realtime_factor"), "ended_early": live.get("ended_early"),
                     "server_dropped": drops.get(n, 0),
                     # Real time = lag not growing AND the server never had to drop data for us
                     # (dropped frames shorten the stream and would hide a growing lag).
                     "keeps_up": bool(live.get("keeps_up")) and drops.get(n, 0) == 0})
    gpu = {}
    if samples:
        busy = samples[len(samples) // 4:]       # skip model loading
        gpu = {"gpu_util_mean": round(sum(u for u, _, _ in busy) / len(busy), 1),
               "gpu_mem_max_mb": max(m for _, m, _ in samples),
               "gpu_clock_mean_mhz": round(sum(c for _, _, c in busy) / len(busy))}
    result = {"n": args.n, "seconds": args.seconds, "cams": args.cams, "model": args.model or args.backend,
              "streams": rows, **gpu}
    out.mkdir(parents=True, exist_ok=True)
    (out / "multi_summary.json").write_text(json.dumps(result, indent=2))
    for r in rows:
        print(r)
    print(gpu, "->", out / "multi_summary.json")


if __name__ == "__main__":
    main()
