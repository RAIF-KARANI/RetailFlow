"""Speed of each inference backend: model only, and the full pipeline (docs/08).

    python tools/benchmark_speed.py                         # writes outputs/speed.csv
    python tools/benchmark_speed.py --backends onnxruntime_fp32   # re-measure one backend

ONNX Runtime's GPU provider needs CUDA 13 + cuDNN 9 DLLs. They come with PyTorch,
which this script (via Ultralytics) imports first; otherwise ORT silently runs on CPU.

Model-only: 200 frames held in memory (no decoding), after 20 warm-up frames.
Ultralytics reports preprocess / inference / postprocess ms per frame.
Full pipeline: Pipeline.run on the first 300 frames of the clip from disk
(detect + track + count + zones + heatmap), reported as avg_proc_ms like every
other run in this project.

Clips: one per camera type, at the profile the scene file would use:
  caviar_front 384x288  -> 960 profile   (small retail camera)
  MOT17-09     1920x1080 -> 960 profile  (1080p shop front)
  MOT20-02     1920x1080 -> 1920 profile (crowd)
"""

from __future__ import annotations

import csv
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow "import retailflow"

import torch
from ultralytics import YOLO

from retailflow.config import load_scene
from retailflow.counting import CountingConfig
from retailflow.detector import PersonTracker
from retailflow.pipeline import Pipeline
from retailflow.sources import FrameSource

CLIPS = [
    ("caviar_front", "data/caviar/TwoEnterShop1front/TwoEnterShop1front.mpg", None, 960),
    ("MOT17-09", "data/MOT17/train/MOT17-09-FRCNN/img1", 30.0, 960),
    ("MOT20-02", "data/MOT20/train/MOT20-02/img1", 25.0, 1920),
]
BACKENDS = [  # (label, model file template, half)
    ("pytorch_fp32", "yolo26m.pt", False),
    ("pytorch_fp16", "yolo26m.pt", True),
    ("onnxruntime_fp32", "models/yolo26m_{imgsz}.onnx", False),
    ("tensorrt_fp16", "models/yolo26m_{imgsz}.engine", True),
]


def frames(src, fps, n):
    out = []
    for f in FrameSource(src, fps=fps, max_frames=n):
        out.append(f.image)
    return out


def model_only(weights: str, imgsz: int, half: bool, imgs) -> dict:
    m = YOLO(weights, task="detect")
    kw = dict(imgsz=imgsz, conf=0.1, classes=[0], device=0, verbose=False, half=half)
    for im in imgs[:20]:
        m.predict(im, **kw)
    sp = {"preprocess": [], "inference": [], "postprocess": []}
    wall = []
    for im in imgs[20:]:
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = m.predict(im, **kw)[0]
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - t) * 1000)
        for k in sp:
            sp[k].append(r.speed[k])
    return {f"{k}_ms": round(statistics.mean(v), 2) for k, v in sp.items()} | {
        "model_total_ms": round(statistics.mean(wall), 2), "input_shape": "x".join(map(str, r.orig_shape))}


def pipeline_ms(weights: str, imgsz: int, cam: str, src: str, fps) -> float:
    scene = load_scene(f"scenes/{cam}.json")
    tracker = PersonTracker(weights, "botsort", imgsz, 0.1, 0)
    with tempfile.TemporaryDirectory() as d:
        s = Pipeline(scene, tracker, CountingConfig(), d).run(FrameSource(src, fps=fps, max_frames=300))
    return s["avg_proc_ms"]


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backends", nargs="+", default=[b[0] for b in BACKENDS],
                   help="re-measure only these; other rows of outputs/speed.csv are kept")
    args = p.parse_args()
    out = Path("outputs/speed.csv")
    keep = []
    if out.exists():
        with open(out) as f:
            keep = [r for r in csv.DictReader(f) if r["backend"] not in args.backends]
    rows = []
    for cam, src, fps, imgsz in CLIPS:
        imgs = frames(src, fps, 220)
        for label, tmpl, half in BACKENDS:
            if label not in args.backends:
                continue
            w = tmpl.format(imgsz=imgsz)
            if not Path(w).exists():
                print(f"[skip] {w} missing")
                continue
            r = {"camera": cam, "imgsz": imgsz, "backend": label, **model_only(w, imgsz, half, imgs)}
            # The tracker pipeline has no half switch for .pt, so pipeline time is
            # measured for FP32 PyTorch, ONNX and TensorRT (FP16 is built into the engine).
            r["pipeline_ms"] = pipeline_ms(w, imgsz, cam, src, fps) if label != "pytorch_fp16" else None
            print(r)
            rows.append(r)
    out.parent.mkdir(exist_ok=True)
    order = {b[0]: i for i, b in enumerate(BACKENDS)}
    cams = {c[0]: i for i, c in enumerate(CLIPS)}
    allrows = sorted(keep + rows, key=lambda r: (cams[r["camera"]], order[r["backend"]]))
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(allrows)
    print("-> outputs/speed.csv")


if __name__ == "__main__":
    main()
