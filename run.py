"""Process one video file, image folder or RTSP stream.

Examples:
  python run.py --source data/caviar/OneLeaveShop1cor/OneLeaveShop1cor.mpg --scene scenes/caviar_cor.json --save-video
  python run.py --source data/MOT17/train/MOT17-09-FRCNN/img1 --scene scenes/MOT17-09.json --fps 30 --imgsz 960
  python run.py --source rtsp://localhost:8554/cam1 --scene scenes/caviar_cor.json --show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from retailflow.config import load_scene
from retailflow.counting import CountingConfig
from retailflow.detector import PersonTracker
from retailflow.pipeline import Pipeline
from retailflow.sources import FrameSource


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RetailFlow: count people on one camera stream")
    p.add_argument("--source", required=True, help="video file, image folder or rtsp:// URL")
    p.add_argument("--scene", required=True, help="scenes/<camera>.json")
    p.add_argument("--out", default=None, help="output folder (default outputs/run_<camera>)")
    p.add_argument("--model", default="yolo26n.pt", help="yolo26n/s/m .pt, .onnx or .engine")
    p.add_argument("--tracker", default="bytetrack", help="bytetrack | botsort | botsort_reid | path.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--point", choices=["feet", "centre"], default="feet")
    p.add_argument("--no-band", action="store_true", help="simple line (band 0)")
    p.add_argument("--min-age", type=int, default=0, help="min track age in frames before counting")
    p.add_argument("--once-per-direction", action="store_true")
    p.add_argument("--fps", type=float, default=None, help="override source fps (image folders)")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--show", action="store_true", help="live preview window (press q to stop)")
    p.add_argument("--device", default="0")
    return p


def main() -> None:
    args = build_parser().parse_args()
    scene = load_scene(args.scene)
    counting = CountingConfig(point=args.point, use_band=not args.no_band,
                              min_track_age=args.min_age, once_per_direction=args.once_per_direction)
    device = int(args.device) if args.device.isdigit() else args.device
    tracker = PersonTracker(args.model, args.tracker, args.imgsz, args.conf, device)
    out = Path(args.out or f"outputs/run_{scene.camera_id}")
    run_info = {"model": args.model, "tracker": args.tracker, "imgsz": args.imgsz, "conf": args.conf}
    summary = Pipeline(scene, tracker, counting, out, args.save_video, run_info).run(
        FrameSource(args.source, fps=args.fps, max_frames=args.max_frames), show=args.show)
    print(json.dumps(summary, indent=2))
    print(f"Outputs in {out}")


if __name__ == "__main__":
    main()
