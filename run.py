"""Process one video file, image folder or RTSP stream.

Examples:
  python run.py --source data/caviar/OneLeaveShop1cor/OneLeaveShop1cor.mpg --scene scenes/caviar_cor.json --save-video
  python run.py --source data/MOT17/train/MOT17-09-FRCNN/img1 --scene scenes/MOT17-09.json --fps 30
  python run.py ... --profile crowd            # force the crowd profile
  python run.py ... --model yolo26n.pt --imgsz 640 --tracker bytetrack   # Phase 2 baseline

Detector settings come from the scene's profile (retailflow/profiles.py)
unless overridden with --model / --imgsz / --tracker / --conf.
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
from retailflow.profiles import PROFILES, get_profile
from retailflow.sender import ApiEventSender
from retailflow.sources import FrameSource


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RetailFlow: count people on one camera stream")
    p.add_argument("--source", required=True, help="video file, image folder or rtsp:// URL")
    p.add_argument("--scene", required=True, help="scenes/<camera>.json")
    p.add_argument("--out", default=None, help="output folder (default outputs/run_<camera>)")
    p.add_argument("--profile", choices=sorted(PROFILES), default=None,
                   help="detector profile (default: the scene file's 'profile')")
    p.add_argument("--backend", choices=["auto", "pt", "trt"], default="auto",
                   help="auto = TensorRT engine if built (tools/export_models.py), else PyTorch")
    p.add_argument("--model", default=None, help="override the profile: yolo26n/s/m .pt, .onnx or .engine")
    p.add_argument("--tracker", default=None, help="override: bytetrack | botsort | botsort_reid | path.yaml")
    p.add_argument("--imgsz", type=int, default=None, help="override the profile's input size")
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--point", choices=["feet", "centre"], default="feet")
    p.add_argument("--no-band", action="store_true", help="simple line (band 0)")
    p.add_argument("--min-age", type=int, default=0, help="min track age in frames before counting")
    p.add_argument("--once-per-direction", action="store_true")
    p.add_argument("--fps", type=float, default=None, help="override source fps (image folders)")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--show", action="store_true", help="live preview window (press q to stop)")
    p.add_argument("--api", default=None, metavar="URL",
                   help="also send events to the RetailFlow API, e.g. http://localhost:8000")
    p.add_argument("--blur-heads", action="store_true",
                   help="pixelate heads in the saved video / preview (detection still uses the original)")
    p.add_argument("--device", default="0")
    return p


def main() -> None:
    args = build_parser().parse_args()
    scene = load_scene(args.scene)
    counting = CountingConfig(point=args.point, use_band=not args.no_band,
                              min_track_age=args.min_age, once_per_direction=args.once_per_direction)
    device = int(args.device) if args.device.isdigit() else args.device
    profile_name = args.profile or scene.profile
    prof = get_profile(profile_name)
    model = args.model or prof.weights(args.backend)
    tracker_name = args.tracker or prof.tracker
    imgsz = args.imgsz or prof.imgsz
    conf = prof.conf if args.conf is None else args.conf
    tracker = PersonTracker(model, tracker_name, imgsz, conf, device)
    out = Path(args.out or f"outputs/run_{scene.camera_id}")
    run_info = {"profile": profile_name, "model": model, "tracker": tracker_name,
                "imgsz": imgsz, "conf": conf}
    source = FrameSource(args.source, fps=args.fps, max_frames=args.max_frames)
    sinks = []
    if args.api:
        sender = ApiEventSender(args.api, scene, args.source, source.fps, run_info)
        print(f"API session {sender.session_id}")
        sinks.append(sender)
    summary = Pipeline(scene, tracker, counting, out, args.save_video, run_info,
                       blur=args.blur_heads, sinks=sinks).run(source, show=args.show)
    print(json.dumps(summary, indent=2))
    print(f"Outputs in {out}")


if __name__ == "__main__":
    main()
