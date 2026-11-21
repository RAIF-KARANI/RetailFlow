"""Track people in every camera of a multi-camera scene (one camera after another).

Runs the deployed detector + tracker (scene profile `standard`: YOLO26m, TensorRT
FP16 if built, BoT-SORT without GMC) on each camera's video and writes
<out>/<camera>/tracks.csv (frame, track_id, x1, y1, x2, y2, conf) plus a small
summary.json. Cross-camera linking happens afterwards (retailflow/mtmc.py), so a
camera only needs to be tracked once.

  python tools/track_multicam.py --scene-dir data/smartspaces/scene_071 --seconds 120
  python tools/track_multicam.py --scene-dir data/smartspaces/scene_071            # whole video

Videos under 1 MB are skipped (the dataset's corrupted camera_0649 is 0.3 MB).
A camera whose summary.json already covers the requested length is skipped too.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retailflow.detector import PersonTracker          # noqa: E402
from retailflow.profiles import get_profile           # noqa: E402
from retailflow.sources import FrameSource             # noqa: E402


def track_camera(video: Path, out_dir: Path, tracker: PersonTracker, max_frames: int | None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    src = FrameSource(str(video), max_frames=max_frames)
    t0, n, ms = time.perf_counter(), 0, 0.0
    with open(out_dir / "tracks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "conf"])
        for fr in src:
            t = time.perf_counter()
            for p in tracker.update(fr.image):
                w.writerow([fr.index, p.track_id, f"{p.x1:.1f}", f"{p.y1:.1f}", f"{p.x2:.1f}",
                            f"{p.y2:.1f}", f"{p.conf:.3f}"])
            ms += (time.perf_counter() - t) * 1000
            n += 1
    summary = {"video": str(video), "frames": n, "fps": src.fps, "avg_proc_ms": round(ms / max(n, 1), 2),
               "wall_s": round(time.perf_counter() - t0, 1)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scene-dir", required=True, help="folder with camera_*/video.mp4")
    p.add_argument("--out", default=None, help="default outputs/mtmc/<scene>/tracks")
    p.add_argument("--seconds", type=float, default=None, help="only the first N seconds of each camera")
    p.add_argument("--profile", default="standard")
    p.add_argument("--cams", nargs="*", default=None, help="only these camera folders")
    p.add_argument("--tracker", default=None, help="override the profile's tracker (name or .yaml)")
    args = p.parse_args()

    scene = Path(args.scene_dir)
    out = Path(args.out or f"outputs/mtmc/{scene.name}/tracks")
    prof = get_profile(args.profile)
    videos = sorted(scene.glob("camera_*/video.mp4"))
    videos = [v for v in videos if v.stat().st_size > 1_000_000 and (not args.cams or v.parent.name in args.cams)]
    print(f"{len(videos)} cameras -> {out}")
    for v in videos:
        cam_out = out / v.parent.name
        fps = FrameSource(str(v)).fps
        want = int(args.seconds * fps) if args.seconds else None
        done = cam_out / "summary.json"
        if done.exists():
            prev = json.loads(done.read_text())
            if want is not None and prev["frames"] >= want or want is None and prev.get("complete"):
                print(f"  {v.parent.name}: already tracked, skipped")
                continue
        # A fresh tracker per camera: track IDs are per camera, like a real deployment.
        tracker = PersonTracker(prof.weights("auto"), args.tracker or prof.tracker, prof.imgsz, prof.conf, 0)
        s = track_camera(v, cam_out, tracker, want)
        s["complete"] = want is None
        (cam_out / "summary.json").write_text(json.dumps(s, indent=2))
        print(f"  {v.parent.name}: {s['frames']} frames, {s['avg_proc_ms']} ms/frame, {s['wall_s']} s", flush=True)


if __name__ == "__main__":
    main()
