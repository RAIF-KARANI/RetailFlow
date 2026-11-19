"""Simulate live CCTV cameras: loop dataset videos into MediaMTX as RTSP streams.

    docker compose --profile streaming up -d mediamtx
    python tools/simulate_rtsp.py caviar_front=data/caviar/TwoEnterShop1front/TwoEnterShop1front.mpg ^
                                  mot17_09=data/MOT17/train/MOT17-09-FRCNN/img1@30
    # -> rtsp://127.0.0.1:8554/caviar_front, rtsp://127.0.0.1:8554/mot17_09   (Ctrl+C stops)

Each argument is name=source. An image folder needs its frame rate: folder@fps.
FFmpeg plays the source at real-time speed (-re), loops it forever, and encodes
H.264 with low-latency settings, like an IP camera (one keyframe per second, no B-frames).

Uses 127.0.0.1 rather than localhost (see TROUBLESHOOTING: IPv6 first is slow).
"""

from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

RTSP = "rtsp://127.0.0.1:8554"


def ffmpeg_cmd(name: str, source: str) -> list[str]:
    fps = None
    if "@" in source:
        source, fps = source.rsplit("@", 1)
    src = Path(source)
    if src.is_dir():
        first = sorted(src.glob("*.jpg"))[0]
        pattern = str(src / f"%0{len(first.stem)}d.jpg")
        inp = ["-framerate", fps or "30", "-i", pattern]
    else:
        inp = ["-i", str(src)]
    gop = str(round(float(fps or 25)))
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-re", "-stream_loop", "-1", *inp,
            "-an", "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-g", gop, "-bf", "0", "-pix_fmt", "yuv420p",
            "-f", "rtsp", "-rtsp_transport", "tcp", f"{RTSP}/{name}"]


def start(streams: dict[str, str]) -> list[subprocess.Popen]:
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH")
    procs = []
    for name, src in streams.items():
        procs.append(subprocess.Popen(ffmpeg_cmd(name, src)))
        print(f"[stream] {RTSP}/{name}  <-  {src}")
    return procs


def stop(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


def parse(args: list[str]) -> dict[str, str]:
    out = {}
    for a in args:
        name, _, src = a.partition("=")
        if not src:
            raise SystemExit(f"expected name=source, got {a!r}")
        out[name] = src
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("streams", nargs="+", help="name=video_or_folder[@fps]")
    args = p.parse_args()
    procs = start(parse(args.streams))
    signal.signal(signal.SIGINT, lambda *_: (stop(procs), sys.exit(0)))
    while all(p.poll() is None for p in procs):
        time.sleep(1)
    stop(procs)
    sys.exit("an ffmpeg process exited; is MediaMTX running? (docker compose --profile streaming up -d mediamtx)")


if __name__ == "__main__":
    main()
