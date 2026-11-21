"""Download one retail scene of NVIDIA PhysicalAI-SmartSpaces (MTMC_Tracking_2024).

Synthetic indoor store (NVIDIA Omniverse), 15-16 synchronised angled cameras,
1920x1080 at 30 fps, about 13 minutes, with multi-camera ground truth (the same
person ID in every camera) and calibration. Licence: CC BY 4.0
(https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces).

Scenes 071-080 of the 2024 test split are the retail store (the card says so);
each is 3.1-4.2 GB. By default only scene_071 is fetched, without the 2025-format
ground truth JSON (the 2024 text file has the same boxes and IDs) and without the
camera the card marks as corrupted (scene_071/camera_0649).

  python tools/download_smartspaces.py                    # scene_071, about 3.0 GB
  python tools/download_smartspaces.py --scene scene_072 --list   # sizes only

Files go to data/smartspaces/<scene>/ (git-ignored). Downloads resume: a file
that already has the right size is skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO = "nvidia/PhysicalAI-SmartSpaces"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
FILES = f"https://huggingface.co/datasets/{REPO}/resolve/main"
CORRUPT = {"MTMC_Tracking_2024/test/scene_071/camera_0649/video.mp4"}   # per the dataset card
SKIP_SUFFIX = ("ground_truth_2025_format.json",)


def listing(path: str) -> list[dict]:
    with urllib.request.urlopen(f"{API}/{path}?recursive=true", timeout=60) as r:
        return [x for x in json.load(r) if x["type"] == "file"]


def fetch(url: str, dest: Path, size: int) -> None:
    if dest.exists() and dest.stat().st_size == size:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    done = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "ab" if done else "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    if tmp.stat().st_size != size:
        raise RuntimeError(f"{dest}: got {tmp.stat().st_size} bytes, expected {size}")
    tmp.replace(dest)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scene", default="scene_071")
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="data/smartspaces")
    p.add_argument("--list", action="store_true", help="print files and sizes, download nothing")
    args = p.parse_args()

    base = f"MTMC_Tracking_2024/{args.split}/{args.scene}"
    files = [f for f in listing(base) if f["path"] not in CORRUPT and not f["path"].endswith(SKIP_SUFFIX)]
    total = sum(f["size"] for f in files)
    print(f"{args.scene}: {len(files)} files, {total / 1e9:.2f} GB")
    if args.list:
        for f in files:
            print(f"  {f['size'] / 1e6:8.1f} MB  {f['path'][len(base) + 1:]}")
        return
    got = 0
    for f in files:
        rel = f["path"][len(base) + 1:]
        fetch(f"{FILES}/{f['path']}", Path(args.out) / args.scene / rel, f["size"])
        got += f["size"]
        print(f"  {got / 1e9:5.2f} / {total / 1e9:.2f} GB  {rel}", flush=True)
    print("done")


if __name__ == "__main__":
    sys.exit(main())
