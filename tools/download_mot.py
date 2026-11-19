"""Download a MOTChallenge zip and extract only the sequences we need.

MOTChallenge only offers whole-benchmark zips (MOT17.zip is ~5.6 GB, MOT20.zip
~4.8 GB). This script downloads the zip once, extracts only the requested
training sequences into data/<benchmark>/train/<seq>/, and optionally deletes
the zip afterwards to save disk space.

Usage:
    python tools/download_mot.py MOT17 --seqs MOT17-02-FRCNN MOT17-04-FRCNN MOT17-09-FRCNN
    python tools/download_mot.py MOT20 --seqs MOT20-01 MOT20-02 --delete-zip
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://motchallenge.net/data/{name}.zip"


def download(url: str, dest: Path) -> None:
    """Stream url to dest with a simple progress line; skips if dest exists."""
    if dest.exists():
        print(f"[skip] {dest} already exists")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "retailflow-downloader"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(8 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"  {done / 2**20:,.0f} / {total / 2**20:,.0f} MB", flush=True)
    tmp.replace(dest)


def extract_sequences(zip_path: Path, out_root: Path, seqs: list[str]) -> None:
    """Extract only members under train/<seq>/ for each requested sequence."""
    with zipfile.ZipFile(zip_path) as zf:
        for seq in seqs:
            prefix_hits = [m for m in zf.namelist() if f"/train/{seq}/" in m or m.startswith(f"train/{seq}/")]
            if not prefix_hits:
                print(f"[warn] {seq} not found in {zip_path.name}")
                continue
            for member in prefix_hits:
                # Strip the leading "<benchmark>/" so files land in out_root/train/<seq>/...
                rel = member[member.index("train/"):]
                target = out_root / rel
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
            print(f"[ok] extracted {seq} ({len(prefix_hits)} entries)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("benchmark", choices=["MOT17", "MOT20"])
    parser.add_argument("--seqs", nargs="+", required=True, help="train sequence folder names")
    parser.add_argument("--data", default="data", help="data root")
    parser.add_argument("--delete-zip", action="store_true", help="remove zip after extraction")
    args = parser.parse_args()

    out_root = Path(args.data) / args.benchmark
    zip_path = Path(args.data) / "_downloads" / f"{args.benchmark}.zip"
    download(URL.format(name=args.benchmark), zip_path)
    extract_sequences(zip_path, out_root, args.seqs)
    if args.delete_zip:
        zip_path.unlink()
        print(f"[ok] deleted {zip_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
