"""Download the CAVIAR Portugal shopping-centre clips (videos + ground truth).

Fetches, for each of the 26 clips and both camera views (corridor + front):
  - <Clip><view>.mpg   : the MPEG-2 video (384x288, 25 fps)
  - <prefix>gt.xml     : hand-labelled ground truth (CVML XML)

Files land in data/caviar/<Clip><view>/. Existing files are skipped, so the
script is safe to re-run after an interrupted download.

Usage:
    python tools/download_caviar.py              # all clips, both views
    python tools/download_caviar.py --views cor  # corridor view only
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA2"

# Clip name -> short ground-truth code. GT file is "c<code>gt.xml" for the
# corridor view and "f<code>gt.xml" for the front view (taken from the
# CAVIAR index page).
CLIPS: dict[str, str] = {
    "WalkByShop1": "wbs1",
    "EnterExitCrossingPaths1": "eecp1",
    "EnterExitCrossingPaths2": "eecp2",
    "OneLeaveShop1": "ols1",
    "OneLeaveShop2": "ols2",
    "OneLeaveShopReenter1": "olsr1",
    "OneLeaveShopReenter2": "olsr2",
    "OneShopOneWait1": "osow1",
    "OneShopOneWait2": "osow2",
    "OneStopEnter1": "ose1",
    "OneStopEnter2": "ose2",
    "OneStopMoveEnter1": "osme1",
    "OneStopMoveEnter2": "osme2",
    "OneStopMoveNoEnter1": "osmne1",
    "OneStopMoveNoEnter2": "osmne2",
    "OneStopNoEnter1": "osne1",
    "OneStopNoEnter2": "osne2",
    "ShopAssistant1": "sa1",
    "ShopAssistant2": "sa2",
    "ThreePastShop1": "3ps1",
    "ThreePastShop2": "3ps2",
    "TwoEnterShop1": "2es1",
    "TwoEnterShop2": "2es2",
    "TwoEnterShop3": "2es3",
    "TwoLeaveShop1": "2ls1",
    "TwoLeaveShop2": "2ls2",
}

VIEWS = {"cor": "c", "front": "f"}  # view suffix -> GT filename prefix


def download(url: str, dest: Path) -> str:
    """Download url to dest unless it already exists. Returns a status word."""
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")  # avoid half-written files
    req = urllib.request.Request(url, headers={"User-Agent": "retailflow-downloader"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/caviar", help="output directory")
    parser.add_argument("--views", nargs="+", default=list(VIEWS), choices=list(VIEWS))
    args = parser.parse_args()

    out = Path(args.out)
    failures = 0
    for clip, code in CLIPS.items():
        for view in args.views:
            folder = f"{clip}{view}"
            files = [f"{folder}.mpg", f"{VIEWS[view]}{code}gt.xml"]
            for name in files:
                url = f"{BASE_URL}/{folder}/{name}"
                try:
                    status = download(url, out / folder / name)
                except Exception as exc:  # report and continue with the rest
                    status, failures = f"FAILED ({exc})", failures + 1
                print(f"[{status}] {folder}/{name}")
    print(f"Done. {failures} failure(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
