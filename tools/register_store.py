"""Register a store (cameras, floor bounds, zones) and its floor images with the API, without a run.

    python tools/register_store.py --store scenes/stores/smartspaces_071.json --api http://127.0.0.1:8000

run_store.py does the same at the start of every run; this is for updating zones or the
floor plan (tools/build_floor_plan.py) so the dashboard shows them right away.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retailflow.sender import StoreApiClient, _request   # noqa: E402
from retailflow.store_live import load_store              # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--store", default="scenes/stores/smartspaces_071.json")
    p.add_argument("--api", default="http://127.0.0.1:8000")
    args = p.parse_args()
    store = load_store(args.store)
    base = args.api.rstrip("/").replace("//localhost:", "//127.0.0.1:")
    _request("PUT", f"{base}/api/stores/{store.store_id}", store.to_api())
    sent = StoreApiClient.upload_floor(base, store.store_id, store.scene_dir)
    print(f"{store.store_id}: {len(store.cameras)} cameras, {len(store.zones)} zones, floor images: {sent or 'none'}")


if __name__ == "__main__":
    main()
