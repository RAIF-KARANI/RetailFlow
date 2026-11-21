"""Draw a store's zones on its top-down floor plan (made by tools/build_floor_plan.py).

    python tools/setup_store_zones.py --store scenes/stores/smartspaces_071.json

A window shows the floor photo with the current zones.
  left click   add a corner to the zone being drawn
  Enter        finish the zone (you are asked for its name in the terminal)
  u            undo the last corner          d   delete the last zone
  s            save to the store file         q   quit (without saving)
Zones are stored in floor metres, so they do not depend on the image scale. Then run
tools/register_store.py (or the next run_store.py) to show them on the dashboard.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retailflow.store_live import load_store   # noqa: E402

PPM = 25.0          # must match build_floor_plan.py --ppm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--store", default="scenes/stores/smartspaces_071.json")
    args = p.parse_args()
    path = Path(args.store)
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    store = load_store(path)
    photo = cv2.imread(str(store.scene_dir / "floor_photo.png"))
    if photo is None:
        sys.exit("No floor_photo.png: run tools/build_floor_plan.py first.")
    x0, y0 = store.floor_bounds[0], store.floor_bounds[1]
    to_px = lambda x, y: (int((x - x0) * PPM), int((y - y0) * PPM))
    to_m = lambda u, v: [round(x0 + u / PPM, 2), round(y0 + v / PPM, 2)]
    zones = [dict(z) for z in raw.get("zones") or []]
    current: list = []

    def redraw():
        img = photo.copy()
        for z in zones:
            pts = np.array([to_px(*pt) for pt in z["polygon"]], np.int32)
            cv2.polylines(img, [pts], True, (0, 220, 255), 2)
            cv2.putText(img, z["name"], tuple(pts[0] + [5, 20]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
        if current:
            pts = np.array([to_px(*pt) for pt in current], np.int32)
            cv2.polylines(img, [pts], False, (80, 255, 80), 2)
            for q in pts:
                cv2.circle(img, tuple(q), 4, (80, 255, 80), -1)
        cv2.imshow("store zones", img)

    def click(event, u, v, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append(to_m(u, v))
            redraw()

    cv2.namedWindow("store zones")
    cv2.setMouseCallback("store zones", click)
    redraw()
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k in (13, 10) and len(current) >= 3:
            name = input("zone name: ").strip().replace(" ", "_") or f"zone_{len(zones) + 1}"
            zones.append({"name": name, "polygon": current.copy()})
            current.clear()
            redraw()
        elif k == ord("u") and current:
            current.pop()
            redraw()
        elif k == ord("d") and zones:
            print("deleted", zones.pop()["name"])
            redraw()
        elif k == ord("s"):
            raw["zones"] = zones
            path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
            print(f"saved {len(zones)} zones to {path}")
        elif k == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
