"""Make a top-down floor plan of a store from its calibrated cameras.

    python tools/build_floor_plan.py --store scenes/stores/smartspaces_071.json

For each camera: a people-free background (median of frames spread over the video),
projected straight down onto the floor with the camera's homography (floor metres ->
image pixels). The cameras are stitched: each floor spot comes from the camera nearest
to it (the least stretched view). Only the floor is correct: anything standing on it
(shelves, counters) is smeared away from the camera, but its footprint and the floor
pattern line up across cameras.

Writes, next to the scene (data/, not committed):
  floor_photo.png       the stitched top-down floor
  floor_blueprint.png   its outlines as light lines on blueprint blue (for the dashboard)
  floor_coverage.png    which camera each spot comes from
Scale: --ppm pixels per metre over the store's floor_bounds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retailflow.heatmap import median_background   # noqa: E402
from retailflow.store_live import load_store         # noqa: E402

BLUE = (87, 47, 15)            # BGR of the dashboard's blueprint paper (#0f2f57)
LINE = (255, 230, 200)         # light blue-white lines


def background(video: Path, n: int = 15) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 24000
    frames = []
    for i in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if ok:
            frames.append(img)
    cap.release()
    return median_background(frames)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--store", default="scenes/stores/smartspaces_071.json")
    p.add_argument("--ppm", type=float, default=25.0, help="pixels per metre")
    p.add_argument("--max-dist", type=float, default=22.0, help="ignore floor further than this from a camera (m)")
    args = p.parse_args()

    store = load_store(args.store)
    x0, y0, x1, y1 = store.floor_bounds
    W, H = int(round((x1 - x0) * args.ppm)), int(round((y1 - y0) * args.ppm))
    A = np.array([[1 / args.ppm, 0, x0], [0, 1 / args.ppm, y0], [0, 0, 1]])   # canvas pixel -> floor metres
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    fx, fy = x0 + uu / args.ppm, y0 + vv / args.ppm

    photo = np.zeros((H, W, 3), np.uint8)
    best = np.full((H, W), np.inf)
    owner = np.full((H, W), -1, int)
    for k, cam in enumerate(store.cameras):
        bg = background(store.scene_dir / cam / "video.mp4")
        M = store.homographies[cam] @ A                     # canvas pixel -> camera image pixel
        warped = cv2.warpPerspective(bg, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        # valid: the floor point projects in front of the camera and inside its image
        q = np.stack([uu, vv, np.ones_like(uu)], -1) @ M.T
        z = q[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            iu, iv = q[..., 0] / z, q[..., 1] / z
        # "In front of the camera" is the sign of z for a floor point it surely sees (bottom-centre of
        # its image); a homography may be scaled by a negative number, so z > 0 alone is not enough.
        Hc = store.homographies[cam]
        f = np.linalg.inv(Hc) @ np.array([bg.shape[1] / 2, bg.shape[0] - 1, 1.0])
        sign = np.sign((Hc @ (f / f[2]))[2])
        valid = (z * sign > 0) & (iu >= 0) & (iu < bg.shape[1]) & (iv >= 0) & (iv < bg.shape[0])
        cx, cy = store.camera_xy[cam]
        dist = np.hypot(fx - cx, fy - cy)
        take = valid & (dist < args.max_dist) & (dist < best)
        photo[take] = warped[take]
        best[take] = dist[take]
        owner[take] = k
        print(f"{cam}: covers {valid.mean() * 100:.0f}% of the canvas, nearest camera for {take.mean() * 100:.0f}%",
              flush=True)

    out = store.scene_dir
    cv2.imwrite(str(out / "floor_photo.png"), photo)

    # Blueprint: edges of the floor photo (floor tiles fade out, fixtures and walls stay),
    # plus the outline of the area the cameras see.
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 40, 7)
    edges = cv2.Canny(gray, 40, 110)
    seen = (owner >= 0).astype(np.uint8)
    seen = cv2.morphologyEx(seen, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    edges[cv2.erode(seen, np.ones((5, 5), np.uint8)) == 0] = 0          # no edges at the seams of the visible area
    # drop specks (the floor tiles' dot pattern): keep only edge pieces of some length
    n, lab, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] < 25
    small[0] = False
    edges[small[lab]] = 0
    contours, _ = cv2.findContours(seen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bp = np.zeros((H, W, 4), np.uint8)                                    # transparent PNG: lines only
    bp[edges > 0] = (*LINE, 170)
    outline = np.zeros((H, W), np.uint8)
    cv2.drawContours(outline, contours, -1, 255, 2)
    bp[outline > 0] = (*LINE, 255)
    cv2.imwrite(str(out / "floor_blueprint.png"), bp)

    rng = np.random.default_rng(0)
    colours = rng.integers(60, 255, (len(store.cameras), 3), dtype=np.uint8)
    cov = np.zeros((H, W, 3), np.uint8)
    cov[owner >= 0] = colours[owner[owner >= 0]]
    cv2.imwrite(str(out / "floor_coverage.png"), cv2.addWeighted(photo, 0.6, cov, 0.4, 0))
    print(f"floor plan {W}x{H} px ({args.ppm:g} px/m) -> {out}/floor_photo.png, floor_blueprint.png, floor_coverage.png")


if __name__ == "__main__":
    main()
