"""Measure camera motion per MOT sequence, to select static-camera sequences.

Method (per sampled pair of consecutive frames):
  1. Mask out ground-truth person boxes, so only the BACKGROUND is used
     (moving people would otherwise look like camera motion).
  2. Detect ORB keypoints in both frames and match them.
  3. Fit a similarity transform (rotation + translation + scale) with RANSAC.
  4. Record how far the transform moves the image centre, in pixels.

A static camera gives ~0 px per frame (noise only); a handheld or vehicle
camera gives several pixels. Run:  python tools/check_camera_motion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow "import retailflow"
from retailflow.groundtruth import discover_sequences  # noqa: E402

STEP = 10          # sample every 10th frame pair
STATIC_PX = 1.0    # median shift below this (px/frame) => static camera


def background_mask(shape: tuple[int, int], boxes: np.ndarray) -> np.ndarray:
    """255 = background (use), 0 = inside a person box (ignore)."""
    mask = np.full(shape, 255, np.uint8)
    for x1, y1, x2, y2 in boxes.astype(int):
        mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 0
    return mask


def pair_shift(a: np.ndarray, b: np.ndarray, mask_a, mask_b, orb, matcher) -> float | None:
    """Pixel displacement of the image centre between frames a and b."""
    ka, da = orb.detectAndCompute(a, mask_a)
    kb, db = orb.detectAndCompute(b, mask_b)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return None
    matches = matcher.match(da, db)
    if len(matches) < 20:
        return None
    pa = np.float32([ka[m.queryIdx].pt for m in matches])
    pb = np.float32([kb[m.trainIdx].pt for m in matches])
    M, _ = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None:
        return None
    h, w = a.shape
    centre = np.array([w / 2, h / 2, 1.0])
    return float(np.linalg.norm(M @ centre - centre[:2]))


def main() -> None:
    orb = cv2.ORB_create(2000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    print(f"{'sequence':<18}{'pairs':>6}{'median px':>11}{'p90 px':>9}  verdict")
    for seq in discover_sequences():
        if seq.dataset == "caviar":
            continue
        gt = seq.load_gt()
        frames = sorted(seq.media.glob("*.jpg"))
        shifts = []
        for i in range(0, len(frames) - 1, STEP):
            a = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
            b = cv2.imread(str(frames[i + 1]), cv2.IMREAD_GRAYSCALE)
            boxes = lambda f: gt.loc[gt.frame == f, ["x1", "y1", "x2", "y2"]].to_numpy()  # noqa: E731
            s = pair_shift(a, b, background_mask(a.shape, boxes(i)),
                           background_mask(b.shape, boxes(i + 1)), orb, matcher)
            if s is not None:
                shifts.append(s)
        med, p90 = np.median(shifts), np.percentile(shifts, 90)
        verdict = "STATIC" if med < STATIC_PX else "MOVING"
        print(f"{seq.name:<18}{len(shifts):>6}{med:>11.2f}{p90:>9.2f}  {verdict}")


if __name__ == "__main__":
    main()
