"""What a tracked person looks like: colour histograms of upper and lower body.

Used to join pieces of one person across cameras when floor position alone
cannot decide (retailflow/mtmc.py). Deliberately simple and dependency-free:
HSV hue/saturation histograms of the torso and legs, averaged over several
clear views of a track, compared with the Hellinger distance (0 = identical,
1 = nothing in common).

Privacy: descriptors are computed and used in memory only and never written
to disk (project rule: appearance features live only during a session). They
describe clothing colours, not faces or identities.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

H_BINS, S_BINS = 16, 4
MIN_H_PX = 60            # smaller boxes have too few pixels for colours


def box_descriptor(img: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray | None:
    """Square-rooted, normalised HS histograms of torso (15-50% of height) and legs (50-90%)."""
    h, w = img.shape[:2]
    x1, x2 = int(max(0, x1)), int(min(w, x2))
    y1i, y2i = int(max(0, y1)), int(min(h, y2))
    bh = y2 - y1
    if bh < MIN_H_PX or x2 - x1 < 8:
        return None
    # keep the middle 60% of the width: the edges of a box are mostly background
    cx, half = (x1 + x2) / 2, 0.3 * (x2 - x1)
    xa, xb = int(cx - half), int(cx + half)
    parts = []
    for a, b in ((0.15, 0.50), (0.50, 0.90)):
        ya, yb = int(max(y1i, y1 + a * bh)), int(min(y2i, y1 + b * bh))
        if yb - ya < 4:
            return None
        hsv = cv2.cvtColor(img[ya:yb, xa:xb], cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 30), (180, 255, 255))        # ignore near-black pixels
        hist = cv2.calcHist([hsv], [0, 1], mask, [H_BINS, S_BINS], [0, 180, 0, 256]).ravel()
        s = hist.sum()
        if s <= 0:
            return None
        parts.append(np.sqrt(hist / s))
    return np.concatenate(parts) / np.sqrt(2.0)          # unit length: two unit halves


def hellinger(a: np.ndarray, b: np.ndarray) -> float:
    """0 = same colours, 1 = no overlap (inputs are square-rooted unit histograms)."""
    return float(np.sqrt(max(0.0, 1.0 - float(a @ b))))


def pick_views(tracks: pd.DataFrame, per_track: int = 6, max_overlap: float = 0.2) -> pd.DataFrame:
    """Rows of `tracks` to describe: per track, up to `per_track` clear, large boxes spread over its life.

    A box is clear when it overlaps no other box of the same frame by more than
    `max_overlap` of its own area (another person in front mixes their colours in).
    """
    t = tracks.assign(h=tracks.y2 - tracks.y1)
    t = t[t.h >= MIN_H_PX]
    clear = np.ones(len(t), bool)
    idx = {i: k for k, i in enumerate(t.index)}
    for _, g in t.groupby("frame"):
        if len(g) < 2:
            continue
        b = g[["x1", "y1", "x2", "y2"]].to_numpy()
        ix = np.clip(np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0]), 0, None)
        iy = np.clip(np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1]), 0, None)
        inter = ix * iy
        np.fill_diagonal(inter, 0)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        bad = (inter / area[:, None]).max(axis=1) > max_overlap
        for i, flag in zip(g.index, bad):
            if flag:
                clear[idx[i]] = False
    t = t[clear]
    picks = []
    for _, g in t.groupby("track_id"):
        if len(g) > per_track:
            g = g.sort_values("frame").iloc[np.linspace(0, len(g) - 1, per_track).round().astype(int)]
        picks.append(g)
    return pd.concat(picks) if picks else t.iloc[0:0]


def track_descriptors(video: str, tracks: pd.DataFrame, per_track: int = 6) -> dict[int, np.ndarray]:
    """track_id -> mean descriptor of its picked views (decodes the video once, in order)."""
    views = pick_views(tracks, per_track)
    want = {f: g for f, g in views.groupby("frame")}
    if not want:
        return {}
    last = max(want)
    acc: dict[int, list] = {}
    cap = cv2.VideoCapture(video)
    f = 0
    while f <= last:
        if f in want:
            ok, img = cap.read()
            if not ok:
                break
            for r in want[f].itertuples():
                d = box_descriptor(img, r.x1, r.y1, r.x2, r.y2)
                if d is not None:
                    acc.setdefault(int(r.track_id), []).append(d)
        elif not cap.grab():
            break
        f += 1
    cap.release()
    out = {}
    for tid, ds in acc.items():
        m = np.mean(ds, axis=0)
        out[tid] = m / (np.linalg.norm(m) or 1.0)
    return out
