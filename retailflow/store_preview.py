"""Live camera view of a store run (run_store.py --preview): all cameras as a mosaic, or one.

Each box is labelled with the person's STORE-WIDE number (from the linking process), so
the same person keeps the same number and colour from camera to camera. A track the
linker has not numbered yet is drawn grey. Heads are pixelated by default.
Only drawn while someone watches (retailflow/preview.py).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from retailflow.privacy import blur_heads

# BGR, bright enough on video; colour = store-wide number
PALETTE = [(255, 180, 90), (80, 200, 255), (210, 140, 255), (120, 230, 140), (100, 130, 255),
           (255, 230, 120), (200, 255, 120), (255, 140, 200)]
GREY = (170, 170, 170)


def _annotate(img: np.ndarray, people, numbers: dict, cam: str, scale: float, blur: bool) -> None:
    boxes = [(p.x1 * scale, p.y1 * scale, p.x2 * scale, p.y2 * scale) for p in people]
    if blur:
        blur_heads(img, boxes)
    th = max(2, round(img.shape[1] / 480))           # line and text size follow the image width
    fs = max(0.4, img.shape[1] / 1100)
    for p, (x1, y1, x2, y2) in zip(people, boxes):
        n = numbers.get((cam, p.track_id))
        col = PALETTE[n % len(PALETTE)] if n is not None else GREY
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, th)
        label = f"#{n}" if n is not None else "?"
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        cv2.rectangle(img, (int(x1), int(y1) - tht - 6), (int(x1) + tw + 6, int(y1)), col, -1)
        cv2.putText(img, label, (int(x1) + 3, int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), th,
                    cv2.LINE_AA)


def _caption(img: np.ndarray, text: str) -> None:
    fs = max(0.5, img.shape[1] / 900)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    cv2.rectangle(img, (0, 0), (tw + 12, th + 12), (15, 47, 87), -1)
    cv2.putText(img, text, (6, th + 6), cv2.FONT_HERSHEY_SIMPLEX, fs, (235, 244, 255), 1, cv2.LINE_AA)


def mosaic(images: dict, people: dict, numbers: dict, cams: list[str], t_s: float,
           tile_w: int = 480, blur: bool = True) -> np.ndarray:
    """All cameras in a grid (newest analysed frame of each), boxes with store-wide numbers."""
    cols = min(4, len(cams))
    rows = math.ceil(len(cams) / cols)
    first = next((im for im in images.values() if im is not None), None)
    h0, w0 = (first.shape[:2] if first is not None else (1080, 1920))
    tile_h = round(tile_w * h0 / w0)
    out = np.full((rows * tile_h, cols * tile_w, 3), 24, np.uint8)
    for i, cam in enumerate(cams):
        im = images.get(cam)
        if im is None:
            continue
        tile = cv2.resize(im, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        _annotate(tile, people.get(cam, []), numbers, cam, tile_w / im.shape[1], blur)
        _caption(tile, cam.replace("camera_", "cam "))
        r, c = divmod(i, cols)
        out[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = tile
    _caption(out, f"video time {t_s:6.1f} s   #n = store-wide person")
    return out


def single(image: np.ndarray, people, numbers: dict, cam: str, t_s: float, blur: bool = True) -> np.ndarray:
    """One camera at full size (copied: the frame itself is not drawn on)."""
    img = image.copy()
    _annotate(img, people, numbers, cam, 1.0, blur)
    _caption(img, f"{cam.replace('camera_', 'cam ')}   video time {t_s:6.1f} s")
    return img
