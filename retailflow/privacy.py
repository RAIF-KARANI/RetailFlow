"""Head blurring for any image the system saves or shows (`--blur-heads`).

The detector finds whole bodies, not faces. In an angled CCTV view a face can
be turned away, tiny or half hidden, so a face detector would miss many of
them. Instead we blur the **top part of every person box**, where the head is:

    head height = head_frac x box height (default 0.22), full box width
                  plus a small margin on each side

For an upright adult the head is about 1/7-1/8 of body height (~0.13-0.14),
so 0.22 leaves margin for boxes that are a little too tall, bent heads and
hair. Pixelation is used by default: it cannot be undone by sharpening, unlike
a light Gaussian blur.

Blurring is applied to the OUTPUT image only (annotated video, preview).
Detection and tracking always run on the original frame, so accuracy is the same.
"""

from __future__ import annotations

import cv2
import numpy as np


def head_box(x1, y1, x2, y2, frame_w, frame_h, head_frac=0.22, margin=0.15):
    """Integer (x1, y1, x2, y2) of the head region of a person box, clipped to the frame."""
    w, h = x2 - x1, y2 - y1
    hx1 = int(max(0, np.floor(x1 - margin * w)))
    hx2 = int(min(frame_w, np.ceil(x2 + margin * w)))
    hy1 = int(max(0, np.floor(y1 - 0.05 * h)))
    hy2 = int(min(frame_h, np.ceil(y1 + head_frac * h)))
    return hx1, hy1, hx2, hy2


def blur_heads(img: np.ndarray, boxes, head_frac=0.22, blocks: int = 6) -> np.ndarray:
    """Pixelate the head region of each (x1, y1, x2, y2) box, in place. Returns img."""
    H, W = img.shape[:2]
    for b in boxes:
        x1, y1, x2, y2 = head_box(*b, W, H, head_frac)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = img[y1:y2, x1:x2]
        small = cv2.resize(roi, (min(blocks, roi.shape[1]), min(blocks, roi.shape[0])),
                           interpolation=cv2.INTER_AREA)
        img[y1:y2, x1:x2] = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    return img
