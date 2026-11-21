"""Live camera view of a store: mosaic layout and store-wide labels."""

import numpy as np

from retailflow.detector import TrackedPerson
from retailflow.store_preview import GREY, PALETTE, mosaic, single


def frame(v=60):
    return np.full((1080, 1920, 3), v, np.uint8)


def test_mosaic_has_a_tile_per_camera_and_leaves_frames_untouched():
    cams = [f"camera_{i}" for i in range(15)]
    imgs = {c: frame() for c in cams}
    people = {"camera_0": [TrackedPerson(1, 900, 300, 1000, 700, 0.9)]}
    out = mosaic(imgs, people, {("camera_0", 1): 7}, cams, 12.0)
    assert out.shape == (4 * 270, 4 * 480, 3)               # 15 cameras -> 4 x 4 grid of 480 x 270 tiles
    assert all((im == 60).all() for im in imgs.values())    # drawing never touches the camera frames


def test_single_view_colours_boxes_by_store_wide_number():
    img = frame()
    p = TrackedPerson(3, 900, 300, 1000, 700, 0.9)
    labelled = single(img, [p], {("camera_a", 3): 5}, "camera_a", 1.0, blur=False)
    unknown = single(img, [p], {}, "camera_a", 1.0, blur=False)
    assert tuple(labelled[500, 900]) == PALETTE[5 % len(PALETTE)]   # left edge of the box
    assert tuple(unknown[500, 900]) == GREY
    assert (img == 60).all()
