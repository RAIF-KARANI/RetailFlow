"""Colour descriptors and appearance rules in cross-camera linking."""

import numpy as np
import pandas as pd

from retailflow.appearance import box_descriptor, hellinger, pick_views
from retailflow.mtmc import MTMCConfig, link
from tests.test_mtmc import FPS, ids, walk


def person(top_bgr, legs_bgr, size=(240, 100)):
    img = np.full((400, 300, 3), 128, np.uint8)
    h, w = size
    img[50:50 + h // 2, 100:100 + w] = top_bgr
    img[50 + h // 2:50 + h, 100:100 + w] = legs_bgr
    return img, (100, 50, 100 + w, 50 + h)


def describe(top, legs):
    img, box = person(top, legs)
    return box_descriptor(img, *box)


def test_same_clothes_are_close_different_clothes_far():
    red, blue, green = (0, 0, 200), (200, 0, 0), (0, 200, 0)
    a, b = describe(red, blue), describe(red, blue)
    c, d = describe(blue, red), describe(green, green)
    assert hellinger(a, b) < 0.05
    assert hellinger(a, c) > 0.6          # swapped shirt and trousers: torso and legs compared separately
    assert hellinger(a, d) > 0.6


def test_small_boxes_have_no_descriptor():
    img, _ = person((0, 0, 200), (200, 0, 0))
    assert box_descriptor(img, 100, 50, 130, 90) is None


def test_views_hidden_behind_another_person_are_not_used():
    t = pd.DataFrame({"frame": [0, 0, 1], "track_id": [1, 2, 1],
                      "x1": [0, 10, 0], "y1": [0, 0, 0], "x2": [100, 110, 100], "y2": [200, 200, 200]})
    v = pick_views(t)
    assert list(v.frame) == [1]           # frame 0: the two boxes overlap 90%


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def test_long_handoff_needs_matching_colours():
    a = walk(1, range(0, 100), (0, 0), (1, 0))                 # ends at x ~ 3.3 m, frame 99
    b = walk(4, range(400, 500), (8, 0), (1, 0))               # 10 s later, 4.7 m on: walkable
    red, blue = unit([1, 0, 0]), unit([0, 1, 0])
    same = link({"camA": a, "camB": b}, FPS, appearance={("camA", 1): red, ("camB", 4): red})
    assert ids(same, "camA", 1) == ids(same, "camB", 4)
    diff = link({"camA": a, "camB": b}, FPS, appearance={("camA", 1): red, ("camB", 4): blue})
    assert ids(diff, "camA", 1) != ids(diff, "camB", 4)
    no_app = link({"camA": a, "camB": b}, FPS)                 # 10 s is beyond the 4 s position-only gap
    assert ids(no_app, "camA", 1) != ids(no_app, "camB", 4)


def test_colour_veto_blocks_a_position_link():
    a = walk(1, range(0, 300), (0, 0), (1, 0))
    b = walk(7, range(0, 300), (0, 0.1), (1, 0), seed=5)       # same place: position says "same"
    red, blue = unit([1, 0, 0]), unit([0, 1, 0])
    vetoed = link({"camA": a, "camB": b}, FPS, appearance={("camA", 1): red, ("camB", 7): blue})
    assert ids(vetoed, "camA", 1) != ids(vetoed, "camB", 7)
    allowed = link({"camA": a, "camB": b}, FPS, MTMCConfig(app_veto=None),
                   appearance={("camA", 1): red, ("camB", 7): blue})
    assert ids(allowed, "camA", 1) == ids(allowed, "camB", 7)
