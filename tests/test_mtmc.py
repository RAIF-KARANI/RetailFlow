"""Cross-camera linking on synthetic floor tracks."""

import numpy as np
import pandas as pd

from retailflow.mtmc import MTMCConfig, floor_points, link

FPS = 30


def walk(track_id, frames, start, velocity, noise=0.05, seed=0):
    rng = np.random.default_rng(seed + track_id)
    f = np.asarray(list(frames))
    xy = np.asarray(start) + np.outer((f - f[0]) / FPS, velocity) + rng.normal(0, noise, (len(f), 2))
    return pd.DataFrame({"frame": f, "track_id": track_id, "xw": xy[:, 0], "yw": xy[:, 1], "usable": True})


def ids(links, cam, tid):
    return int(links[(links.camera == cam) & (links.track_id == tid)].global_id.iloc[0])


def test_same_person_in_two_overlapping_cameras_gets_one_id():
    a = pd.concat([walk(1, range(0, 300), (0, 0), (1, 0)), walk(2, range(0, 300), (0, 5), (0.5, 0))])
    b = pd.concat([walk(7, range(50, 300), (50 / FPS, 0), (1, 0), seed=9),
                   walk(8, range(0, 200), (0, 5), (0.5, 0), seed=9)])
    links = link({"camA": a, "camB": b}, FPS)
    assert ids(links, "camA", 1) == ids(links, "camB", 7)
    assert ids(links, "camA", 2) == ids(links, "camB", 8)
    assert links.global_id.nunique() == 2


def test_two_tracks_of_one_camera_at_the_same_time_are_never_merged():
    # camA sees two people standing 0.4 m apart; camB sees only one of them, in between.
    a = pd.concat([walk(1, range(0, 200), (0, 0), (0, 0)), walk(2, range(0, 200), (0.4, 0), (0, 0))])
    b = walk(5, range(0, 200), (0.2, 0), (0, 0))
    links = link({"camA": a, "camB": b}, FPS)
    assert ids(links, "camA", 1) != ids(links, "camA", 2)


def test_handoff_between_cameras_without_overlap():
    a = walk(1, range(0, 100), (0, 0), (1, 0))                    # ends at about x = 3.3 m
    b = pd.concat([walk(4, range(110, 250), (3.7, 0), (1, 0)),    # 0.33 s later, 0.4 m on: same person
                   walk(6, range(110, 250), (12, 8), (1, 0))])    # far away: someone else
    links = link({"camA": a, "camB": b}, FPS)
    assert ids(links, "camA", 1) == ids(links, "camB", 4)
    assert ids(links, "camA", 1) != ids(links, "camB", 6)
    no_handoff = link({"camA": a, "camB": b}, FPS, MTMCConfig(gap_s=0.0))
    assert no_handoff.global_id.nunique() == 3


def test_floor_points_use_inverse_homography_and_flag_cut_off_feet():
    H = np.array([[100.0, 0, 960], [0, 100.0, 540], [0, 0, 1]])   # 1 m = 100 px, origin at image centre
    t = pd.DataFrame({"frame": [0, 0], "track_id": [1, 2], "x1": [1000, 0], "x2": [1040, 50],
                      "y1": [400, 900], "y2": [740, 1079]})
    p = floor_points(t, H, 1080, MTMCConfig())
    assert np.allclose(p[["xw", "yw"]].iloc[0], [0.6, 2.0])
    assert p.usable.tolist() == [True, False]
