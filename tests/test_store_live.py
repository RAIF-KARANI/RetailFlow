"""Live store state: stable person numbers, frozen old tracks, dashboard summary."""

from pathlib import Path

import numpy as np

from retailflow.detector import TrackedPerson
from retailflow.store_live import StoreConfig, StoreLinker, camera_floor_position

FPS = 30.0
H = np.array([[100.0, 0, 960], [0, 100.0, 540], [0, 0, 1]])     # 1 m = 100 px around the image centre


def store(cams=("camera_a", "camera_b")):
    return StoreConfig("test_store", "", Path("."), list(cams), {c: H for c in cams},
                       {c: (0.0, 0.0) for c in cams}, (-10, -10, 10, 10), fps=FPS)


def box(tid, x, y):
    """A person standing at floor (x, y) metres, 1.7 m tall in pixels."""
    u, v = 960 + 100 * x, 540 + 100 * y
    return TrackedPerson(tid, u - 20, v - 170, u + 20, v, 0.9)


def feed(lk, frames, people_a, people_b):
    for f in frames:
        lk.add("camera_a", f, [box(t, x(f), y(f)) for t, x, y in people_a])
        lk.add("camera_b", f, [box(t, x(f), y(f)) for t, x, y in people_b])


def walker(x0, y0, vx):
    return (lambda f: x0 + vx * f / FPS), (lambda f: y0)


def test_same_person_in_two_cameras_gets_one_number_that_stays():
    lk = StoreLinker(store(), appearance=False)
    xa, ya = walker(-3, 0, 0.5)
    xb, yb = walker(3, 2, -0.5)
    feed(lk, range(0, 150, 6), [(1, xa, ya), (2, xb, yb)], [(7, xa, ya), (8, xb, yb)])
    links = lk.relink()
    num = {(r.camera, r.track_id): r.person for r in links.itertuples()}
    assert num[("camera_a", 1)] == num[("camera_b", 7)] != num[("camera_a", 2)] == num[("camera_b", 8)]
    feed(lk, range(150, 300, 6), [(1, xa, ya), (2, xb, yb)], [(7, xa, ya), (8, xb, yb)])
    again = {(r.camera, r.track_id): r.person for r in lk.relink().itertuples()}
    assert again == num                                  # re-linking keeps the numbers


def test_old_tracks_keep_their_number_outside_the_window():
    lk = StoreLinker(store(), appearance=False, window_s=5)
    x, y = walker(0, 0, 0.0)
    feed(lk, range(0, 60, 6), [(1, x, y)], [(4, x, y)])
    first = {(r.camera, r.track_id): r.person for r in lk.relink().itertuples()}
    x2, y2 = walker(5, 5, 0.0)
    feed(lk, range(600, 660, 6), [(9, x2, y2)], [])      # 18 s later: tracks 1 and 4 are outside the window
    later = {(r.camera, r.track_id): r.person for r in lk.relink().itertuples()}
    assert later[("camera_a", 1)] == first[("camera_a", 1)] == later[("camera_b", 4)]
    assert later[("camera_a", 9)] != later[("camera_a", 1)]


def test_state_summarises_people_now_trails_and_camera_order():
    lk = StoreLinker(store(), appearance=False)
    xa, ya = walker(-3, 0, 0.5)
    lk_frames = range(0, 300, 6)
    for f in lk_frames:                                   # camera_a sees the person first, camera_b from 2 s on
        lk.add("camera_a", f, [box(1, xa(f), ya(f))])
        if f >= 60:
            lk.add("camera_b", f, [box(7, xa(f), ya(f))])
    st = lk.state()
    assert st["unique_people"] == 1 and len(st["people_now"]) == 1
    p = st["people"][0]
    assert p["cameras"] == ["a", "b"] and p["seen_s"] >= 9
    now = st["people_now"][0]
    assert abs(now["x"] - xa(294)) < 0.6 and now["cameras"] == ["a", "b"]
    assert len(st["trails"][str(p["id"])] if str(p["id"]) in st["trails"] else st["trails"][p["id"]]) >= 5
    assert st["camera_moves"]["matrix"] == [[0, 1], [0, 0]]


def test_camera_position_from_projection_matrix():
    # camera at (2, -3) metres, 4 m up, looking straight down
    K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
    R = np.diag([1.0, -1.0, -1.0])
    C = np.array([2.0, -3.0, 4.0])
    P = K @ np.c_[R, -R @ C]
    x, y = camera_floor_position(P)
    assert abs(x - 2) < 1e-6 and abs(y + 3) < 1e-6


def test_floor_heatmap_counts_each_person_once_per_second():
    lk = StoreLinker(store(), appearance=False)
    for f in range(0, 300, 6):                          # 10 s, standing at (1.2, -0.3), two cameras
        lk.add("camera_a", f, [box(1, 1.2, -0.3)])
        lk.add("camera_b", f, [box(5, 1.2, -0.3)])
    h = lk.state()["heatmap"]
    assert h["total_s"] == 10                           # not 20: one person, not two cameras' worth
    assert len(h["cells"]) == 1
    r, c, v = h["cells"][0]
    assert v == 10 and abs(h["x0"] + (c + 0.5) * h["cell"] - 1.2) <= h["cell"] / 2 + 1e-9
    assert abs(h["y0"] + (r + 0.5) * h["cell"] + 0.3) <= h["cell"] / 2 + 1e-9
