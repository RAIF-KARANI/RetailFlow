"""End-to-end test of the running API + PostgreSQL (docker compose up -d db api).

Skipped automatically when the API is not reachable at RETAILFLOW_API
(default http://127.0.0.1:8000). Uses a throw-away camera and deletes it.
"""

import json
import os
import urllib.error
import urllib.request
import uuid

import pytest

API = os.environ.get("RETAILFLOW_API", "http://127.0.0.1:8000")   # not "localhost": IPv6 fallback is slow


def call(method, path, body=None):
    req = urllib.request.Request(f"{API}{path}", method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read() or b"null")


def api_up():
    try:
        return call("GET", "/api/health")["status"] == "ok"
    except (urllib.error.URLError, OSError):
        return False


pytestmark = pytest.mark.skipif(not api_up(), reason=f"API not running at {API}")


@pytest.fixture
def camera():
    cam = f"test_{uuid.uuid4().hex[:8]}"
    call("PUT", f"/api/cameras/{cam}", {
        "camera_id": cam, "frame_size": [100, 100],
        "line": {"p1": [0, 50], "p2": [100, 50], "in_point": [50, 80], "band_px": 2},
        "zones": [{"name": "door", "polygon": [[0, 0], [50, 0], [50, 50], [0, 50]]}]})
    yield cam
    call("DELETE", f"/api/cameras/{cam}")


def test_staff_and_static_flags_are_excluded_retroactively(camera):
    sid = call("POST", "/api/sessions", {"camera_id": camera, "source": "clip.mp4", "video_fps": 10})["session_id"]
    E = lambda **k: {"frame": int(k["timestamp"] * 10), **k}
    call("POST", f"/api/sessions/{sid}/events", [
        E(timestamp=1.0, type="line_cross", track_id=1, direction="in"),
        E(timestamp=2.0, type="line_cross", track_id=2, direction="in"),     # staff, flagged later
        E(timestamp=3.0, type="line_cross", track_id=3, direction="out"),
        E(timestamp=1.0, type="zone_enter", track_id=1, zone="door"),
        E(timestamp=5.0, type="zone_exit", track_id=1, zone="door", value=4.1),
        E(timestamp=0.0, type="zone_enter", track_id=9, zone="door"),         # mannequin
        E(timestamp=12.0, type="static_flag", track_id=9),
        E(timestamp=15.0, type="staff_flag", track_id=2, staff=True),
    ])
    call("POST", f"/api/sessions/{sid}/finish", {"summary": {"frames": 200},
                                                 "heatmap": {"cell": 4, "grid": [[0, 1], [2, 3]]}})
    s = call("GET", f"/api/sessions/{sid}/summary")
    assert (s["cust_in"], s["cust_out"], s["all_in"]) == (1, 1, 2)
    assert (s["staff_tracks"], s["static_tracks"]) == (1, 1)
    door = s["zones"][0]
    assert door["visits"] == 1 and door["mean_dwell_s"] == 4.1 and door["occupancy_now"] == 0  # mannequin gone
    assert s["video_seconds"] == 20.0

    counts = call("GET", f"/api/sessions/{sid}/counts?bucket_s=10")
    assert counts[0] == {"t": 0.0, "in": 1, "out": 1}
    occ = call("GET", f"/api/sessions/{sid}/occupancy?bucket_s=10")
    assert occ[0]["door"] == pytest.approx(0.4)          # 4 s of 10 s
    h = call("GET", f"/api/sessions/{sid}/heatmap")
    assert h["grid"] == [[0, 1], [2, 3]] and h["has_background"] is False
    cams = {c["camera_id"]: c for c in call("GET", "/api/cameras")}
    assert cams[camera]["cust_in"] == 1 and cams[camera]["sessions"] == 1


def test_people_in_view_and_total(camera):
    sid = call("POST", "/api/sessions", {"camera_id": camera, "source": "clip.mp4", "video_fps": 10})["session_id"]
    call("POST", f"/api/sessions/{sid}/people", [{"t": 0, "now": 2.0, "total": 0},
                                                 {"t": 1, "now": 3.0, "total": 1},
                                                 {"t": 2, "now": 1.0, "total": 2}])
    s = call("GET", f"/api/sessions/{sid}/summary")
    assert s["people_now"] == 1.0 and s["people_total"] == 2               # live: latest second
    rows = call("GET", f"/api/sessions/{sid}/people?bucket_s=2")
    assert rows == [{"t": 0.0, "now": 2.5, "total": 1}, {"t": 2.0, "now": 1.0, "total": 2}]
    call("POST", f"/api/sessions/{sid}/finish", {"summary": {"frames": 30, "people_total": 1,
                                                             "people_now_mean": 2.0}})
    s = call("GET", f"/api/sessions/{sid}/summary")
    assert s["people_now"] is None and s["people_total"] == 1 and s["people_now_mean"] == 2.0


def test_rejects_unknown_event_type(camera):
    sid = call("POST", "/api/sessions", {"camera_id": camera, "source": "x"})["session_id"]
    with pytest.raises(urllib.error.HTTPError) as e:
        call("POST", f"/api/sessions/{sid}/events", [{"frame": 0, "timestamp": 0, "type": "face_id"}])
    assert e.value.code == 422


def test_unknown_camera_session_is_404():
    with pytest.raises(urllib.error.HTTPError) as e:
        call("POST", "/api/sessions", {"camera_id": f"nope_{uuid.uuid4().hex}", "source": "x"})
    assert e.value.code == 404


def test_store_run_state_and_finish():
    sid = f"test_store_{uuid.uuid4().hex[:8]}"
    cfg = {"store_id": sid, "description": "t", "cameras": ["camera_a", "camera_b"],
           "camera_xy": {"camera_a": [0, 0], "camera_b": [5, 5]}, "floor_bounds": [-10, -10, 10, 10]}
    call("PUT", f"/api/stores/{sid}", cfg)
    try:
        rid = call("POST", f"/api/stores/{sid}/runs", {"run_info": {"cameras": 2}})["run_id"]
        state = {"t": 12.5, "people_now": [{"id": 3, "x": 1.0, "y": 2.0, "cameras": ["a"]}],
                 "unique_people": 1, "people": [], "trails": {"3": [[1, 2]]}}
        call("PUT", f"/api/store-runs/{rid}/state", state)
        r = call("GET", f"/api/store-runs/{rid}")
        assert not r["finished"] and r["state"]["people_now"][0]["id"] == 3
        stores = {s["store_id"]: s for s in call("GET", "/api/stores")}
        assert stores[sid]["live"] and stores[sid]["cameras"] == 2
        call("POST", f"/api/store-runs/{rid}/finish", {"state": {**state, "t": 20.0}, "summary": {"ok": 1}})
        s = call("GET", f"/api/stores/{sid}")
        assert s["config"]["floor_bounds"] == [-10, -10, 10, 10]
        assert s["runs"][0]["finished_at"] and s["runs"][0]["t"] == 20.0
    finally:
        call("DELETE", f"/api/stores/{sid}")
