"""Host runner for the dashboard's Add camera flow (runner.py): upload, frame, scene, checks."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import cv2
import numpy as np
import pytest

import runner

ORIGIN = "http://localhost:3001"


@pytest.fixture()
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(runner, "SCENES", tmp_path / "scenes")
    monkeypatch.setattr(runner, "OUT", tmp_path / "out")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), runner.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", tmp_path
    httpd.shutdown()


def call(base, method, path, body=None, origin=ORIGIN, ctype="application/json"):
    req = urllib.request.Request(base + path, data=body, method=method,
                                 headers={"Origin": origin, "Content-Type": ctype} if origin else {"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def make_video(path, w=160, h=120, n=10):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for i in range(n):
        vw.write(np.full((h, w, 3), i * 20, np.uint8))
    vw.release()
    return path.read_bytes()


def test_upload_frame_default_scene_and_save(server):
    base, tmp = server
    data = make_video(tmp / "v.mp4")
    code, body, headers = call(base, "POST", "/api/cameras/shop_1/video?filename=v.mp4", data, ctype="video/mp4")
    assert code == 200, body
    info = json.loads(body)
    assert (info["width"], info["height"]) == (160, 120)
    assert headers.get("Access-Control-Allow-Origin") == ORIGIN
    assert call(base, "POST", "/api/cameras/shop_1/video?filename=v.mp4", data)[0] == 409   # no silent overwrite
    assert call(base, "POST", "/api/cameras/shop_1/video?filename=v.mp4&replace=1", data)[0] == 200

    code, jpg, h = call(base, "GET", "/api/cameras/shop_1/frame?n=3")
    assert code == 200 and h["Content-Type"] == "image/jpeg" and jpg[:2] == b"\xff\xd8"

    scene = json.loads(call(base, "GET", "/api/cameras/shop_1/scene")[1])
    assert scene["exists"] is False and scene["scene"]["frame_size"] == [160, 120]
    s = scene["scene"]
    s["zones"] = [{"name": "door", "polygon": [[10, 10], [60, 10], [60, 60]]}]
    code, body, _ = call(base, "PUT", "/api/cameras/shop_1/scene", json.dumps(s).encode())
    assert code == 200, body
    saved = json.loads((tmp / "scenes" / "shop_1.json").read_text())
    assert saved["camera_id"] == "shop_1" and saved["zones"][0]["name"] == "door"
    assert json.loads(call(base, "GET", "/api/cameras/shop_1/scene")[1])["exists"] is True


def test_rejects_bad_input(server):
    base, tmp = server
    assert call(base, "GET", "/api/cameras/..%2Fetc/scene")[0] == 400          # camera name
    assert call(base, "POST", "/api/cameras/a/video?filename=x.exe", b"x")[0] == 400
    cv2.imwrite(str(tmp / "img.jpg"), np.zeros((50, 50, 3), np.uint8))
    code, body, _ = call(base, "POST", "/api/cameras/a/video?filename=x.mp4", (tmp / "img.jpg").read_bytes())
    assert code == 400 and b"not a video" in body                              # an image renamed .mp4
    assert not list((tmp / "uploads").glob("a.*"))
    make_video(tmp / "v.mp4")
    call(base, "POST", "/api/cameras/a/video?filename=v.mp4", (tmp / "v.mp4").read_bytes())
    bad = {"frame_size": [160, 120], "line": {"p1": [0, 0]}}
    assert call(base, "PUT", "/api/cameras/a/scene", json.dumps(bad).encode())[0] == 400
    assert call(base, "POST", "/api/cameras/a/start", b'{"minutes": 1}')[0] == 400  # no scene saved yet


def test_other_sites_cannot_write(server):
    base, _ = server
    code, _, h = call(base, "POST", "/api/cameras/a/start", b"{}", origin="http://evil.example")
    assert code == 403 and "Access-Control-Allow-Origin" not in h
    assert call(base, "OPTIONS", "/api/cameras/a/start", origin="http://evil.example")[0] == 403
