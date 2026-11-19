"""Tests for retailflow/sender.py against a tiny fake API (standard library HTTP server)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from retailflow.config import scene_from_dict
from retailflow.events import Event
from retailflow.sender import ApiEventSender

SCENE = scene_from_dict({"camera_id": "cam", "frame_size": [100, 100],
                         "line": {"p1": [0, 50], "p2": [100, 50], "in_point": [50, 80]}})


class FakeApi:
    def __init__(self):
        self.calls: list[tuple[str, str, object]] = []
        self.fail_events = False
        self.failed = 0
        api = self

        class H(BaseHTTPRequestHandler):
            def _handle(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"null")
                if self.path.endswith("/events") and api.fail_events:
                    api.failed += 1
                    self.send_response(503); self.end_headers(); return
                api.calls.append((self.command, self.path, body))   # successful calls only
                out = {"session_id": "s1"} if self.path == "/api/sessions" else {}
                data = json.dumps(out).encode()
                self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers()
                self.wfile.write(data)

            do_PUT = do_POST = _handle

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def events_sent(self):
        return [e for m, p, b in self.calls if p.endswith("/events") and m == "POST" for e in b]


@pytest.fixture
def api():
    a = FakeApi()
    yield a
    a.server.shutdown()


def ev(i):
    return Event("cam", i, i / 10, "line_cross", track_id=i, direction="in")


def test_registers_camera_and_session(api):
    s = ApiEventSender(api.url, SCENE, "video.mp4", 25.0)
    assert s.session_id == "s1"
    assert api.calls[0][:2] == ("PUT", "/api/cameras/cam")
    assert api.calls[1][2]["source"] == "video.mp4"


def test_batches_and_flushes_on_close(api):
    s = ApiEventSender(api.url, SCENE, "v", 25.0, batch=3, flush_s=999)
    for i in range(7):
        s.write(ev(i))
    assert len(api.events_sent()) == 6          # two full batches of 3
    s.close()
    sent = api.events_sent()
    assert len(sent) == 7 and "camera_id" not in sent[0] and sent[0]["type"] == "line_cross"


def test_keeps_events_when_api_fails_then_retries(api):
    s = ApiEventSender(api.url, SCENE, "v", 25.0, batch=2, flush_s=999)
    api.fail_events = True
    for i in range(4):
        s.write(ev(i))
    assert api.events_sent() == [] and len(s.buffer) == 4
    assert api.failed == 1          # after a failure it waits retry_s instead of retrying on every event
    api.fail_events = False
    assert s.flush() and s.sent == 4 and s.buffer == []
    assert [e["frame"] for e in api.events_sent()] == [0, 1, 2, 3]


def test_localhost_is_rewritten_to_ipv4(api):
    s = ApiEventSender(api.url.replace("127.0.0.1", "localhost"), SCENE, "v", 25.0)
    assert s.base == api.url and s.session_id == "s1"


def test_finish_sends_summary_and_heatmap(api):
    s = ApiEventSender(api.url, SCENE, "v", 25.0)
    s.finish({"frames": 10}, np.ones((2, 3)), cell=4.0, background=np.zeros((4, 4, 3), np.uint8))
    m, p, body = api.calls[-1]
    assert p == "/api/sessions/s1/finish" and body["summary"]["frames"] == 10
    assert body["heatmap"]["grid"] == [[1.0] * 3] * 2 and body["background_png_b64"]
