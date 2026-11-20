"""Live browser preview: MJPEG server fed by the pipeline."""

import threading
import time
import urllib.request

import numpy as np

from retailflow.preview import PreviewServer


def test_page_and_no_frames_without_viewers():
    srv = PreviewServer("cam_x", port=0)
    try:
        assert srv.url.startswith("http://127.0.0.1:")
        html = urllib.request.urlopen(srv.url, timeout=5).read().decode()
        assert "cam_x" in html and 'src="/stream"' in html
        assert not srv.wants_frame()             # nobody watching: the pipeline draws nothing
    finally:
        srv.close()


def test_stream_sends_jpeg_frames_scaled_down():
    srv = PreviewServer("cam_x", port=0, max_fps=50, width=320)
    stop = threading.Event()

    def feed():                                   # stands in for the pipeline loop
        while not stop.is_set():
            if srv.wants_frame():
                srv.publish(np.full((540, 960, 3), 128, np.uint8))
            time.sleep(0.005)

    t = threading.Thread(target=feed, daemon=True)
    t.start()
    try:
        r = urllib.request.urlopen(srv.url + "stream", timeout=5)
        assert r.headers["Content-Type"].startswith("multipart/x-mixed-replace")
        head = b""
        while b"\r\n\r\n" not in head:
            head += r.read(1)
        length = int(head.split(b"Content-Length: ")[1].split(b"\r\n")[0])
        jpg = r.read(length)
        assert jpg[:2] == b"\xff\xd8"             # a JPEG
        import cv2
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        assert img.shape[1] == 320 and img.shape[0] == 180
        assert srv.viewers == 1
        r.close()
    finally:
        stop.set()
        srv.close()


def test_preview_rate_is_limited():
    srv = PreviewServer("cam_x", port=0, max_fps=10)
    try:
        srv._viewers = 1                          # pretend someone is watching
        assert srv.wants_frame()
        srv.publish(np.zeros((4, 4, 3), np.uint8))
        assert not srv.wants_frame()              # next frame only after 1/10 s
        time.sleep(0.11)
        assert srv.wants_frame()
    finally:
        srv._viewers = 0
        srv.close()


def test_stop_needs_post_with_header():
    import urllib.error
    srv = PreviewServer("cam_x", port=0)
    try:
        def post(headers):
            req = urllib.request.Request(srv.url + "stop", data=b"", method="POST", headers=headers)
            try:
                return urllib.request.urlopen(req, timeout=5).status
            except urllib.error.HTTPError as e:
                return e.code
        assert post({}) == 403 and not srv.stop_requested           # e.g. a form from another site
        assert post({"X-RetailFlow-Stop": "1"}) == 200 and srv.stop_requested
        html = urllib.request.urlopen(srv.url, timeout=5).read().decode()
        assert "Stop run" in html
    finally:
        srv.close()
