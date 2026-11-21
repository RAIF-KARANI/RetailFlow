"""Live preview of the RUNNING pipeline in a browser (`run.py --preview`).

The pipeline hands its annotated frame (boxes, feet points, counting line, zones,
heatmap overlay, blurred heads if --blur-heads) to this server; the browser shows
it as an MJPEG stream at http://127.0.0.1:<port>/. No second model runs.

Cost control:
  - frames are only drawn and handed over while at least one viewer is connected;
  - at most `max_fps` frames per second, scaled down to `width` pixels;
  - JPEG encoding happens in the server's own thread, not in the pipeline loop.
The server listens on 127.0.0.1 only: the preview shows people, so it is not
exposed on the network.

Stop button: POST /stop (with the X-RetailFlow-Stop header, which a page from another
site cannot send without a CORS preflight this server never answers) asks the pipeline
to end the run cleanly: outputs and the API summary are written as for a normal end.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RetailFlow live: {cam}</title>
<style>
 :root {{ color-scheme: light dark; }}
 body {{ margin:0; font-family:system-ui,sans-serif; background:#111; color:#eee; }}
 header {{ padding:10px 16px; font-size:14px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }}
 button {{ margin-left:auto; font:inherit; padding:6px 14px; border-radius:6px; border:1px solid #a33;
          background:#3a1414; color:#fdd; cursor:pointer; }}
 button:disabled {{ opacity:.6; cursor:default; }}
 #msg {{ color:#fc6; }}
 select {{ font:inherit; padding:5px 8px; border-radius:6px; background:#222; color:#eee; border:1px solid #555; }}
 .dot {{ width:9px; height:9px; border-radius:50%; background:#e33; }}
 main {{ padding:0 16px 16px; }}
 img {{ width:100%; max-width:1280px; height:auto; display:block; background:#000; }}
 small {{ color:#aaa; }}
</style></head><body>
<header><span class="dot" id="dot"></span><b>{cam}</b><small>live pipeline output, {fps} fps preview, video time on the frame</small>
{picker}<span id="msg"></span><button id="stop" title="End this run now; results and the dashboard summary are saved">&#9632; Stop run</button></header>
<main><img id="v" src="/stream" alt="annotated live camera (waiting for the pipeline)"></main>
<script>
 const v = document.getElementById("v"), b = document.getElementById("stop"), m = document.getElementById("msg");
 let stopped = false;
 // Reconnect when the stream ends or is not up yet (model still loading), unless stopped.
 v.onerror = () => {{ if (!stopped) setTimeout(() => {{ v.src = "/stream?" + Date.now(); }}, 1000); }};
 const f = document.getElementById("focus");
 if (f) f.onchange = () => fetch("/focus?v=" + encodeURIComponent(f.value));
 b.onclick = async () => {{
   if (!confirm("Stop this run? Counts so far are kept and the session is closed on the dashboard.")) return;
   b.disabled = true; m.textContent = "Stopping...";
   try {{
     const r = await fetch("/stop", {{ method: "POST", headers: {{ "X-RetailFlow-Stop": "1" }} }});
     if (!r.ok) throw new Error(r.status);
     stopped = true; m.textContent = "Run stopped. Results are being saved; you can close this page.";
     document.getElementById("dot").style.background = "#777";
   }} catch (e) {{ b.disabled = false; m.textContent = "Could not stop (" + e + ")"; }}
 }};
</script>
</body></html>"""


class PreviewServer:
    """MJPEG server fed with the pipeline's newest annotated frame."""

    def __init__(self, camera_id: str, port: int = 8090, max_fps: float = 10.0,
                 width: int = 960, quality: int = 75, host: str = "127.0.0.1",
                 views: list[str] | None = None):
        """`views`: optional choices for the page's picker (e.g. "all" + camera names, run_store.py);
        the choice is in `self.focus` for the pipeline to draw."""
        self.camera_id = camera_id
        self.max_fps = max_fps
        self.width = width
        self.quality = quality
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self.published = 0              # frames drawn for viewers (for the run summary)
        self._viewers = 0
        self._last_pub = 0.0
        self._closed = False
        self.stop_requested = False        # set by the page's Stop button (POST /stop)
        self.views = list(views or [])
        self.focus = self.views[0] if self.views else None
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):          # keep the pipeline's console clean
                pass

            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/focus":
                    v = self.path.split("v=", 1)[-1] if "v=" in self.path else ""
                    from urllib.parse import unquote
                    v = unquote(v)
                    if v in server.views:
                        server.focus = v
                        server._last_pub = 0.0          # next frame right away
                    self.send_response(204)
                    self.end_headers()
                    return
                if path in ("/", "/index.html"):
                    picker = ""
                    if server.views:
                        opts = "".join(f'<option value="{v}"{" selected" if v == server.focus else ""}>'
                                       f'{"All cameras" if v == "all" else v.replace("camera_", "Camera ")}</option>'
                                       for v in server.views)
                        picker = f'<select id="focus" aria-label="Camera">{opts}</select>'
                    body = PAGE.format(cam=server.camera_id, fps=int(server.max_fps), picker=picker).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif path == "/stream":
                    server._stream(self)
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path.split("?")[0] != "/stop" or self.headers.get("X-RetailFlow-Stop") != "1":
                    self.send_error(403 if self.path.startswith("/stop") else 404)
                    return
                server.stop_requested = True
                body = b'{"stopping": true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        self.url = f"http://{host}:{self.port}/"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    # ---- pipeline side -------------------------------------------------
    def wants_frame(self) -> bool:
        """True when a viewer is connected and the next preview frame is due."""
        return self._viewers > 0 and time.perf_counter() - self._last_pub >= 1.0 / self.max_fps

    def publish(self, image: np.ndarray) -> None:
        """Hand over an annotated frame (not copied: the caller must not draw on it afterwards)."""
        self._last_pub = time.perf_counter()
        self.published += 1
        with self._cond:
            self._frame = image
            self._seq += 1
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._httpd.shutdown()
        self._httpd.server_close()

    # ---- viewer side (server threads) ----------------------------------
    def _encode(self, img: np.ndarray) -> bytes:
        h, w = img.shape[:2]
        if w > self.width:
            img = cv2.resize(img, (self.width, round(h * self.width / w)), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return jpg.tobytes() if ok else b""

    def _stream(self, handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        with self._cond:
            seen = self._seq                        # wait for the next frame drawn for us
            self._viewers += 1
        try:
            while True:
                with self._cond:
                    self._cond.wait_for(lambda: self._seq != seen or self._closed, timeout=5)
                    if self._closed:
                        return
                    if self._seq == seen or self._frame is None:
                        continue
                    seen, img = self._seq, self._frame
                jpg = self._encode(img)
                handler.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                    + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass                                    # viewer closed the tab
        finally:
            with self._cond:
                self._viewers -= 1

    @property
    def viewers(self) -> int:
        return self._viewers
