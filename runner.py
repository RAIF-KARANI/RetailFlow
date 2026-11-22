"""Host runner for the dashboard's "Add camera" flow: upload -> set up -> live.

    python runner.py            # http://127.0.0.1:8095 (the dashboard calls it)

The pipeline needs the GPU, so it runs on the host, not in Docker. This small server lets
the dashboard do on the host what you would otherwise type in PowerShell:

  1. upload a video          -> data/uploads/<camera>.<ext>
  2. show one frame of it    -> the dashboard's setup page draws the line, IN side and zones
  3. save the scene          -> scenes/<camera>.json (checked by retailflow.config)
  4. start a live run        -> the video is looped as an RTSP camera (tools/simulate_rtsp.py,
                                as in demo.ps1) and run.py analyses it with --api --preview
  5. stop the run            -> the preview's Stop (outputs and summary are saved as normal)

Standard library only. Listens on 127.0.0.1 and answers only the dashboard's origins
(RUNNER_ORIGINS), because frames show people. No new data leaves the machine.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from retailflow.config import LineConfig, Scene, scene_from_dict, scene_to_dict  # noqa: E402

PORT = int(os.environ.get("RUNNER_PORT", "8095"))
API = os.environ.get("RUNNER_API", "http://127.0.0.1:8000")
ORIGINS = set(os.environ.get(
    "RUNNER_ORIGINS", "http://localhost:3001,http://127.0.0.1:3001,http://localhost:3000").split(","))
UPLOADS = ROOT / "data" / "uploads"
SCENES = ROOT / "scenes"
OUT = ROOT / "outputs" / "web"
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v"}
MAX_UPLOAD = 8 * 1024**3                       # 8 GB
CAMERA_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$")
PREVIEW_PORTS = range(8090, 8110)
RTSP = "rtsp://127.0.0.1:8554"


class HttpError(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


def camera_id(raw: str) -> str:
    cam = unquote(raw)
    if not CAMERA_RE.match(cam):
        raise HttpError(400, "camera name: letters, digits, _ or -, up to 48 characters")
    return cam


def video_for(cam: str) -> Path:
    for p in sorted(UPLOADS.glob(f"{cam}.*")):
        if p.suffix.lower() in VIDEO_EXT:
            return p
    raise HttpError(404, f"no uploaded video for {cam}")


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        # Two frames: OpenCV also "opens" a single image renamed to .mp4.
        if not (cap.read()[0] and cap.read()[0]):
            raise HttpError(400, "this file is not a video OpenCV can read")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        n = n if 1 < n < 10**9 else None          # MPEG files often report nonsense; None = unknown
        return {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": round(fps, 3), "frames": n, "seconds": round(n / fps, 1) if n and fps else None}
    finally:
        cap.release()


def frame_jpeg(path: Path, index: int) -> bytes:
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
        ok, img = cap.read()
        if not ok:
            raise HttpError(400, f"no frame {index} in this video")
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
    finally:
        cap.release()


def default_scene(cam: str, w: int, h: int) -> dict:
    """Same start as tools/setup_scene.py: a horizontal line at 2/3 height, IN = downwards."""
    y = round(h * 2 / 3)
    return scene_to_dict(Scene(cam, (w, h), LineConfig((w * 0.2, y), (w * 0.8, y), (w * 0.5, h * 0.9),
                                                       round(h / 60))))


def free_port() -> int:
    busy = {r.preview_port for r in RUNS.values() if r.alive()}
    for p in PREVIEW_PORTS:
        if p in busy:
            continue
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise HttpError(503, "no free preview port (8090-8109): stop a run first")


def kill_tree(p: subprocess.Popen | None) -> None:
    """End a process and its children. On Windows terminate() would leave simulate_rtsp.py's FFmpeg running."""
    if p is None or p.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
    else:
        p.terminate()


class Run:
    """One live run: the looped RTSP camera plus run.py on it."""

    def __init__(self, cam: str, video: Path, minutes: float):
        self.cam, self.video, self.minutes = cam, video, minutes
        self.preview_port = free_port()
        self.state = "starting"                   # starting -> running -> stopping -> finished | failed
        self.error: str | None = None
        self.sim: subprocess.Popen | None = None
        self.proc: subprocess.Popen | None = None
        self.started = time.time()
        self.out = OUT / cam
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = self.out / "run.log"
        threading.Thread(target=self._go, daemon=True).start()

    @property
    def preview_url(self) -> str:
        return f"http://127.0.0.1:{self.preview_port}/"

    def alive(self) -> bool:
        return self.state in ("starting", "running", "stopping")

    def _go(self) -> None:
        try:
            r = subprocess.run(["docker", "compose", "--profile", "streaming", "up", "-d", "mediamtx"],
                               cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError("could not start the RTSP server (is Docker Desktop running?): "
                                   + (r.stderr.strip().splitlines() or [""])[-1])
            if self.state == "stopping":
                return self._end("finished")
            py = sys.executable
            with open(self.log, "w", encoding="utf-8") as log:
                self.sim = subprocess.Popen([py, str(ROOT / "tools" / "simulate_rtsp.py"), f"{self.cam}={self.video}"],
                                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                time.sleep(4)                     # as demo.ps1: let FFmpeg publish the stream
                if self.sim.poll() is not None:
                    raise RuntimeError("the simulated camera stopped (is FFmpeg installed?)")
                cmd = [py, str(ROOT / "run.py"), "--source", f"{RTSP}/{self.cam}",
                       "--scene", str(SCENES / f"{self.cam}.json"), "--out", str(self.out),
                       "--api", API, "--preview", str(self.preview_port), "--blur-heads"]
                if self.minutes > 0:
                    cmd += ["--duration", str(round(self.minutes * 60))]
                self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                if self.state == "starting":
                    self.state = "running"
                code = self.proc.wait()
            self._end("finished" if code == 0 else "failed",
                      None if code == 0 else f"run.py exited with code {code}: see {self.log}")
        except Exception as e:
            self._end("failed", str(e))

    def _end(self, state: str, error: str | None = None) -> None:
        kill_tree(self.sim)
        self.state, self.error = state, error

    def stop(self) -> None:
        if not self.alive():
            return
        self.state = "stopping"
        try:                                       # clean stop: run.py saves outputs and the summary
            req = urllib.request.Request(self.preview_url + "stop", method="POST",
                                         headers={"X-RetailFlow-Stop": "1"}, data=b"")
            urllib.request.urlopen(req, timeout=3).read()
        except Exception:
            pass

        def force():
            time.sleep(30)                         # still going (e.g. still loading the model)
            kill_tree(self.proc)
            kill_tree(self.sim)
        threading.Thread(target=force, daemon=True).start()

    def status(self) -> dict:
        tail = ""
        if self.log.exists():
            lines = self.log.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(l for l in lines[-8:] if "[TRT]" not in l)
        return {"camera_id": self.cam, "state": self.state, "error": self.error,
                "preview_url": self.preview_url if self.alive() else None,
                "minutes": self.minutes, "started": self.started, "log_tail": tail}


RUNS: dict[str, Run] = {}
LOCK = threading.Lock()


def camera_list() -> list[dict]:
    cams = {}
    for p in sorted(UPLOADS.glob("*")):
        if p.suffix.lower() in VIDEO_EXT:
            cams[p.stem] = {"camera_id": p.stem, "video": p.name, "has_scene": (SCENES / f"{p.stem}.json").exists()}
    for cam, r in RUNS.items():
        cams.setdefault(cam, {"camera_id": cam, "video": r.video.name, "has_scene": True})["run"] = r.status()
    return list(cams.values())


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    # --- helpers ---------------------------------------------------------------------------
    def _cors(self) -> None:
        origin = self.headers.get("Origin")
        if origin in ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _body_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 1_000_000:
            raise HttpError(413, "request too large")
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            raise HttpError(400, "invalid JSON")

    def _check_origin(self) -> None:
        # Writes only from the dashboard (or tools without an Origin, e.g. curl on this machine).
        origin = self.headers.get("Origin")
        if origin is not None and origin not in ORIGINS:
            raise HttpError(403, "origin not allowed")

    def _route(self, method: str) -> None:
        url = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        parts = [p for p in url.path.split("/") if p]
        try:
            if method != "GET":
                self._check_origin()
            if parts == ["api", "health"]:
                return self._json({"ok": True})
            if parts == ["api", "cameras"] and method == "GET":
                with LOCK:
                    return self._json(camera_list())
            if len(parts) == 4 and parts[:2] == ["api", "cameras"]:
                cam, what = camera_id(parts[2]), parts[3]
                handler = getattr(self, f"_{method.lower()}_{what}", None)
                if handler:
                    return handler(cam, q)
            raise HttpError(404, "not found")
        except HttpError as e:
            self._json({"detail": str(e)}, e.code)
        except (ConnectionError, TimeoutError):
            pass
        except Exception as e:                    # keep the server up; report the error
            self._json({"detail": f"{type(e).__name__}: {e}"}, 500)

    def do_OPTIONS(self):
        self.send_response(204 if self.headers.get("Origin") in ORIGINS else 403)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def do_PUT(self):
        self._route("PUT")

    # --- endpoints: /api/cameras/<cam>/<what> ---------------------------------------------------
    def _post_video(self, cam: str, q: dict) -> None:
        """Body = the raw video file. ?filename=x.mp4 gives the extension; ?replace=1 overwrites."""
        ext = Path(q.get("filename", "video.mp4")).suffix.lower()
        if ext not in VIDEO_EXT:
            raise HttpError(400, f"video type {ext or '?'} not supported: use {', '.join(sorted(VIDEO_EXT))}")
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_UPLOAD:
            raise HttpError(413, "the video must be between 1 byte and 8 GB")
        with LOCK:
            if cam in RUNS and RUNS[cam].alive():
                raise HttpError(409, f"{cam} is running: stop it first")
            old = [p for p in UPLOADS.glob(f"{cam}.*") if p.suffix.lower() in VIDEO_EXT]
            if old and q.get("replace") != "1":
                raise HttpError(409, f"a video named {cam} is already uploaded")
        UPLOADS.mkdir(parents=True, exist_ok=True)
        tmp = UPLOADS / f".{cam}{ext}.part"
        left = n
        with open(tmp, "wb") as f:
            while left:
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        if left:
            tmp.unlink(missing_ok=True)
            raise HttpError(400, "upload interrupted")
        try:
            info = video_info(tmp)
        except HttpError:
            tmp.unlink(missing_ok=True)
            raise
        for p in old:
            p.unlink()
        dest = UPLOADS / f"{cam}{ext}"
        tmp.replace(dest)
        self._json({"camera_id": cam, "video": dest.name, **info})

    def _get_video(self, cam: str, q: dict) -> None:
        p = video_for(cam)
        self._json({"camera_id": cam, "video": p.name, **video_info(p)})

    def _get_frame(self, cam: str, q: dict) -> None:
        self._send(200, frame_jpeg(video_for(cam), int(q.get("n", "50"))), "image/jpeg")

    def _get_scene(self, cam: str, q: dict) -> None:
        path = SCENES / f"{cam}.json"
        if path.exists():
            return self._json({"exists": True, "scene": json.loads(path.read_text(encoding="utf-8"))})
        info = video_info(video_for(cam))
        self._json({"exists": False, "scene": default_scene(cam, info["width"], info["height"])})

    def _put_scene(self, cam: str, q: dict) -> None:
        d = self._body_json()
        d["camera_id"] = cam
        try:
            scene = scene_from_dict(d)             # same checks as every scene file
        except (KeyError, TypeError, ValueError) as e:
            raise HttpError(400, f"scene not valid: {e}")
        out = scene_to_dict(scene)
        SCENES.mkdir(exist_ok=True)
        (SCENES / f"{cam}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        self._json({"saved": f"scenes/{cam}.json", "scene": out})

    def _post_start(self, cam: str, q: dict) -> None:
        minutes = float(self._body_json().get("minutes", 5))
        if not 0 <= minutes <= 24 * 60:
            raise HttpError(400, "minutes: 0 (until stopped) to 1440")
        if not (SCENES / f"{cam}.json").exists():
            raise HttpError(400, "save the scene first")
        video = video_for(cam)
        with LOCK:
            if cam in RUNS and RUNS[cam].alive():
                raise HttpError(409, f"{cam} is already running")
            RUNS[cam] = Run(cam, video, minutes)
            self._json(RUNS[cam].status())

    def _post_stop(self, cam: str, q: dict) -> None:
        with LOCK:
            r = RUNS.get(cam)
        if not r:
            raise HttpError(404, f"{cam} was not started from the dashboard")
        r.stop()
        self._json(r.status())

    def _get_run(self, cam: str, q: dict) -> None:
        with LOCK:
            r = RUNS.get(cam)
        self._json(r.status() if r else {"camera_id": cam, "state": "none"})


def main() -> None:
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    httpd.daemon_threads = True
    print(f"RetailFlow runner on http://127.0.0.1:{PORT}  (dashboard: http://localhost:3001, Add camera)",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with LOCK:
            for r in RUNS.values():
                r.stop()
        print("Runner stopped. Live runs were asked to stop.", flush=True)


if __name__ == "__main__":
    main()
