"""Send pipeline events to the RetailFlow API (`run.py --api http://localhost:8000`).

Uses only the Python standard library (urllib), so the pipeline needs no extra packages.

Flow:
  1. PUT  /api/cameras/{id}          register the camera's scene (line, zones...)
  2. POST /api/sessions              start a session for this source
  3. POST /api/sessions/{id}/events  batches of events (every `batch` events or `flush_s` seconds)
  4. POST /api/sessions/{id}/finish  summary, heatmap grid and people-free background

If the API is unreachable, events stay in memory and are retried after
`retry_s` seconds (up to `max_buffer`, after which the oldest are dropped and
counted in the summary as `events_dropped`).
The local events.csv is always written as well, so no data depends on the network.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict

import cv2
import numpy as np

from retailflow.config import Scene, scene_to_dict
from retailflow.events import Event


class ApiError(RuntimeError):
    pass


def _request(method: str, url: str, body=None, timeout: float = 10.0) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise ApiError(f"{method} {url} -> {e.code}: {e.read().decode(errors='replace')[:300]}") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise ApiError(f"{method} {url} -> {e}") from e


class ApiEventSender:
    """An event sink (write / close / finish) that posts to the API."""

    def __init__(self, base_url: str, scene: Scene, source: str, video_fps: float | None,
                 run_info: dict | None = None, batch: int = 200, flush_s: float = 1.0,
                 max_buffer: int = 100_000, retry_s: float = 5.0):
        # "localhost" first tries IPv6 (::1); Docker publishes the API on IPv4 only,
        # so every request would wait ~2 s for the fallback (docs/TROUBLESHOOTING.md).
        self.base = base_url.rstrip("/").replace("//localhost:", "//127.0.0.1:")
        self.batch, self.flush_s, self.max_buffer, self.retry_s = batch, flush_s, max_buffer, retry_s
        self._retry_at = 0.0
        self.buffer: list[dict] = []
        self.sent = 0
        self.dropped = 0
        self._last_flush = time.monotonic()
        _request("PUT", f"{self.base}/api/cameras/{scene.camera_id}", scene_to_dict(scene))
        r = _request("POST", f"{self.base}/api/sessions",
                     {"camera_id": scene.camera_id, "source": source, "video_fps": video_fps,
                      "run_info": run_info or {}})
        self.session_id = r["session_id"]

    def write(self, event: Event) -> None:
        e = asdict(event)
        e.pop("camera_id")  # implied by the session
        self.buffer.append(e)
        now = time.monotonic()
        due = len(self.buffer) >= self.batch or now - self._last_flush >= self.flush_s
        if due and now >= self._retry_at:   # after a failure, wait retry_s before trying again
            self.flush()

    def flush(self) -> bool:
        """Try to send everything buffered. Returns False (and keeps the events) on failure."""
        self._last_flush = time.monotonic()
        while self.buffer:
            chunk = self.buffer[:1000]
            try:
                _request("POST", f"{self.base}/api/sessions/{self.session_id}/events", chunk)
            except ApiError as err:
                if len(self.buffer) > self.max_buffer:
                    drop = len(self.buffer) - self.max_buffer
                    del self.buffer[:drop]
                    self.dropped += drop
                print(f"[sender] {err}; keeping {len(self.buffer)} events, retry in {self.retry_s:g} s")
                self._retry_at = time.monotonic() + self.retry_s
                return False
            del self.buffer[:len(chunk)]
            self.sent += len(chunk)
        return True

    def close(self) -> None:
        self.flush()

    def finish(self, summary: dict, heat_grid: np.ndarray | None = None, cell: float | None = None,
               floor: bool = False, background: np.ndarray | None = None) -> None:
        """Send the run summary and the heatmap once the source has ended."""
        self.flush()
        body: dict = {"summary": {**summary, "events_sent": self.sent, "events_dropped": self.dropped}}
        if heat_grid is not None:
            body["heatmap"] = {"cell": cell, "floor": floor, "grid": np.round(heat_grid, 4).tolist()}
        if background is not None:
            ok, png = cv2.imencode(".png", background)
            if ok:
                body["background_png_b64"] = base64.b64encode(png.tobytes()).decode()
        _request("POST", f"{self.base}/api/sessions/{self.session_id}/finish", body, timeout=60)


class StoreApiClient:
    """Sends a live store run (run_store.py) to the API: config, state every few seconds, summary.

    The state holds only anonymous data: person numbers, floor positions, camera names, times.
    A failed post is skipped (the next state replaces it); the run itself never stops for the API.
    """

    def __init__(self, base_url: str, store: dict, run_info: dict | None = None, scene_dir=None):
        self.base = base_url.rstrip("/").replace("//localhost:", "//127.0.0.1:")
        self.store_id = store["store_id"]
        _request("PUT", f"{self.base}/api/stores/{self.store_id}", store)
        if scene_dir is not None:
            self.upload_floor(self.base, self.store_id, scene_dir)
        r = _request("POST", f"{self.base}/api/stores/{self.store_id}/runs", {"run_info": run_info or {}})
        self.run_id = r["run_id"]
        self.failed = 0

    @staticmethod
    def upload_floor(base_url: str, store_id: str, scene_dir) -> list[str]:
        """Send the store's top-down floor images (tools/build_floor_plan.py), if they exist."""
        from pathlib import Path
        base = base_url.rstrip("/").replace("//localhost:", "//127.0.0.1:")
        sent = []
        bp = Path(scene_dir) / "floor_blueprint.png"
        if bp.exists():
            _request("PUT", f"{base}/api/stores/{store_id}/floor/blueprint",
                     {"content_type": "image/png", "data_b64": base64.b64encode(bp.read_bytes()).decode()}, timeout=30)
            sent.append("blueprint")
        ph = Path(scene_dir) / "floor_photo.png"
        if ph.exists():
            ok, jpg = cv2.imencode(".jpg", cv2.imread(str(ph)), [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                _request("PUT", f"{base}/api/stores/{store_id}/floor/photo",
                         {"content_type": "image/jpeg", "data_b64": base64.b64encode(jpg.tobytes()).decode()},
                         timeout=30)
                sent.append("photo")
        return sent

    @classmethod
    def attach(cls, base_url: str, store_id: str, run_id: str) -> "StoreApiClient":
        """A client for a run that already exists (the linking process of run_store.py)."""
        c = cls.__new__(cls)
        c.base = base_url.rstrip("/").replace("//localhost:", "//127.0.0.1:")
        c.store_id, c.run_id, c.failed = store_id, run_id, 0
        return c

    def state(self, state: dict) -> None:
        try:
            _request("PUT", f"{self.base}/api/store-runs/{self.run_id}/state", state, timeout=10)
        except ApiError as e:
            self.failed += 1
            print(f"[store sender] {e}")

    def finish(self, state: dict, summary: dict) -> None:
        _request("POST", f"{self.base}/api/store-runs/{self.run_id}/finish",
                 {"state": state, "summary": summary}, timeout=60)
