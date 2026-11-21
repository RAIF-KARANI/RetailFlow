"""Frame sources: a video file, a folder of images, or a live RTSP stream.

All sources produce the same thing: (frame_index, timestamp_seconds, image).

Timestamps:
  - files / image folders: VIDEO TIMELINE, frame_index / fps. Dataset videos
    have no real clock time, so we never pretend they do.
  - RTSP / live: seconds since the stream was opened (wall clock).

Live streams are read by a background thread that decodes EVERY frame and keeps
only the newest one (`LatestFrameReader`). When processing is slower than the
camera for a moment, the pipeline skips whole frames instead of the RTSP server
dropping packets mid-frame (which corrupts H.264 frames and loses time silently).
Frame.index stays the frame's true position in the stream, so skipped frames show
up as gaps in the index and video time stays correct.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
LIVE_TIMEOUT_MS = 15_000     # no frame for this long = the stream has ended


@dataclass
class Frame:
    index: int
    timestamp: float  # seconds (video timeline for files, wall clock for live)
    image: np.ndarray


class LatestFrameReader:
    """Decode a live capture in a thread; hand out only the newest frame.

    `get()` blocks until a frame newer than the last one handed out exists and
    returns (index, image), or None when the stream has ended. Frames that were
    decoded but never handed out are counted in `skipped`.
    """

    def __init__(self, cap, max_frames: int | None = None, pace_fps: float | None = None,
                 keep_every: int = 1):
        """`pace_fps`: release frames no faster than this (a file replayed as a live camera).
        `keep_every`: only turn every n-th frame into an image (the others are still decoded,
        to keep time, but never copied into Python): for many cameras analysed at a low rate."""
        self._cap = cap
        self.max_frames = max_frames
        self.pace_fps = pace_fps
        self.keep_every = max(1, int(keep_every))
        self._cond = threading.Condition()
        self._latest: tuple[int, np.ndarray] | None = None
        self._last_out = -1
        self._done = False
        self.received = 0
        self.skipped = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        i = 0
        t0 = time.perf_counter()
        while not self._done and (self.max_frames is None or i < self.max_frames):
            if i % self.keep_every:
                ok = self._cap.grab()         # decode only: no image array, almost no GIL use
                if not ok:
                    break
                if self.pace_fps:
                    wait = t0 + i / self.pace_fps - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                i += 1
                continue
            ok, img = self._cap.read()        # OpenCV releases the GIL while decoding
            if not ok:
                break
            if self.pace_fps:                 # a live camera delivers frame i at i / fps
                wait = t0 + i / self.pace_fps - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
            with self._cond:
                self._latest = (i, img)
                self.received = i + 1
                self._cond.notify()
            i += 1
        with self._cond:
            self._done = True
            self._cond.notify()

    def _has_new(self) -> bool:
        return self._latest is not None and self._latest[0] > self._last_out

    def get(self) -> tuple[int, np.ndarray] | None:
        with self._cond:
            while not self._has_new() and not self._done:
                self._cond.wait()
            if not self._has_new():
                return None
            i, img = self._latest
            self.skipped += i - self._last_out - 1
            self._last_out = i
            return i, img

    def newest(self) -> tuple[int, np.ndarray] | None:
        """Non-blocking: the newest frame if it is newer than the last one handed out, else None."""
        with self._cond:
            if not self._has_new():
                return None
            i, img = self._latest
            self.skipped += i - self._last_out - 1
            self._last_out = i
            return i, img

    @property
    def finished(self) -> bool:
        """The stream has ended and its last frame has been handed out."""
        with self._cond:
            return self._done and not self._has_new()

    def stop(self) -> None:
        with self._cond:
            self._done = True
        self._thread.join(timeout=5)


class FrameSource:
    """Iterate frames from a path or URL. Use as: `for f in FrameSource(...)`."""

    def __init__(self, source: str | Path, fps: float | None = None, max_frames: int | None = None):
        self.source = str(source)
        self.max_frames = max_frames
        self.is_live = self.source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        self.is_folder = Path(self.source).is_dir()
        self._fps_override = fps
        self.reader: LatestFrameReader | None = None
        self._cap = None

        if self.is_folder:
            self._images = sorted(p for p in Path(self.source).iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if not self._images:
                raise FileNotFoundError(f"no images in {self.source}")
            first = cv2.imread(str(self._images[0]))
            self.height, self.width = first.shape[:2]
            self.fps = fps or 30.0
            self.frame_count = len(self._images)
        else:
            if not self.is_live and not Path(self.source).exists():
                raise FileNotFoundError(self.source)
            if self.source.lower().startswith("rtsp://"):
                # RTSP over TCP: no lost packets (= no smeared frames) on a busy machine.
                os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            # Live: give up after LIVE_TIMEOUT_MS without a frame, so a dead camera or
            # server ends the run instead of blocking forever (seen when Docker stopped).
            params = ([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, LIVE_TIMEOUT_MS,
                       cv2.CAP_PROP_READ_TIMEOUT_MSEC, LIVE_TIMEOUT_MS] if self.is_live else [])
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, params)
            if not self._cap.isOpened():
                raise RuntimeError(f"cannot open {self.source}")
            self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = fps or self._cap.get(cv2.CAP_PROP_FPS) or 25.0
            n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_count = n if n > 0 and not self.is_live else None

    @property
    def frames_skipped(self) -> int:
        """Live only: frames decoded but skipped because processing was behind."""
        return self.reader.skipped if self.reader else 0

    def _iter_live(self) -> Iterator[Frame]:
        self.reader = LatestFrameReader(self._cap, self.max_frames)
        start = time.perf_counter()
        try:
            while (got := self.reader.get()) is not None:
                i, img = got
                yield Frame(i, time.perf_counter() - start, img)
        finally:
            self.reader.stop()
            self.close()

    def __iter__(self) -> Iterator[Frame]:
        if self.is_live:
            yield from self._iter_live()
            return
        start = time.perf_counter()
        i = 0
        while self.max_frames is None or i < self.max_frames:
            if self.is_folder:
                if i >= len(self._images):
                    break
                img = cv2.imread(str(self._images[i]))
            else:
                ok, img = self._cap.read()
                if not ok:
                    break
            ts = (time.perf_counter() - start) if self.is_live else i / self.fps
            yield Frame(i, ts, img)
            i += 1
        self.close()

    def close(self) -> None:
        if not self.is_folder and self._cap is not None:
            self._cap.release()
