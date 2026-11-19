"""Frame sources: a video file, a folder of images, or a live RTSP stream.

All sources produce the same thing: (frame_index, timestamp_seconds, image).

Timestamps:
  - files / image folders: VIDEO TIMELINE, frame_index / fps. Dataset videos
    have no real clock time, so we never pretend they do.
  - RTSP / live: seconds since the stream was opened (wall clock).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


@dataclass
class Frame:
    index: int
    timestamp: float  # seconds (video timeline for files, wall clock for live)
    image: np.ndarray


class FrameSource:
    """Iterate frames from a path or URL. Use as: `for f in FrameSource(...)`."""

    def __init__(self, source: str | Path, fps: float | None = None, max_frames: int | None = None):
        self.source = str(source)
        self.max_frames = max_frames
        self.is_live = self.source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        self.is_folder = Path(self.source).is_dir()
        self._fps_override = fps

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
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if not self._cap.isOpened():
                raise RuntimeError(f"cannot open {self.source}")
            self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = fps or self._cap.get(cv2.CAP_PROP_FPS) or 25.0
            n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_count = n if n > 0 and not self.is_live else None

    def __iter__(self) -> Iterator[Frame]:
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
