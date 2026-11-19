"""Person detection + tracking with Ultralytics YOLO26 and ByteTrack / BoT-SORT.

One `PersonTracker` per camera stream. Each call to `update(frame)` runs the
detector on the frame and associates detections with existing tracks, giving
every person a stable integer ID.

The model file decides the backend, so the same code runs every backend:
  yolo26n.pt      -> PyTorch
  yolo26n.onnx    -> ONNX Runtime            (Phase 6)
  yolo26n.engine  -> TensorRT                 (Phase 6)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

PERSON_CLASS = 0  # COCO class 0 = person
TRACKER_DIR = Path(__file__).resolve().parent.parent / "trackers"

# Short names -> tracker YAML. Built-ins ship with Ultralytics; custom ones live in trackers/.
TRACKERS = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
    "botsort_reid": str(TRACKER_DIR / "botsort_reid.yaml"),  # Phase 3
}


@dataclass
class TrackedPerson:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def feet(self) -> tuple[float, float]:
        """Bottom-centre of the box: where the person stands on the floor."""
        return ((self.x1 + self.x2) / 2, self.y2)

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class PersonTracker:
    """YOLO26 detector + multi-object tracker for one video stream."""

    def __init__(self, model: str = "yolo26n.pt", tracker: str = "bytetrack",
                 imgsz: int = 640, conf: float = 0.1, device: int | str = 0):
        # conf=0.1 is deliberately LOW: ByteTrack's key idea is to use low-score
        # detections (0.1-0.25) in a second association round to keep tracks
        # alive through partial occlusion. The tracker's own thresholds
        # (track_high_thresh, new_track_thresh) decide what starts a new track.
        self.model = YOLO(model, task="detect")
        self.tracker_cfg = TRACKERS.get(tracker, tracker)
        self.imgsz, self.conf, self.device = imgsz, conf, device

    def update(self, image: np.ndarray) -> list[TrackedPerson]:
        """Detect and track people in one frame. Call once per frame, in order."""
        results = self.model.track(
            image, persist=True,              # keep tracker state between calls
            tracker=self.tracker_cfg, classes=[PERSON_CLASS],
            imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:  # no confirmed tracks this frame
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        return [TrackedPerson(int(i), *map(float, b), float(c)) for i, b, c in zip(ids, xyxy, confs)]

    @property
    def last_speed_ms(self) -> dict:
        """Ultralytics' own pre/inference/post timing for the last frame (ms)."""
        pred = getattr(self.model, "predictor", None)
        res = getattr(pred, "results", None)
        return dict(res[0].speed) if res else {}
