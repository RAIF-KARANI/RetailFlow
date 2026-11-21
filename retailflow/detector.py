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
    "botsort_nogmc": str(TRACKER_DIR / "botsort_nogmc.yaml"),  # Phase 6: static cameras, no GMC
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


class MultiCameraTracker:
    """One detector for many cameras, one tracker per camera (run_site.py).

    `PersonTracker` loads one model per camera, and 15 TensorRT contexts do not fit
    in 8 GB of GPU memory. Here ONE model serves every camera, and each camera has its
    own BoT-SORT/ByteTrack instance (same settings file as `PersonTracker`). With
    `batch` > 1 the cameras' frames go through the detector together, up to `batch` per
    call (needs a batch engine: tools/export_models.py --max-batch 16); with 1, one at a
    time. Track IDs are per camera, as before.
    """

    def __init__(self, model: str, tracker: str, imgsz: int, conf: float, device: int | str,
                 cameras: list[str], batch: int = 1):
        from ultralytics.trackers.track import TRACKER_MAP
        from ultralytics.utils import YAML, IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml

        self.model = YOLO(model, task="detect")
        cfg = IterableSimpleNamespace(**YAML.load(check_yaml(TRACKERS.get(tracker, tracker))))
        self.trackers = {c: TRACKER_MAP[cfg.tracker_type](args=cfg) for c in cameras}
        self.imgsz, self.conf, self.device, self.batch = imgsz, conf, device, max(1, batch)

    def update(self, frames: dict[str, np.ndarray]) -> dict[str, list[TrackedPerson]]:
        """Detect and track the given cameras' frames. Call with each camera's frames in order."""
        out = {}
        cams = list(frames)
        results = []
        for i in range(0, len(cams), self.batch):
            chunk = [frames[c] for c in cams[i:i + self.batch]]
            results += self.model.predict(chunk if len(chunk) > 1 else chunk[0], classes=[PERSON_CLASS],
                                          imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
        for cam, r in zip(cams, results):
            tracks = self.trackers[cam].update(r.boxes.cpu().numpy(), frames[cam])
            # rows: x1, y1, x2, y2, track id, score, class, detection index
            out[cam] = [TrackedPerson(int(t[4]), float(t[0]), float(t[1]), float(t[2]), float(t[3]), float(t[5]))
                        for t in tracks]
        return out
