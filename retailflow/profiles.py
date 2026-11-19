"""Detector/tracker profiles: one setting for normal scenes, one for crowds.

Why two profiles (docs/04_COUNTING_RELIABILITY.md section 5.5): the grid chose
YOLO26m at 960 px on dev data. In packed crowds (MOT20) that setting detects
only a fraction of the people, and running the detector on the full-resolution
frame recovers many of them, at roughly 3-4x the cost. Paying that cost on
every camera would waste GPU on quiet shop entrances, so the input size is a
per-camera choice, stored in the scene file:

    "profile": "crowd"      # in scenes/<camera>.json; default "standard"

The profile is decided when the camera is set up, from how busy the scene
usually is (mean people per frame), never from counting results.

Phase 6 changes (docs/08_STREAMING_PERFORMANCE.md), both checked on dev first:
- tracker `botsort_nogmc`: BoT-SORT without camera-motion compensation. All
  cameras are static and GMC cost up to 31 ms per 1080p frame. Dev errors 32 -> 30.
- backend: the TensorRT FP16 engine (models/yolo26m_<imgsz>.engine) is used when
  it has been built on this machine (tools/export_models.py), else the .pt file.
  Same dev errors (32 vs 32 with GMC), about 2x faster.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Same threshold as the "busy" density group used in evaluation (fixed in Phase 2).
CROWD_MIN_PEOPLE = 15.0
MODELS_DIR = Path("models")


@dataclass(frozen=True)
class Profile:
    model: str
    imgsz: int
    tracker: str
    conf: float = 0.1

    def run_name(self, tracker: str | None = None, suffix: str = "") -> str:
        """Benchmark folder name, e.g. 'm960_botsort_nogmc' or 'm960_botsort_nogmc_trt'."""
        size = self.model.removeprefix("yolo26").removesuffix(".pt")
        return f"{size}{self.imgsz}_{tracker or self.tracker}{suffix}"

    @property
    def engine(self) -> Path:
        """Where tools/export_models.py writes this profile's TensorRT engine."""
        return MODELS_DIR / f"{Path(self.model).stem}_{self.imgsz}.engine"

    def weights(self, backend: str = "auto") -> str:
        """Model file for a backend: 'pt', 'trt', or 'auto' (engine if built, else .pt)."""
        if backend == "pt":
            return self.model
        if backend == "trt" or (backend == "auto" and self.engine.exists()):
            if not self.engine.exists():
                raise FileNotFoundError(f"{self.engine} missing: run python tools/export_models.py")
            return str(self.engine)
        return self.model


PROFILES: dict[str, Profile] = {
    "standard": Profile("yolo26m.pt", 960, "botsort_nogmc"),   # grid winner (dev) + Phase 6 tracker
    "crowd": Profile("yolo26m.pt", 1920, "botsort_nogmc"),     # full-resolution input
}


def suggest_profile(mean_people_per_frame: float) -> str:
    """Profile to put in a new scene file, given how busy the camera usually is."""
    return "crowd" if mean_people_per_frame >= CROWD_MIN_PEOPLE else "standard"


def get_profile(name: str) -> Profile:
    if name not in PROFILES:
        raise ValueError(f"unknown profile '{name}', expected one of {sorted(PROFILES)}")
    return PROFILES[name]
