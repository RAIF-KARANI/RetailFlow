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
"""

from __future__ import annotations

from dataclasses import dataclass

# Same threshold as the "busy" density group used in evaluation (fixed in Phase 2).
CROWD_MIN_PEOPLE = 15.0


@dataclass(frozen=True)
class Profile:
    model: str
    imgsz: int
    tracker: str
    conf: float = 0.1

    @property
    def run_name(self) -> str:
        """Benchmark folder name, e.g. 'm960_botsort' (same format as benchmark.py)."""
        size = self.model.removeprefix("yolo26").removesuffix(".pt")
        return f"{size}{self.imgsz}_{self.tracker}"


PROFILES: dict[str, Profile] = {
    "standard": Profile("yolo26m.pt", 960, "botsort"),   # grid winner (dev)
    "crowd": Profile("yolo26m.pt", 1920, "botsort"),     # full-resolution input
}


def suggest_profile(mean_people_per_frame: float) -> str:
    """Profile to put in a new scene file, given how busy the camera usually is."""
    return "crowd" if mean_people_per_frame >= CROWD_MIN_PEOPLE else "standard"


def get_profile(name: str) -> Profile:
    if name not in PROFILES:
        raise ValueError(f"unknown profile '{name}', expected one of {sorted(PROFILES)}")
    return PROFILES[name]
