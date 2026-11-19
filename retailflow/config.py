"""Per-camera scene configuration, stored as JSON in scenes/<camera_id>.json.

A scene describes everything that depends on WHERE the camera is mounted:

{
  "camera_id": "caviar_cor",
  "description": "CAVIAR corridor view, line across the corridor",
  "profile": "standard",                 # or "crowd" (retailflow/profiles.py)
  "frame_size": [384, 288],
  "line": {
    "p1": [45, 190], "p2": [265, 190],   # counting line ON THE FLOOR (pixels)
    "in_point": [150, 250],              # any point on the IN side of the line
    "band_px": 5                         # half-width of the hysteresis band
  },
  "zones": [{"name": "entrance", "polygon": [[x, y], ...]}],     # Phase 4
  "homography": {"image_points": [[x, y] x4], "floor_points": [[X, Y] x4]}  # Phase 4
}

Settings that are NOT camera-specific (model, tracker, min track age...) are
passed at run time instead, so the same scene can be benchmarked many ways.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from retailflow.profiles import PROFILES

Point = tuple[float, float]


@dataclass(frozen=True)
class LineConfig:
    """A counting line segment plus which side counts as IN."""

    p1: Point
    p2: Point
    in_point: Point       # a point on the IN side; defines the IN direction
    band_px: float = 0.0  # hysteresis half-width; 0 = simple line


@dataclass(frozen=True)
class Zone:
    """A named polygon on the floor (image coordinates)."""

    name: str
    polygon: tuple[Point, ...]


@dataclass(frozen=True)
class Homography:
    """Four image points and the matching floor-plan points (e.g. metres)."""

    image_points: tuple[Point, ...]
    floor_points: tuple[Point, ...]


@dataclass(frozen=True)
class Scene:
    camera_id: str
    frame_size: tuple[int, int]
    line: LineConfig
    zones: tuple[Zone, ...] = ()
    homography: Homography | None = None
    description: str = ""
    profile: str = "standard"  # detector profile, see retailflow/profiles.py
    extra: dict = field(default_factory=dict)  # free-form settings (staff rules etc.)


def _pt(p) -> Point:
    return (float(p[0]), float(p[1]))


def scene_from_dict(d: dict) -> Scene:
    """Build a Scene from parsed JSON, validating the essentials."""
    ln = d["line"]
    line = LineConfig(_pt(ln["p1"]), _pt(ln["p2"]), _pt(ln["in_point"]), float(ln.get("band_px", 0)))
    if line.p1 == line.p2:
        raise ValueError(f"{d['camera_id']}: line p1 and p2 are the same point")
    zones = tuple(Zone(z["name"], tuple(_pt(p) for p in z["polygon"])) for z in d.get("zones", []))
    for z in zones:
        if len(z.polygon) < 3:
            raise ValueError(f"zone '{z.name}' needs at least 3 points")
    homo = None
    if d.get("homography"):
        h = d["homography"]
        homo = Homography(tuple(_pt(p) for p in h["image_points"]), tuple(_pt(p) for p in h["floor_points"]))
        if len(homo.image_points) < 4 or len(homo.image_points) != len(homo.floor_points):
            raise ValueError("homography needs >= 4 matching image/floor points")
    profile = d.get("profile", "standard")
    if profile not in PROFILES:
        raise ValueError(f"{d['camera_id']}: unknown profile '{profile}'")
    return Scene(
        camera_id=d["camera_id"],
        frame_size=tuple(d["frame_size"]),
        line=line,
        zones=zones,
        homography=homo,
        description=d.get("description", ""),
        profile=profile,
        extra=d.get("extra", {}),
    )


def scene_to_dict(s: Scene) -> dict:
    """Inverse of scene_from_dict, used by tools/setup_scene.py when saving."""
    d = {
        "camera_id": s.camera_id,
        "description": s.description,
        "profile": s.profile,
        "frame_size": list(s.frame_size),
        "line": {"p1": list(s.line.p1), "p2": list(s.line.p2),
                 "in_point": list(s.line.in_point), "band_px": s.line.band_px},
        "zones": [{"name": z.name, "polygon": [list(p) for p in z.polygon]} for z in s.zones],
        "homography": None if s.homography is None else {
            "image_points": [list(p) for p in s.homography.image_points],
            "floor_points": [list(p) for p in s.homography.floor_points],
        },
    }
    if s.extra:
        d["extra"] = s.extra
    return d


def load_scene(path: str | Path) -> Scene:
    # utf-8-sig also accepts a byte-order mark, which Windows PowerShell 5.1
    # adds when writing "utf8" files and which plain json.loads rejects.
    return scene_from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def save_scene(scene: Scene, path: str | Path) -> None:
    Path(path).write_text(json.dumps(scene_to_dict(scene), indent=2) + "\n", encoding="utf-8")


def scene_path_for_sequence(seq_name: str, scenes_dir: str | Path = "scenes") -> Path:
    """Map a dataset sequence to its camera's scene file.

    All 26 CAVIAR corridor clips come from ONE physical camera, so they share
    scenes/caviar_cor.json (same for the front view). MOT sequences each have
    their own camera.
    """
    scenes_dir = Path(scenes_dir)
    if seq_name.endswith("cor"):
        return scenes_dir / "caviar_cor.json"
    if seq_name.endswith("front"):
        return scenes_dir / "caviar_front.json"
    # "MOT17-04-FRCNN" -> "MOT17-04"
    return scenes_dir / f"{'-'.join(seq_name.split('-')[:2])}.json"
