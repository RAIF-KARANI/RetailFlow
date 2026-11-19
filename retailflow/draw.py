"""Drawing helpers for annotated output video (no analytics logic here)."""

from __future__ import annotations

import cv2
import numpy as np

from retailflow.config import LineConfig
from retailflow.detector import TrackedPerson

GREEN, RED, YELLOW, WHITE, BLACK = (0, 200, 0), (0, 0, 230), (0, 220, 255), (255, 255, 255), (0, 0, 0)
CYAN, ORANGE = (230, 200, 0), (0, 140, 255)


def _ipt(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def scale_for(image: np.ndarray) -> float:
    """Text/line scale so overlays look the same at 384 px and 1920 px wide."""
    return max(image.shape[1] / 1280, 0.35)


def draw_line(image: np.ndarray, line: LineConfig, band: float) -> None:
    """Counting line, its hysteresis band (thin parallel lines) and an IN arrow."""
    s = scale_for(image)
    a, b = np.array(line.p1), np.array(line.p2)
    d = b - a
    normal = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    # Orient the normal towards the IN side.
    if np.dot(np.array(line.in_point) - a, normal) < 0:
        normal = -normal
    if band > 0:
        for off in (band, -band):
            cv2.line(image, _ipt(a + normal * off), _ipt(b + normal * off), YELLOW, 1, cv2.LINE_AA)
    cv2.line(image, _ipt(a), _ipt(b), YELLOW, max(1, int(3 * s)), cv2.LINE_AA)
    mid = (a + b) / 2
    cv2.arrowedLine(image, _ipt(mid), _ipt(mid + normal * 40 * s), GREEN, max(1, int(3 * s)), tipLength=0.4)
    cv2.putText(image, "IN", _ipt(mid + normal * 50 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, GREEN, max(1, int(2 * s)))


def draw_zones(image: np.ndarray, zones, occupancy: dict[str, int] | None = None) -> None:
    """Zone outlines with name and current occupancy."""
    s = scale_for(image)
    for z in zones:
        pts = np.array([_ipt(p) for p in z.polygon], np.int32)
        cv2.polylines(image, [pts], True, CYAN, max(1, int(2 * s)), cv2.LINE_AA)
        label = z.name if occupancy is None else f"{z.name}: {occupancy.get(z.name, 0)}"
        cv2.putText(image, label, _ipt((pts[:, 0].min() + 3, pts[:, 1].min() + 14 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, CYAN, max(1, int(s)))


def draw_people(image: np.ndarray, people: list[TrackedPerson], point_mode: str,
                staff: set[int] | None = None) -> None:
    """Boxes, IDs and the anchor point used for counting. Staff tracks are drawn orange."""
    s = scale_for(image)
    staff = staff or set()
    for p in people:
        col = ORANGE if p.track_id in staff else WHITE
        tag = f"{p.track_id} staff" if p.track_id in staff else str(p.track_id)
        cv2.rectangle(image, _ipt((p.x1, p.y1)), _ipt((p.x2, p.y2)), col, max(1, int(2 * s)))
        cv2.putText(image, tag, _ipt((p.x1, p.y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * s, col, max(1, int(2 * s)))
        anchor = p.feet if point_mode == "feet" else p.centre
        cv2.circle(image, _ipt(anchor), max(2, int(5 * s)), RED, -1)


def draw_hud(image: np.ndarray, lines: list[str]) -> None:
    """Text panel in the top-left corner (counts, FPS...)."""
    s = scale_for(image)
    h = int(28 * s) + 4
    w = int(max(len(t) for t in lines) * 13 * s) + 10
    cv2.rectangle(image, (0, 0), (w, h * len(lines) + 6), BLACK, -1)
    for i, text in enumerate(lines):
        cv2.putText(image, text, (5, h * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, WHITE, max(1, int(2 * s)))
