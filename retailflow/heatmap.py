"""Occupancy heatmaps: where people spend time on the floor.

Each frame, every person adds `1 / fps` seconds at their **feet point** to a
grid. The finished grid is "person-seconds per cell": a busy aisle is hot
because many people pass, a display is hot because a few people stand there long.

Image grid vs floor grid
------------------------
By default the grid lies over the image (one cell = `cell_px` pixels). In an
angled camera this is distorted: far away, one pixel covers much more floor
than close to the camera, so far areas look colder than they are. If the scene
has a homography (4+ image points with known floor coordinates, set with
tools/setup_scene.py), points are first mapped to the floor plan and the grid
is in floor units (e.g. 0.25 m cells), which removes the distortion.
None of the public datasets used here ship floor measurements, so their scenes
use the image grid (docs/05_ZONES_DWELL_HEATMAP.md).

Privacy: the heatmap holds only aggregated time per cell. The background image
for rendering is the **median** of sampled frames, which removes moving people.
"""

from __future__ import annotations

import cv2
import numpy as np

from retailflow.config import Homography


class Heatmap:
    def __init__(self, frame_size: tuple[int, int], fps: float, cell_px: int = 4,
                 homography: Homography | None = None, floor_cell: float = 0.25):
        self.fps = fps
        self.H = None
        if homography is not None:
            src = np.array(homography.image_points, np.float32)
            dst = np.array(homography.floor_points, np.float32)
            self.H, _ = cv2.findHomography(src, dst)
            self.origin = dst.min(axis=0)
            span = dst.max(axis=0) - self.origin
            self.cell = floor_cell
            self.shape = (int(np.ceil(span[1] / floor_cell)) + 1, int(np.ceil(span[0] / floor_cell)) + 1)
        else:
            self.origin = np.zeros(2, np.float32)
            self.cell = float(cell_px)
            w, h = frame_size
            self.shape = (int(np.ceil(h / cell_px)), int(np.ceil(w / cell_px)))
        self.grid = np.zeros(self.shape, np.float64)

    def to_cells(self, points: np.ndarray) -> np.ndarray:
        """Image points (N, 2) -> integer (row, col) cells, dropping points off the grid."""
        pts = np.asarray(points, np.float32).reshape(-1, 2)
        if self.H is not None and len(pts):
            pts = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), self.H).reshape(-1, 2)
        cells = np.floor((pts - self.origin) / self.cell).astype(int)[:, ::-1]  # (row, col)
        ok = (cells[:, 0] >= 0) & (cells[:, 0] < self.shape[0]) & (cells[:, 1] >= 0) & (cells[:, 1] < self.shape[1])
        return cells[ok]

    def add(self, points, frames: int = 1) -> None:
        """Add one frame's feet points (each person = frames/fps seconds)."""
        cells = self.to_cells(np.asarray(points, np.float32).reshape(-1, 2))
        np.add.at(self.grid, (cells[:, 0], cells[:, 1]), frames / self.fps)

    def add_cells(self, cell_counts: dict[tuple[int, int], int]) -> None:
        """Add pre-binned person-frames {(row, col): frames} (see Pipeline's track buffer)."""
        for (r, c), k in cell_counts.items():
            self.grid[r, c] += k / self.fps

    def smoothed(self, sigma_cells: float = 2.0) -> np.ndarray:
        return cv2.GaussianBlur(self.grid.astype(np.float32), (0, 0), sigma_cells)

    def render(self, background: np.ndarray | None = None, sigma_cells: float = 2.0,
               alpha: float = 0.55) -> np.ndarray:
        """Colour image of the heatmap (square-root scaled), over the background if given.

        Overlaying on the camera image only makes sense for the image grid.
        """
        g = self.smoothed(sigma_cells)
        g = np.sqrt(g / g.max()) if g.max() > 0 else g
        col = cv2.applyColorMap((g * 255).astype(np.uint8), cv2.COLORMAP_JET)
        if background is None or self.H is not None:
            return col
        col = cv2.resize(col, (background.shape[1], background.shape[0]), interpolation=cv2.INTER_LINEAR)
        return cv2.addWeighted(background, 1 - alpha, col, alpha, 0)


def overlay_colors(grid: np.ndarray, size: tuple[int, int], sigma_cells: float = 2.0,
                   min_frac: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    """Colour image + alpha mask (both at image `size` = (w, h)) for drawing a heatmap
    over live video. Unlike `render`, cells nobody visited stay transparent, so the
    camera picture is only tinted where people have spent time."""
    g = cv2.GaussianBlur(grid.astype(np.float32), (0, 0), sigma_cells)
    pos = g[g > 0]
    # Scale to the 99th percentile, not the maximum: one spot where someone stood for
    # minutes would otherwise make every other visited area nearly invisible.
    m = float(np.percentile(pos, 99)) if pos.size else 0.0
    v = np.sqrt(np.clip(g / m, 0, 1)) if m > 0 else np.zeros_like(g)
    col = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_JET)
    col = cv2.resize(col, size, interpolation=cv2.INTER_LINEAR)
    # Visited cells get at least 20 % opacity, the hottest 65 %; unvisited stay clear.
    alpha = np.where(v > min_frac, 0.20 + 0.45 * v, 0.0)
    alpha = cv2.resize(alpha.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    return col, alpha


def blend_overlay(image: np.ndarray, col: np.ndarray, alpha: np.ndarray) -> None:
    """Blend an overlay from `overlay_colors` into `image`, in place."""
    # cv2.blendLinear: per-pixel weights in optimised code, about 1.4 ms on a 1920x1072
    # frame against 64 ms for the same blend in NumPy (it decided whether a live preview
    # keeps up). Results differ by at most 1 grey level (rounding).
    a = alpha.astype(np.float32, copy=False)
    image[:] = cv2.blendLinear(image, col, 1.0 - a, a)


def heatmap_from_table(table, frame_size, fps, cell_px=4, homography=None) -> Heatmap:
    """Offline heatmap from a tracks or ground-truth table (feet points)."""
    hm = Heatmap(frame_size, fps, cell_px, homography)
    if len(table):
        fx = table.feet_x if "feet_x" in table else (table.x1 + table.x2) / 2
        fy = table.feet_y if "feet_y" in table else table.y2
        hm.add(np.column_stack([fx, fy]))  # one call; each row is one person-frame
    return hm


def similarity(a: np.ndarray, b: np.ndarray, top: float = 0.10) -> dict:
    """Compare two (smoothed) heatmaps of the same shape.

    pearson     correlation of cell values (1 = same pattern)
    hotspot_iou overlap of the hottest `top` share of cells in each map
    mass_ratio  total person-seconds predicted / true
    """
    a, b = a.ravel().astype(float), b.ravel().astype(float)
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    k = max(1, int(len(a) * top))
    ha, hb = set(np.argsort(a)[-k:]), set(np.argsort(b)[-k:])
    return {"pearson": round(r, 3), "hotspot_iou": round(len(ha & hb) / len(ha | hb), 3),
            "mass_ratio": round(a.sum() / b.sum(), 3) if b.sum() > 0 else float("nan")}


def median_background(frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median of sampled frames: a people-free background for rendering."""
    return np.median(np.stack(frames), axis=0).astype(np.uint8)
