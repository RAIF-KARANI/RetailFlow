"""Interactive scene setup: click on a camera frame to define the counting line,
the IN direction, zones and homography points. Saves scenes/<camera_id>.json.

Usage:
  python tools/setup_scene.py --source data/caviar/OneLeaveShop1cor/OneLeaveShop1cor.mpg --camera caviar_cor
  python tools/setup_scene.py --source data/MOT17/train/MOT17-09-FRCNN/img1 --camera MOT17-09 --frame 50

If scenes/<camera_id>.json already exists, it is loaded so you can edit it.

Keys (press in the image window):
  l   line mode: click 2 points ON THE FLOOR (where feet touch the ground)
  i   IN side: click 1 point on the side people are walking INTO
  z   zone mode: click polygon corners, press ENTER to finish, then type a name in the console
  h   homography: click 4 floor points, then type their floor-plan coordinates (metres) in the console
  c   clear all zones
  b / B  band width -1 / +1 pixel
  s   save            q / ESC  quit (asks to save if unsaved)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retailflow import draw  # noqa: E402
from retailflow.config import Homography, LineConfig, Scene, Zone, load_scene, save_scene  # noqa: E402
from retailflow.sources import FrameSource  # noqa: E402

ZONE_COLOURS = [(255, 128, 0), (0, 200, 255), (200, 0, 200), (0, 255, 128), (128, 128, 255)]


def default_scene(camera_id: str, w: int, h: int) -> Scene:
    """A starting scene: a horizontal line at 2/3 height, IN = downwards."""
    y = round(h * 2 / 3)
    return Scene(camera_id, (w, h), LineConfig((w * 0.2, y), (w * 0.8, y), (w * 0.5, h * 0.9), round(h / 60)))


def parse_floor_points(text: str) -> list[tuple[float, float]]:
    """Parse "x1,y1; x2,y2; x3,y3; x4,y4" (metres) into 4 points."""
    pts = [tuple(float(v) for v in chunk.split(",")) for chunk in text.split(";") if chunk.strip()]
    if len(pts) != 4 or any(len(p) != 2 for p in pts):
        raise ValueError("need exactly 4 points like: 0,0; 5,0; 5,10; 0,10")
    return pts


def grab_frame(source: str, index: int) -> np.ndarray:
    for f in FrameSource(source, max_frames=index + 1):
        img = f.image
    return img


class SceneEditor:
    def __init__(self, image: np.ndarray, scene: Scene, path: Path):
        self.base = image
        self.scene = scene
        self.path = path
        self.mode = "line"
        self.clicks: list[tuple[int, int]] = []
        self.dirty = False
        # Display larger for small CAVIAR frames so clicking is precise.
        self.zoom = 2 if image.shape[1] < 800 else (0.5 if image.shape[1] > 1600 else 1)

    # ---- mouse --------------------------------------------------------------
    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        p = (x / self.zoom, y / self.zoom)  # back to original image pixels
        ln = self.scene.line
        if self.mode == "line":
            self.clicks.append(p)
            if len(self.clicks) == 2:
                self._set_line(LineConfig(self.clicks[0], self.clicks[1], ln.in_point, ln.band_px))
                self.clicks = []
        elif self.mode == "in":
            self._set_line(LineConfig(ln.p1, ln.p2, p, ln.band_px))
        elif self.mode in ("zone", "homography"):
            self.clicks.append(p)
            if self.mode == "homography" and len(self.clicks) == 4:
                self._finish_homography()

    def _set_line(self, line: LineConfig) -> None:
        self._replace(line=line)

    def _replace(self, **kw) -> None:
        d = dict(camera_id=self.scene.camera_id, frame_size=self.scene.frame_size, line=self.scene.line,
                 zones=self.scene.zones, homography=self.scene.homography,
                 description=self.scene.description, extra=self.scene.extra)
        d.update(kw)
        self.scene = Scene(**d)
        self.dirty = True

    def _finish_zone(self) -> None:
        if len(self.clicks) < 3:
            print("a zone needs at least 3 points")
            return
        name = input("zone name (e.g. entrance / aisle / checkout): ").strip() or f"zone{len(self.scene.zones) + 1}"
        self._replace(zones=self.scene.zones + (Zone(name, tuple(self.clicks)),))
        self.clicks = []
        print(f"added zone '{name}'")

    def _finish_homography(self) -> None:
        print("Floor-plan coordinates (metres) for the 4 clicked points, in click order.")
        while True:
            try:
                floor = parse_floor_points(input("e.g. 0,0; 5,0; 5,10; 0,10 > "))
                break
            except ValueError as e:
                print(e)
        self._replace(homography=Homography(tuple(self.clicks), tuple(floor)))
        self.clicks = []
        print("homography points saved")

    # ---- drawing ------------------------------------------------------------
    def render(self) -> np.ndarray:
        img = self.base.copy()
        for k, z in enumerate(self.scene.zones):
            pts = np.array(z.polygon, np.int32)
            cv2.polylines(img, [pts], True, ZONE_COLOURS[k % len(ZONE_COLOURS)], 2)
            cv2.putText(img, z.name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * draw.scale_for(img) + 0.2,
                        ZONE_COLOURS[k % len(ZONE_COLOURS)], 1)
        draw.draw_line(img, self.scene.line, self.scene.line.band_px)
        if self.scene.homography:
            for p in self.scene.homography.image_points:
                cv2.drawMarker(img, (int(p[0]), int(p[1])), (255, 0, 255), cv2.MARKER_CROSS, 14, 2)
        for p in self.clicks:
            cv2.circle(img, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)
        draw.draw_hud(img, [f"mode: {self.mode}  band: {self.scene.line.band_px:g}px",
                            "l line  i IN-side  z zone  h homography  s save  q quit"])
        return cv2.resize(img, None, fx=self.zoom, fy=self.zoom, interpolation=cv2.INTER_NEAREST)

    # ---- main loop ----------------------------------------------------------
    def run(self) -> None:
        win = f"setup_scene: {self.scene.camera_id}"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self.on_mouse)
        while True:
            cv2.imshow(win, self.render())
            key = cv2.waitKey(30) & 0xFF
            ln = self.scene.line
            if key == ord("l"):
                self.mode, self.clicks = "line", []
            elif key == ord("i"):
                self.mode, self.clicks = "in", []
            elif key == ord("z"):
                self.mode, self.clicks = "zone", []
            elif key == ord("h"):
                self.mode, self.clicks = "homography", []
            elif key == 13 and self.mode == "zone":  # ENTER
                self._finish_zone()
            elif key == ord("c"):
                self._replace(zones=())
            elif key in (ord("b"), ord("B")):
                band = max(0.0, ln.band_px + (1 if key == ord("B") else -1))
                self._set_line(LineConfig(ln.p1, ln.p2, ln.in_point, band))
            elif key == ord("s"):
                save_scene(self.scene, self.path)
                self.dirty = False
                print(f"saved {self.path}")
            elif key in (ord("q"), 27):
                if self.dirty and input("unsaved changes - save? [y/N] ").lower().startswith("y"):
                    save_scene(self.scene, self.path)
                break
        cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser(description="Click-to-configure a camera scene")
    p.add_argument("--source", required=True, help="video file, image folder or rtsp:// URL")
    p.add_argument("--camera", required=True, help="camera_id, also the JSON file name")
    p.add_argument("--frame", type=int, default=0, help="which frame to show")
    p.add_argument("--scenes-dir", default="scenes")
    args = p.parse_args()

    image = grab_frame(args.source, args.frame)
    path = Path(args.scenes_dir) / f"{args.camera}.json"
    h, w = image.shape[:2]
    scene = load_scene(path) if path.exists() else default_scene(args.camera, w, h)
    print(__doc__.split("Keys")[1])
    SceneEditor(image, scene, path).run()


if __name__ == "__main__":
    main()
