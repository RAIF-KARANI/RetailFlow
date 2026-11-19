"""Common ground-truth loader for CAVIAR (CVML XML) and MOTChallenge (gt.txt).

Every dataset is converted to ONE table format so that evaluation code never
needs to know where the labels came from:

    frame     int   0-based frame index, matching the order frames are decoded
    track_id  int   person ID, unique within one sequence
    x1, y1    float top-left corner of the box (pixels)
    x2, y2    float bottom-right corner of the box (pixels)
    feet_x    float bottom-centre x  -> where the person touches the floor
    feet_y    float bottom-centre y
    role      str   CAVIAR role label (e.g. "walker"); "" for MOT

A `Sequence` describes the media (video file or image folder) plus fps and size.
"""

from __future__ import annotations

import configparser
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

GT_COLUMNS = ["frame", "track_id", "x1", "y1", "x2", "y2", "feet_x", "feet_y", "role"]

# MOTChallenge class IDs. Only "pedestrian" (1) is counted; MOT20 also labels
# static people (7), people on vehicles, reflections, etc., which we exclude.
MOT_PEDESTRIAN = 1


@dataclass(frozen=True)
class Sequence:
    """One evaluation sequence: where its frames are and how to read its labels."""

    name: str          # e.g. "EnterExitCrossingPaths1cor" or "MOT17-04-FRCNN"
    dataset: str       # "caviar" | "mot17" | "mot20"
    media: Path        # video file (CAVIAR) or image folder (MOT)
    gt_path: Path      # XML (CAVIAR) or gt.txt (MOT)
    fps: float
    width: int
    height: int

    def load_gt(self) -> pd.DataFrame:
        """Load this sequence's ground truth in the common table format."""
        if self.dataset == "caviar":
            return load_caviar_gt(self.gt_path)
        return load_mot_gt(self.gt_path)


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    """Add feet point columns, fix dtypes and sort. Shared by both loaders."""
    if df.empty:
        return pd.DataFrame(columns=GT_COLUMNS)
    df["feet_x"] = (df["x1"] + df["x2"]) / 2.0  # bottom-centre x
    df["feet_y"] = df["y2"]                     # bottom edge = floor contact
    df = df.astype({"frame": int, "track_id": int})
    return df[GT_COLUMNS].sort_values(["frame", "track_id"]).reset_index(drop=True)


# --------------------------------------------------------------------- CAVIAR
def load_caviar_gt(xml_path: str | Path) -> pd.DataFrame:
    """Parse a CAVIAR CVML ground-truth file.

    CAVIAR boxes are centre-based: <box xc yc w h>. Only <objectlist> entries
    are individual people; <grouplist> boxes cover several people and are
    ignored (using them would double-count).
    """
    rows = []
    root = ET.parse(xml_path).getroot()
    for frame_el in root.iter("frame"):
        frame = int(frame_el.get("number"))
        objlist = frame_el.find("objectlist")
        if objlist is None:
            continue
        for obj in objlist.findall("object"):
            box = obj.find("box")
            if box is None:
                continue
            xc, yc = float(box.get("xc")), float(box.get("yc"))
            w, h = float(box.get("w")), float(box.get("h"))
            role_el = obj.find(".//role")
            rows.append({
                "frame": frame,
                "track_id": int(obj.get("id")),
                "x1": xc - w / 2, "y1": yc - h / 2,
                "x2": xc + w / 2, "y2": yc + h / 2,
                "role": (role_el.text or "").strip() if role_el is not None else "",
            })
    return _finish(pd.DataFrame(rows))


def caviar_sequence(clip_dir: str | Path) -> Sequence:
    """Build a Sequence for a downloaded CAVIAR clip folder (e.g. .../OneLeaveShop1cor)."""
    clip_dir = Path(clip_dir)
    xml = next(clip_dir.glob("*gt.xml"))
    # Fixed by the dataset: half-resolution PAL, 384x288 at 25 fps.
    return Sequence(clip_dir.name, "caviar", clip_dir / f"{clip_dir.name}.mpg", xml, 25.0, 384, 288)


# ------------------------------------------------------------- MOTChallenge
def load_mot_gt(gt_txt: str | Path) -> pd.DataFrame:
    """Parse a MOTChallenge gt.txt file.

    Columns: frame, id, left, top, width, height, consider_flag, class, visibility.
    Frames are 1-based in the file; we convert to 0-based. We keep rows that are
    marked "consider" (flag 1) and are pedestrians (class 1), which is the
    official MOT evaluation subset.
    """
    cols = ["frame", "track_id", "left", "top", "w", "h", "consider", "cls", "vis"]
    raw = pd.read_csv(gt_txt, header=None, names=cols)
    raw = raw[(raw["consider"] == 1) & (raw["cls"] == MOT_PEDESTRIAN)].copy()
    df = pd.DataFrame({
        "frame": raw["frame"] - 1,
        "track_id": raw["track_id"],
        "x1": raw["left"], "y1": raw["top"],
        "x2": raw["left"] + raw["w"], "y2": raw["top"] + raw["h"],
        "role": "",
    })
    return _finish(df)


def mot_sequence(seq_dir: str | Path, dataset: str) -> Sequence:
    """Build a Sequence for a MOT17/MOT20 train folder using its seqinfo.ini."""
    seq_dir = Path(seq_dir)
    ini = configparser.ConfigParser()
    ini.read(seq_dir / "seqinfo.ini")
    s = ini["Sequence"]
    return Sequence(
        seq_dir.name, dataset, seq_dir / s.get("imDir", "img1"), seq_dir / "gt" / "gt.txt",
        float(s["frameRate"]), int(s["imWidth"]), int(s["imHeight"]),
    )


def discover_sequences(data_root: str | Path = "data") -> list[Sequence]:
    """Find every downloaded sequence that has ground truth."""
    root = Path(data_root)
    seqs: list[Sequence] = []
    for clip in sorted((root / "caviar").glob("*")):
        if clip.is_dir() and any(clip.glob("*gt.xml")) and any(clip.glob("*.mpg")):
            seqs.append(caviar_sequence(clip))
    for bench in ("MOT17", "MOT20"):
        for seq in sorted((root / bench / "train").glob("*")):
            if (seq / "gt" / "gt.txt").exists():
                seqs.append(mot_sequence(seq, bench.lower()))
    return seqs
