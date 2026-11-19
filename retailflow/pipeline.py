"""The per-camera processing loop: source -> detect/track -> analytics -> draw -> outputs.

Per frame:
  1. detector + tracker  -> people (track ID + box)
  2. staff filter         -> flags tracks that stay in staff zones (staff_flag event)
  3. line counter         -> line_cross events (IN / OUT)
  4. zone tracker         -> zone_enter / zone_exit events (exit carries dwell seconds)
                             and occupancy events once per second
  5. heatmap              -> person-seconds per cell, customers only. Each track's
                             cells are buffered until it ends (unseen 2 s); then it
                             is dropped if staff or static (mannequin, static_flag)
  6. drawing (optional)   -> head blur first, then overlays

Outputs written to `out_dir`:
  tracks.csv    every tracked box in every frame (frame, track_id, x1..y2, conf).
                Saving raw tracks lets evaluate.py re-count offline with
                different settings WITHOUT re-running the detector.
  events.csv    line crossings, zone visits, occupancy samples, staff flags
  heatmap.npy   raw person-seconds grid;  heatmap.png  rendered over a people-free background
  summary.json  configuration, counts, zone summary and timing for this run
  annotated.mp4 optional, with --save-video (heads blurred with --blur-heads)
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from retailflow import draw
from retailflow.config import Scene
from retailflow.counting import IN, OUT, CountingConfig, LineCounter
from retailflow.detector import PersonTracker
from retailflow.events import CsvEventWriter, Event
from retailflow.heatmap import Heatmap, median_background
from retailflow.privacy import blur_heads
from retailflow.sources import FrameSource
from retailflow.staff import StaffFilter
from retailflow.static_filter import is_static
from retailflow.zones import ZoneConfig, ZoneTracker

BACKGROUND_SAMPLES = 15      # frames kept for the median background
BACKGROUND_EVERY_S = 2.0     # one sample every 2 s of video
TRACK_END_S = 2.0            # a track unseen this long is treated as finished


def heatmap_cell_px(width: int) -> int:
    """Image-grid cell size giving roughly 190 cells across (2 px at 384, 10 px at 1920)."""
    return max(2, round(width / 192))


class _TrackBuffer:
    """What the pipeline remembers about a live track until it ends.

    Heatmap contributions are kept as {cell: frames} (bounded by the cells a
    track touches, so a mannequin seen for hours stays small), plus the feet
    range and box heights the static-object rule needs.
    """

    def __init__(self):
        self.cells: Counter = Counter()
        self.feet: list[tuple[float, float]] = []
        self.heights: list[float] = []
        self.last_frame = -1

    def add(self, frame: int, p, heat: Heatmap) -> None:
        self.last_frame = frame
        self.feet.append(p.feet)
        self.heights.append(p.y2 - p.y1)
        for r, c in heat.to_cells(np.array([p.feet])):
            self.cells[(int(r), int(c))] += 1


class Pipeline:
    """Runs one camera stream end to end."""

    def __init__(self, scene: Scene, tracker: PersonTracker, counting: CountingConfig,
                 out_dir: str | Path, save_video: bool = False, run_info: dict | None = None,
                 blur: bool = False, zone_cfg: ZoneConfig = ZoneConfig()):
        self.scene = scene
        self.tracker = tracker
        self.counting = counting
        self.counter = LineCounter(scene.line, counting)
        self.out_dir = Path(out_dir)
        self.save_video = save_video
        self.run_info = run_info or {}
        self.blur = blur
        self.zone_cfg = zone_cfg

    def _close_track(self, tid: int, buf: _TrackBuffer, heat: Heatmap, staff: StaffFilter,
                     events: CsvEventWriter, frame: int, ts: float) -> None:
        """Track ended: flag it as a static object, or add it to the heatmap if it is a customer."""
        if self.scene.static_filter and is_static(np.array(buf.feet), np.array(buf.heights), heat.fps):
            self.static_ids.add(tid)
            events.write(Event(self.scene.camera_id, frame, ts, "static_flag", tid))
        elif not staff.is_staff(tid):
            heat.add_cells(buf.cells)

    def run(self, source: FrameSource, show: bool = False) -> dict:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        cam, fps = self.scene.camera_id, source.fps
        events = CsvEventWriter(self.out_dir / "events.csv")
        tracks_f = open(self.out_dir / "tracks.csv", "w", newline="")
        tracks_w = csv.writer(tracks_f)
        tracks_w.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "conf"])
        writer = None
        if self.save_video:
            writer = cv2.VideoWriter(str(self.out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (source.width, source.height))

        zones = ZoneTracker(self.scene.zones, fps, self.zone_cfg)
        staff = StaffFilter(self.scene.staff, self.scene.zones, fps)
        heat = Heatmap((source.width, source.height), fps, heatmap_cell_px(source.width),
                       self.scene.homography)
        crossings: list[tuple[int, str]] = []          # (track_id, direction), for the summary
        visits_n = 0
        bg_frames: list[np.ndarray] = []
        bg_every = max(1, round(BACKGROUND_EVERY_S * fps))
        occ_every = max(1, round(fps))                 # occupancy sample once per second
        buffers: dict[int, _TrackBuffer] = {}          # live tracks, for end-of-track decisions
        track_end_gap = max(1, round(TRACK_END_S * fps))
        self.static_ids: set[int] = set()

        band = self.counter.band
        proc_ms: list[float] = []   # detect + track + analytics per frame
        ids_seen: set[int] = set()
        t_start = time.perf_counter()
        n = 0
        for frame in source:
            t0 = time.perf_counter()
            fi, ts = frame.index, round(frame.timestamp, 3)
            people = self.tracker.update(frame.image)
            feet = {p.track_id: p.feet for p in people}

            for tid in staff.update(feet):
                events.write(Event(cam, fi, ts, "staff_flag", tid, staff=True))

            for p in people:
                ids_seen.add(p.track_id)
                tracks_w.writerow([fi, p.track_id, f"{p.x1:.1f}", f"{p.y1:.1f}",
                                   f"{p.x2:.1f}", f"{p.y2:.1f}", f"{p.conf:.3f}"])
                anchor = p.feet if self.counting.point == "feet" else p.centre
                direction = self.counter.update(p.track_id, fi, anchor)
                if direction:
                    crossings.append((p.track_id, direction))
                    events.write(Event(cam, fi, ts, "line_cross", p.track_id, direction,
                                       x=anchor[0], y=anchor[1], staff=staff.is_staff(p.track_id)))

            for kind, tid, zname, visit in zones.update(fi, feet):
                is_staff = staff.is_staff(tid)
                if kind == "enter":
                    events.write(Event(cam, fi, ts, "zone_enter", tid, zone=zname, staff=is_staff))
                else:
                    visits_n += 1
                    events.write(Event(cam, fi, ts, "zone_exit", tid, zone=zname,
                                       value=visit.dwell_s, staff=is_staff))
            if self.scene.zones and fi % occ_every == 0:
                for zname, count in zones.occupancy().items():
                    events.write(Event(cam, fi, ts, "occupancy", zone=zname, value=count))

            # Heatmap: whether a track is staff or a static object (mannequin) is only
            # known when it ends, so each track's heatmap cells wait in a buffer.
            for p in people:
                buffers.setdefault(p.track_id, _TrackBuffer()).add(fi, p, heat)
            for tid in [t for t, b in buffers.items() if fi - b.last_frame > track_end_gap]:
                self._close_track(tid, buffers.pop(tid), heat, staff, events, fi, ts)
            proc_ms.append((time.perf_counter() - t0) * 1000)
            n += 1

            if fi % bg_every == 0 and len(bg_frames) < BACKGROUND_SAMPLES:
                bg_frames.append(frame.image.copy())   # before any drawing

            if writer is not None or show:
                img = frame.image
                if self.blur:
                    blur_heads(img, [(p.x1, p.y1, p.x2, p.y2) for p in people])
                draw.draw_zones(img, self.scene.zones, zones.occupancy())
                draw.draw_line(img, self.scene.line, band)
                draw.draw_people(img, people, self.counting.point, staff.staff)
                rate = 1000 / (sum(proc_ms[-30:]) / len(proc_ms[-30:]))
                draw.draw_hud(img, [f"IN {self.counter.counts[IN]}   OUT {self.counter.counts[OUT]}",
                                    f"people {len(people)}   {rate:.0f} FPS",
                                    f"t={frame.timestamp:.1f}s (video time)"])
                if writer is not None:
                    writer.write(img)
                if show:
                    cv2.imshow(cam, img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        for _, tid, zname, visit in zones.finish():
            visits_n += 1
            events.write(Event(cam, max(n - 1, 0), round((n - 1) / fps, 3) if fps else 0.0, "zone_exit",
                               tid, zone=zname, value=visit.dwell_s, staff=staff.is_staff(tid)))
        last_ts = round((n - 1) / fps, 3) if fps else 0.0
        for tid in list(buffers):
            self._close_track(tid, buffers.pop(tid), heat, staff, events, max(n - 1, 0), last_ts)
        wall_s = time.perf_counter() - t_start
        events.close()
        tracks_f.close()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        np.save(self.out_dir / "heatmap.npy", heat.grid)
        if bg_frames:
            cv2.imwrite(str(self.out_dir / "heatmap.png"), heat.render(median_background(bg_frames)))

        # Customer counts: drop every crossing by a track that was EVER flagged staff.
        customer = {IN: 0, OUT: 0}
        for tid, d in crossings:
            if not staff.is_staff(tid):
                customer[d] += 1
        summary = {
            **self.run_info,
            "camera_id": cam,
            "source": source.source,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "counting": asdict(self.counting),
            "band_px": band,
            "frames": n,
            "video_fps": fps,
            "counts": dict(self.counter.counts),
            "counts_customers": customer,
            "staff_track_ids": sorted(staff.staff),
            "static_track_ids": sorted(self.static_ids),
            "zone_visits": visits_n,
            "blur_heads": self.blur,
            "unique_track_ids": len(ids_seen),
            "avg_proc_ms": round(sum(proc_ms) / max(n, 1), 2),
            # Throughput including decoding and drawing (what a user would see).
            "throughput_fps": round(n / wall_s, 2) if wall_s > 0 else None,
        }
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary
