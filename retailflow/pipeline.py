"""The per-camera processing loop: source -> detect/track -> analytics -> draw -> outputs.

Outputs written to `out_dir`:
  tracks.csv    every tracked box in every frame (frame, track_id, x1..y2, conf).
                Saving raw tracks lets evaluate.py re-count offline with
                different counting settings WITHOUT re-running the detector.
  events.csv    line crossings (and zone events from Phase 4)
  summary.json  configuration, counts and timing for this run
  annotated.mp4 optional, with --save-video
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2

from retailflow import draw
from retailflow.config import Scene
from retailflow.counting import CountingConfig, LineCounter
from retailflow.detector import PersonTracker
from retailflow.events import CsvEventWriter, Event
from retailflow.sources import FrameSource


class Pipeline:
    """Runs one camera stream end to end."""

    def __init__(self, scene: Scene, tracker: PersonTracker, counting: CountingConfig,
                 out_dir: str | Path, save_video: bool = False, run_info: dict | None = None):
        self.scene = scene
        self.tracker = tracker
        self.counting = counting
        self.counter = LineCounter(scene.line, counting)
        self.out_dir = Path(out_dir)
        self.save_video = save_video
        self.run_info = run_info or {}

    def run(self, source: FrameSource, show: bool = False) -> dict:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        events = CsvEventWriter(self.out_dir / "events.csv")
        tracks_f = open(self.out_dir / "tracks.csv", "w", newline="")
        tracks_w = csv.writer(tracks_f)
        tracks_w.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "conf"])
        writer = None
        if self.save_video:
            writer = cv2.VideoWriter(str(self.out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     source.fps, (source.width, source.height))

        band = self.counter.band
        proc_ms: list[float] = []   # detect + track + analytics per frame
        ids_seen: set[int] = set()
        t_start = time.perf_counter()
        n = 0
        for frame in source:
            t0 = time.perf_counter()
            people = self.tracker.update(frame.image)
            for p in people:
                ids_seen.add(p.track_id)
                tracks_w.writerow([frame.index, p.track_id, f"{p.x1:.1f}", f"{p.y1:.1f}",
                                   f"{p.x2:.1f}", f"{p.y2:.1f}", f"{p.conf:.3f}"])
                anchor = p.feet if self.counting.point == "feet" else p.centre
                direction = self.counter.update(p.track_id, frame.index, anchor)
                if direction:
                    events.write(Event(self.scene.camera_id, frame.index, round(frame.timestamp, 3),
                                       "line_cross", p.track_id, direction, x=anchor[0], y=anchor[1]))
            proc_ms.append((time.perf_counter() - t0) * 1000)
            n += 1

            if writer is not None or show:
                img = frame.image
                draw.draw_line(img, self.scene.line, band)
                draw.draw_people(img, people, self.counting.point)
                fps = 1000 / (sum(proc_ms[-30:]) / len(proc_ms[-30:]))
                draw.draw_hud(img, [f"IN {self.counter.counts['in']}   OUT {self.counter.counts['out']}",
                                    f"people {len(people)}   {fps:.0f} FPS",
                                    f"t={frame.timestamp:.1f}s (video time)"])
                if writer is not None:
                    writer.write(img)
                if show:
                    cv2.imshow(self.scene.camera_id, img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        wall_s = time.perf_counter() - t_start
        events.close()
        tracks_f.close()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        summary = {
            **self.run_info,
            "camera_id": self.scene.camera_id,
            "source": source.source,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "counting": asdict(self.counting),
            "band_px": band,
            "frames": n,
            "video_fps": source.fps,
            "counts": dict(self.counter.counts),
            "unique_track_ids": len(ids_seen),
            "avg_proc_ms": round(sum(proc_ms) / max(n, 1), 2),
            # Throughput including decoding and drawing (what a user would see).
            "throughput_fps": round(n / wall_s, 2) if wall_s > 0 else None,
        }
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary
