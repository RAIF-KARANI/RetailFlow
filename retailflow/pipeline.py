"""The per-camera processing loop: source -> detect/track -> analytics -> draw -> outputs.

Per frame:
  1. detector + tracker  -> people (track ID + box)
  2. staff filter         -> flags tracks that stay in staff zones (staff_flag event)
  3. line counter         -> line_cross events (IN / OUT)
  4. zone tracker         -> zone_enter (at the visit's real start) and zone_exit
                             (at its real end, value = dwell seconds). Occupancy over
                             time is rebuilt from these visits by the backend.
  5. heatmap              -> person-seconds per cell, customers only. Each track's
                             cells are buffered until it ends (unseen 2 s); then it
                             is dropped if staff or static (mannequin, static_flag)
  6. drawing (optional)   -> head blur first, then overlays

Outputs written to `out_dir`:
  tracks.csv    every tracked box in every frame (frame, track_id, x1..y2, conf).
                Saving raw tracks lets evaluate.py re-count offline with
                different settings WITHOUT re-running the detector.
  events.csv    line crossings, zone visits, staff / static flags
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
from retailflow.events import CsvEventWriter, Event, MultiSink
from retailflow.heatmap import Heatmap, blend_overlay, median_background, overlay_colors
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


def live_lag_stats(lags: list[float], fps: float, expected_frames: int | None = None,
                   received: int | None = None, skipped: int = 0,
                   indices: list[int] | None = None, max_skip_frac: float = 0.01) -> dict:
    """Did a live stream keep up with real time?

    lag(frame) = wall time since the stream opened - frame index / fps.
    At start-up lag jumps (model warm-up while frames queue) and then usually
    FALLS while that backlog is drained faster than real time. So the decision
    uses the TREND of lag over the second half of the run (least-squares slope,
    seconds of lag per second of video):
        slope ~ 0  -> keeping up;   slope > 0 -> falling behind
        realtime_factor = 1 / (1 + slope)   (video seconds processed per wall second)
    A stream that delivered far fewer frames than expected (the server dropped a
    too-slow reader, or the connection broke) does not keep up either.

    Live sources skip frames when processing is behind (sources.LatestFrameReader),
    which keeps the lag flat. So keeping up ALSO requires that at most
    `max_skip_frac` of the received frames were skipped; `processed_frac` says
    how many were analysed.
    """
    a = np.asarray(lags, float)
    idx = np.asarray(indices if indices is not None else range(len(a)), float)
    received = received if received is not None else len(a)
    half = slice(len(a) // 2, None)
    t = idx[half] / fps
    slope = float(np.polyfit(t, a[half], 1)[0]) if len(a[half]) >= 2 * fps else float("nan")
    ended_early = expected_frames is not None and received < 0.95 * expected_frames
    skip_frac = skipped / received if received else 0.0
    # One lag value per second of video (the first frame processed in each second).
    secs = (idx // max(1, round(fps))).astype(int)
    first = np.r_[True, secs[1:] != secs[:-1]]
    return {"frames": len(a), "received": received, "skipped": skipped,
            "processed_frac": round(1 - skip_frac, 4),
            "expected_frames": expected_frames, "ended_early": ended_early,
            "lag_max_s": round(float(a.max()), 3), "lag_end_s": round(float(a[-1]), 3),
            "lag_slope": round(slope, 4) if slope == slope else None,
            "realtime_factor": round(1 / (1 + slope), 3) if slope == slope else None,
            "keeps_up": (not ended_early) and slope == slope and slope < 0.02
                        and skip_frac <= max_skip_frac,
            "lag_per_s": [round(float(v), 3) for v in a[first]]}


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

    def add(self, frame: int, p, heat: Heatmap, step: int = 1) -> None:
        self.last_frame = frame
        self.feet.append(p.feet)
        self.heights.append(p.y2 - p.y1)
        for r, c in heat.to_cells(np.array([p.feet])):
            self.cells[(int(r), int(c))] += step


class Pipeline:
    """Runs one camera stream end to end."""

    def __init__(self, scene: Scene, tracker: PersonTracker, counting: CountingConfig,
                 out_dir: str | Path, save_video: bool = False, run_info: dict | None = None,
                 blur: bool = False, zone_cfg: ZoneConfig = ZoneConfig(), sinks=(),
                 heat_overlay: bool = False, preview=None):
        self.scene = scene
        self.tracker = tracker
        self.counting = counting
        self.counter = LineCounter(scene.line, counting)
        self.out_dir = Path(out_dir)
        self.save_video = save_video
        self.run_info = run_info or {}
        self.blur = blur
        self.zone_cfg = zone_cfg
        self.sinks = list(sinks)      # extra event sinks, e.g. ApiEventSender
        self.heat_overlay = heat_overlay
        # Live browser preview (retailflow.preview.PreviewServer): gets the annotated frame,
        # with the heatmap overlay, only while someone is watching.
        self.preview = preview

    def _close_track(self, tid: int, buf: _TrackBuffer, heat: Heatmap, staff: StaffFilter,
                     events: MultiSink, frame: int, ts: float) -> None:
        """Track ended: flag it as a static object, or add it to the heatmap if it is a customer."""
        if self.scene.static_filter and is_static(np.array(buf.feet), np.array(buf.heights), heat.fps):
            self.static_ids.add(tid)
            events.write(Event(self.scene.camera_id, frame, ts, "static_flag", tid))
        elif not staff.is_staff(tid):
            heat.add_cells(buf.cells)

    def run(self, source: FrameSource, show: bool = False) -> dict:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        cam, fps = self.scene.camera_id, source.fps
        events = MultiSink([CsvEventWriter(self.out_dir / "events.csv"), *self.sinks])
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
        # Recent frame timestamps, so back-dated zone events get the right time on
        # live streams too (where timestamps are wall-clock, not frame / fps).
        lags: list[float] = []
        lag_idx: list[int] = []
        recent_ts: dict[int, float] = {}
        keep = max(1, round(10 * fps))

        def frame_time(f: int) -> float:
            if f in recent_ts:
                return recent_ts[f]
            # A frame the live reader skipped: nearest processed frame +- the gap.
            near = [k for k in recent_ts if abs(k - f) <= keep]
            if not near:
                return round(f / fps, 3)
            k = min(near, key=lambda k: abs(k - f))
            return round(recent_ts[k] + (f - k) / fps, 3)
        buffers: dict[int, _TrackBuffer] = {}          # live tracks, for end-of-track decisions
        # Live overlay heatmap (--heatmap-overlay): every person counts AT ONCE, because the
        # staff / mannequin decision only exists when a track ends. The saved heatmap and the
        # dashboard keep using the filtered one (`heat`).
        live_heat = Heatmap((source.width, source.height), fps, heatmap_cell_px(source.width))
        overlay, overlay_fi = None, -10**9
        overlay_every = max(1, round(fps / 2))         # recolour twice per second
        use_overlay = self.heat_overlay or self.preview is not None
        track_end_gap = max(1, round(TRACK_END_S * fps))
        self.static_ids: set[int] = set()

        band = self.counter.band
        proc_ms: list[float] = []   # detect + track + analytics per frame
        ids_seen: set[int] = set()
        t_start = time.perf_counter()
        n = 0
        prev_fi = -1
        stopped_by_user = False
        for frame in source:
            if self.preview is not None and self.preview.stop_requested:
                stopped_by_user = True                   # Stop button on the preview page
                break
            t0 = time.perf_counter()
            fi, ts = frame.index, round(frame.timestamp, 3)
            # Frames this one stands for: 1, or more when a live stream skipped frames
            # because processing was behind (sources.LatestFrameReader).
            step, prev_fi = max(1, fi - prev_fi), fi
            if source.is_live:
                # Lag = wall time since the stream opened - position of this frame in the
                # stream. Flat = keeping up with real time; growing = falling behind.
                lags.append(frame.timestamp - fi / fps)
                lag_idx.append(fi)
            recent_ts[fi] = ts
            recent_ts.pop(fi - keep, None)
            people = self.tracker.update(frame.image)
            feet = {p.track_id: p.feet for p in people}

            for tid in staff.update(feet, step):
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

            # Zone events carry the TRUE visit start / end frame (they are detected a bit
            # later, after the debounce), so a database can rebuild occupancy over time.
            for kind, tid, zname, info in zones.update(fi, feet, step):
                is_staff = staff.is_staff(tid)
                if kind == "enter":
                    events.write(Event(cam, info, frame_time(info), "zone_enter", tid,
                                       zone=zname, staff=is_staff))
                else:
                    visits_n += 1
                    events.write(Event(cam, info.end_frame, frame_time(info.end_frame), "zone_exit", tid,
                                       zone=zname, value=info.dwell_s, staff=is_staff))

            # Heatmap: whether a track is staff or a static object (mannequin) is only
            # known when it ends, so each track's heatmap cells wait in a buffer.
            for p in people:
                buffers.setdefault(p.track_id, _TrackBuffer()).add(fi, p, heat, step)
            for tid in [t for t, b in buffers.items() if fi - b.last_frame > track_end_gap]:
                self._close_track(tid, buffers.pop(tid), heat, staff, events, fi, ts)
            proc_ms.append((time.perf_counter() - t0) * 1000)
            n += 1

            if fi // bg_every != (fi - step) // bg_every and len(bg_frames) < BACKGROUND_SAMPLES:
                bg_frames.append(frame.image.copy())   # before any drawing

            if use_overlay:                               # build up on EVERY frame
                pts = [pt for tid, pt in feet.items() if not staff.is_staff(tid)]
                if pts:
                    live_heat.add(pts, step)
            to_preview = self.preview is not None and self.preview.wants_frame()
            if writer is not None or show or to_preview:
                img = frame.image
                if self.blur:
                    blur_heads(img, [(p.x1, p.y1, p.x2, p.y2) for p in people])
                if use_overlay:
                    if overlay is None or fi - overlay_fi >= overlay_every:
                        overlay = overlay_colors(live_heat.grid, (img.shape[1], img.shape[0]))
                        overlay_fi = fi
                    blend_overlay(img, *overlay)
                draw.draw_zones(img, self.scene.zones, zones.occupancy())
                draw.draw_line(img, self.scene.line, band)
                draw.draw_people(img, people, self.counting.point, staff.staff)
                rate = 1000 / (sum(proc_ms[-30:]) / len(proc_ms[-30:]))
                draw.draw_hud(img, [f"IN {self.counter.counts[IN]}   OUT {self.counter.counts[OUT]}",
                                    f"people {len(people)}   {rate:.0f} FPS",
                                    f"t={fi / fps:.1f}s (video time)"])
                if writer is not None:
                    writer.write(img)
                if to_preview:
                    self.preview.publish(img)
                if show:
                    cv2.imshow(cam, img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        for _, tid, zname, visit in zones.finish():
            visits_n += 1
            events.write(Event(cam, visit.end_frame, frame_time(visit.end_frame), "zone_exit",
                               tid, zone=zname, value=visit.dwell_s, staff=staff.is_staff(tid)))
        last_fi = max(prev_fi, 0)
        last_ts = frame_time(last_fi)
        for tid in list(buffers):
            self._close_track(tid, buffers.pop(tid), heat, staff, events, last_fi, last_ts)
        wall_s = time.perf_counter() - t_start
        events.close()
        tracks_f.close()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        np.save(self.out_dir / "heatmap.npy", heat.grid)
        background = median_background(bg_frames) if bg_frames else None
        if background is not None:
            cv2.imwrite(str(self.out_dir / "heatmap.png"), heat.render(background))

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
            "frames": n,                       # frames analysed
            "video_frames": prev_fi + 1,       # stream position reached (includes skipped frames)
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
        if self.preview is not None:
            summary["preview_frames"] = self.preview.published
            summary["stopped_by_user"] = stopped_by_user
        if lags:
            summary["live"] = live_lag_stats(lags, fps, source.max_frames,
                                             received=prev_fi + 1, skipped=source.frames_skipped,
                                             indices=lag_idx)
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        for sink in self.sinks:
            if hasattr(sink, "finish"):
                # The background is only sent when the scene uses the image grid
                # (a floor-plan grid is not drawn over the camera image).
                try:
                    sink.finish(summary, heat.grid, heat.cell, floor=heat.H is not None,
                                background=background if heat.H is None else None)
                except Exception as e:   # API down: the local outputs are already written
                    print(f"WARNING: could not send the session summary: {e}")
                    summary.setdefault("sink_errors", []).append(str(e))
        return summary
