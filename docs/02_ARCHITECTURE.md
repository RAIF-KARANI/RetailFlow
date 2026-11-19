# 02 · Architecture

## System overview

```
                      ┌──────────────── HOST (GPU) ─────────────────────────────────┐
  video file ─┐       │                                                              │
  image dir  ─┼─► sources.py ─► detector.py ─────► counting.py ──► events.py ──┬──► events.csv / tracks.csv
  RTSP stream ┘   FrameSource    YOLO26 +          LineCounter     Event       │    summary.json
                  (frame, ts,    ByteTrack /       zones.py (P4)   (anonymous) │    annotated.mp4 (draw.py)
                   image)        BoT-SORT          heatmap.py (P4)             │
                                 → TrackedPerson                               └──► HTTP POST (Phase 5)
                      │          (id, box, conf)                                         │
                      │  config.py: scenes/<camera>.json (line, IN side, band, zones, homography)
                      └──────────────────────────────────────────────────────────────┼───┘
                                                                                     ▼
                      ┌──────────── DOCKER COMPOSE (Phase 5) ─────────────────────────┐
                      │  FastAPI backend  ──►  PostgreSQL  ◄──  Next.js dashboard     │
                      └───────────────────────────────────────────────────────────────┘

  OFFLINE EVALUATION
  data/ (CAVIAR XML, MOT gt.txt) ─► groundtruth.py ─┐
  outputs/runs/<config>/<seq>/tracks.csv ───────────┼─► evaluate.py ─► outputs/results.csv
  scenes/*.json ────────────────────────────────────┘    (true vs predicted counts, IDF1, WAPE)
```

## Modules

| Module | Responsibility |
|---|---|
| `retailflow/config.py` | Load/validate/save per-camera scene JSON; map dataset sequences to camera scenes |
| `retailflow/sources.py` | One iterator for video files, image folders and RTSP; yields `(index, timestamp, image)` |
| `retailflow/detector.py` | `PersonTracker`: YOLO26 + tracker → list of `TrackedPerson(id, box, conf)` with `feet` / `centre` |
| `retailflow/counting.py` | Line geometry, band/hysteresis crossing logic, min track age, once-per-direction |
| `retailflow/events.py` | Anonymous `Event` schema and CSV writer |
| `retailflow/draw.py` | Overlays for the annotated video (line, band, IN arrow, boxes, feet, HUD) |
| `retailflow/pipeline.py` | The per-frame loop tying everything together; writes all outputs |
| `retailflow/groundtruth.py` | CAVIAR + MOT ground truth → one common table |
| `retailflow/evaluate.py` | True counts from GT, recounting saved tracks, IDF1/MOTA, WAPE summaries |
| `run.py` | CLI: process one source |
| `benchmark.py` | Run detector/tracker configs over all sequences (resumable) |
| `evaluate.py` | CLI: evaluate all runs → `outputs/results.csv` |
| `tools/setup_scene.py` | Click to define line, IN side, zones, homography |

## Event schema

Every analytic output is an `Event` (see `retailflow/events.py`):

| Field | Type | Example | Notes |
|---|---|---|---|
| `camera_id` | str | `caviar_front` | which camera |
| `frame` | int | 812 | frame index in the source |
| `timestamp` | float | 32.48 | seconds on the **video timeline** (files) or since stream start (live) |
| `type` | str | `line_cross` | `line_cross`, `zone_enter`, `zone_exit`, `occupancy` (P4) |
| `track_id` | int | 17 | session-local integer, not linked to any identity |
| `direction` | str | `in` | for `line_cross` |
| `zone` | str | `checkout` | for zone events (P4) |
| `x`, `y` | float | 270.5, 140.0 | anchor point in pixels at the event |
| `value` | float | 4 | occupancy count / dwell seconds (P4) |

No images, crops or appearance features are ever stored (see Privacy below).

## Key design decisions

1. **Save raw tracks, count offline for evaluation.** The pipeline writes every
   tracked box to `tracks.csv`. Detection/tracking is the expensive GPU step;
   counting is cheap. So counting variants (centre vs feet, band, min age) are
   compared on the *same* tracks in seconds, which isolates the counting logic's
   effect from detector randomness.
2. **One scene file per camera, not per video.** All 26 CAVIAR corridor clips
   come from one physical camera and share `scenes/caviar_cor.json`, the same
   way a real store would configure each camera once.
3. **Geometry lives in the scene; behaviour lives in run-time settings.** The line,
   IN side and band width depend on where the camera hangs. Model, tracker,
   image size and counting filters are passed at run time, so one scene can be
   benchmarked many ways.
4. **Truth is defined once.** True IN/OUT come from running the counting logic on
   ground-truth feet points with the scene band. This never changes between
   experiments, so before/after comparisons are fair.
5. **Model file picks the backend.** `.pt` / `.onnx` / `.engine` go through the
   same `PersonTracker`, so the Phase 6 speed comparison uses identical
   analytics code.
6. **Video-timeline timestamps.** Dataset videos have no real clock time; we
   report seconds into the video and label charts accordingly. No fake business hours.
7. **CV on the host, services in Docker.** The pipeline needs the GPU directly,
   while the database, API and dashboard are easy to containerise (Phase 5).

## Privacy

- Stored data: camera ID, frame/time, integer track ID, positions, zone names, directions.
- Never stored: images (except the optional demo video), face crops, identities,
  ReID embeddings. ReID features exist only in tracker memory during a run (Phase 3).
- Demo videos use `--blur-heads` (Phase 4).
