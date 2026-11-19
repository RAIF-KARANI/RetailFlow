# 03 · Detection and tracking

## The two-step idea

A camera gives us independent frames. To count people we need to know that the
person in frame 100 is the same person as in frame 101. So we:

1. **Detect** every person in every frame (boxes, no identity): YOLO26.
2. **Track**: link boxes across frames into trajectories with stable integer IDs, using ByteTrack or BoT-SORT.

Counting, zones and dwell all work on these trajectories.

## YOLO26 (detector)

- A single-stage CNN detector from Ultralytics. One forward pass gives every box
  with a class and a confidence score. We keep class 0 (`person`) only.
- **Sizes n / s / m** trade speed for accuracy (more layers and channels). On an
  8 GB laptop GPU, `n` and `s` are the realistic real-time choices; `m` is the accuracy reference.
- **Input size (`imgsz`)**: the frame is resized so its long side is `imgsz`
  (640 / 960 / 1280) before detection.
  **Why bigger helps in angled views:** in an angled camera, people far away are
  small. Measured from ground truth: in MOT17-02 (1920 px wide) the smallest 5%
  of labelled people are ≤45 px tall (median 102 px). At `imgsz=640` the frame is
  shrunk 3×, so those people become ≤15 px, only about two cells of the
  network's finest feature map (stride 8), which is too little detail to detect reliably.
  At 1280 they stay ≤30 px. The cost: compute grows with pixel count, so 1280
  is roughly 4× the work of 640. CAVIAR frames are only 384 px wide (smallest
  5% of people ≤30 px, corridor view), so a larger `imgsz` *upsamples* them
  and adds no real detail; whether it still helps is an experiment (Phase 3).
- YOLO26 is NMS-free end-to-end, so post-processing is light and the
  ONNX/TensorRT export (Phase 6) is simpler.

## ByteTrack (tracker, baseline)

Each frame, the tracker:
1. **Predicts** where each existing track should be now with a Kalman filter (a
   constant-velocity motion model over box centre, aspect ratio and height).
2. **Matches** detections to predicted tracks by box overlap (IoU) with the Hungarian algorithm.
3. **Key idea: uses low-confidence boxes too.** First it matches high-score
   detections (≥ `track_high_thresh`, 0.25). Then the leftover tracks get a
   *second* chance against low-score detections (0.1–0.25). A partly occluded
   person often gets a low score, and most trackers would drop them; ByteTrack keeps the track alive.
4. Unmatched detections scoring ≥ `new_track_thresh` (0.25) start new tracks;
   tracks unmatched for `track_buffer` frames (30, i.e. ~1 s) are deleted.

Defaults in the installed Ultralytics 8.4.215 `bytetrack.yaml`:
`track_high_thresh 0.25, track_low_thresh 0.1, new_track_thresh 0.25,
track_buffer 30, match_thresh 0.8, fuse_score True`. BoT-SORT uses the same
values plus `gmc_method sparseOptFlow, proximity_thresh 0.5,
appearance_thresh 0.8, with_reid False, model auto`.

That is why we run the detector with `conf=0.1`: the low-score boxes are
needed for step 3. Whether a box can *start* a new track is decided by the
tracker's own `new_track_thresh`.

**Example.** A shopper walks behind a pillar. Their detection score drops from
0.8 to 0.15 for 10 frames. A tracker with a 0.25 cut-off loses them and assigns
a new ID when they reappear (an ID switch, which can double count on the line).
ByteTrack matches the 0.15 boxes in its second round and keeps the same ID.

## BoT-SORT (+ ReID)

BoT-SORT extends the same idea with:
- **Camera-motion compensation (GMC)**: estimates global image motion so the
  Kalman prediction isn't fooled by a moving camera. Our cameras are static
  (measured in Phase 1), so this mostly costs time; it can be turned off.
- **ReID (appearance)**: a small embedding of each person's appearance is
  compared between detections and tracks, combined with the IoU distance.

**What ReID adds:** IoU only works if a person's box in the next frame overlaps
their previous box. When two people cross paths, or a person is occluded for
longer than the motion model can bridge, IoU and motion alone can swap or lose
IDs. Appearance (clothing colour and shape) gives a second, motion-independent
cue to re-link the right person.

**Installed version check (Phase 1):** Ultralytics 8.4.215 ships
`botsort.yaml` with `with_reid: False` and `model: auto`. With `auto`, the
appearance features are taken from the YOLO detector's own internal features
(no second network). `trackers/botsort_reid.yaml` is the same file with
`with_reid: True`. We checked that the encoder is really active when it is set.

**Privacy:** ReID embeddings live only in the tracker's memory during a run and
are never written to disk.

## How it is wired (`retailflow/detector.py`)

```python
tracker = PersonTracker("yolo26n.pt", tracker="bytetrack", imgsz=640, conf=0.1)
for frame in FrameSource(path):
    people = tracker.update(frame.image)   # [TrackedPerson(id, x1, y1, x2, y2, conf), ...]
```

`update` calls Ultralytics `model.track(..., persist=True)`, which keeps the
tracker state between calls. Each video gets a **fresh** `PersonTracker`, so IDs
and Kalman state never leak between videos.

## Results

Tracker × model × image-size comparison: `10_RESULTS.md` §2.1. In short:
- **Image size:** 640 → 960 gave the biggest gain on dev (43 → 36 errors). 1280 was
  not better (37). That includes the 384 px CAVIAR clips, where 960 means upsampling.
- **Model:** YOLO26m beat n and s (34 errors), at about 22 ms/frame against about 12.
- **Tracker:** BoT-SORT beat ByteTrack (32 vs 34). Adding ReID raised IDF1 slightly
  (0.716 → 0.720) but did not improve counts, so it is off by default.
- **Crowds:** in MOT20 the detector itself was the bottleneck, so dense cameras use a
  `crowd` profile at the full 1920 px (`retailflow/profiles.py`, `10_RESULTS.md` §2.3).
- **Phase 6:** BoT-SORT's camera-motion compensation (GMC) was switched off
  (`trackers/botsort_nogmc.yaml`): all cameras are static, it cost up to 31 ms per 1080p frame,
  and dev errors went 32 → 30. The detector runs as a TensorRT FP16 engine when one is built
  (`08_STREAMING_PERFORMANCE.md`). Keep GMC on for moving (PTZ) cameras.

Final tracking quality (per-camera profile, mean IDF1): caviar_cor 0.741,
caviar_front 0.466, MOT17-02 0.398, MOT17-04 0.616, MOT17-09 0.552, MOT20 0.34–0.62.

Baseline tracking quality (YOLO26n, ByteTrack, `imgsz=640`; mean over sequences):

| Camera | Mean IDF1 | Notes |
|---|---|---|
| caviar_cor (26) | 0.720 | |
| caviar_front (26) | 0.490 | smaller, partly occluded people behind railings |
| MOT17-02 | 0.320 | detection recall 0.237: most small, distant people missed at 640 |
| MOT17-04 | 0.351 | |
| MOT17-09 | 0.519 | recall 0.664, 47 ID switches |

Low recall on the 1080p MOT17 scenes was the motivation for testing
larger `imgsz` values and models (above).
