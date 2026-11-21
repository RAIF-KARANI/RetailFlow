# 00 · Progress log

Every step: what was done, the commands, the result, and what I learned.

---

## Phase 1: Setup and data

**Goal:** working GPU environment, the public angled-view datasets downloaded,
and one loader that reads all ground truth in the same format.

### 1.1 Environment check
- **Commands:** `py -0p`, `nvidia-smi`, `docker --version`, `node --version`, `ffmpeg -version`, `git --version`
- **Result:** everything already installed. Python 3.12.9 is available alongside the 3.13 default;
  RTX 4060 Laptop with 8 GB and driver 621.01; Docker 29.9.2; Node 22.20.0; FFmpeg 9.0.2; 139 GB free.
- **Learned:** pin the venv to 3.12 explicitly (`py -3.12 -m venv`), because the system default is 3.13.

### 1.2 Project skeleton and venv
- Created the folder structure, replaced the Flash `.gitignore` with Python/Node/data rules,
  and created `.venv` on Python 3.12.9.

### 1.3 PyTorch with CUDA + dependencies
- **Commands:** `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130`, then `pip install -r requirements.txt`
- **Result:** torch 2.14.1+cu130, ultralytics 8.4.215, motmetrics 1.4.0, lap 0.5.14.
- **Learned:** the default PyPI torch wheel on Windows is CPU-only. The `+cu130` suffix confirms the CUDA build.

### 1.4 GPU check (`tools/check_gpu.py`)
- **Result:** `CUDA available: True`, RTX 4060 with 8.0 GB. YOLO26n ran at 55.4 ms/frame
  (39.3 ms steady-state, 34.0 ms model-only).
- **Finding:** the GPU was throttled (1170 of 3105 MHz) by the laptop power profile.
  Speed numbers wait until the profile is switched. Logged in TROUBLESHOOTING.
- **Finding:** BoT-SORT in this Ultralytics version supports `with_reid` (default `False`).

### 1.5 Datasets
- **CAVIAR:** `python tools/download_caviar.py`. All 26 Portugal clips, both views,
  video + GT: 104 files, 581 MB, 0 failures.
- **MOT17:** `python tools/download_mot.py MOT17 --seqs ...` downloaded the 5.6 GB zip and extracted the 7 FRCNN train sequences.
- **Camera motion:** `python tools/check_camera_motion.py`. 02/04/09 are static (≤0.05 px/frame median);
  05/10/11/13 are moving (3.4–4.5 px/frame). The moving ones were deleted.
- **Learned:** measuring is better than trusting the paper. The background-only mask matters,
  because otherwise a crowd walking in one direction looks like camera motion.

### 1.6 Common ground-truth loader (`retailflow/groundtruth.py`)
- Converts CAVIAR XML (centre boxes, ignore group boxes) and MOT gt.txt
  (consider=1, pedestrian class, 1-based → 0-based frames) into one table with feet points.
- **Tests:** `pytest -q` → 2 passed.
- **Inventory:** `python tools/inventory.py` loaded all 55 sequences. Frame counts line up,
  some CAVIAR front-view labels are incomplete, paired views differ in length, and density
  ranges from 0.3 to 45 people per frame.
- **Learned:** corridor views are the reliable CAVIAR counting benchmark. Front views need care.

### Phase 1 status
- Done and committed. Open item: re-run the GPU check in performance mode.

---

## Phase 2: Pipeline and evaluation baseline

**Goal:** a working end-to-end pipeline, an honest evaluation, and the "before" numbers.

### 2.1 GPU re-check
- `python tools/check_gpu.py` gave YOLO26n 22.4 ms/frame (was 55.4), but the GPU was
  still power-limited during later runs. Speed is deferred to Phase 6.

### 2.2 Choosing counting lines (before running anything)
- Drew every GT feet point per camera on a frame (traffic plots), then placed lines where
  people actually cross: across the CAVIAR corridor, on the Promod shop threshold
  (front view), and across the main flow in each MOT17 scene. Saved as `scenes/*.json`.
- One scene per **camera**: all 26 corridor clips share `caviar_cor.json`.
- **Learned:** PowerShell 5.1 writes UTF-8 with a BOM, which breaks `json.loads`. The loader now accepts it.

### 2.3 Core modules
- `config.py`, `sources.py`, `detector.py`, `counting.py`, `events.py`, `draw.py`, `pipeline.py`, `run.py`.
- Smoke test: `python run.py --source data/caviar/OneLeaveShop1cor/OneLeaveShop1cor.mpg --scene scenes/caviar_cor.json --save-video`
  → 295 frames, 9 track IDs, annotated video checked visually (line, band, IN arrow, feet points).
- **Design:** the pipeline saves every tracked box (`tracks.csv`) so counting variants
  can be evaluated offline on identical tracks.

### 2.4 Evaluation
- `retailflow/evaluate.py`: true counts = same line logic on GT feet points with the band
  (fixed definition); WAPE that cannot cancel errors; IDF1/MOTA via motmetrics.
- **Learned:** motmetrics' `iou_matrix` uses `np.asfarray` (gone in NumPy 2), so we compute IoU ourselves.
- Truth sanity check: shop-entrance truth matches clip names (`WalkByShop1front` = 0 crossings,
  `TwoLeaveShop*` = OUTs only). 210 true crossings over 55 sequences.
- `tools/setup_scene.py`: click tool for line, IN side, zones, homography.
- Tests: `pytest -q` → 27 passed.

### 2.5 Baseline run
- `python benchmark.py --configs baseline` (YOLO26n, ByteTrack, 640): 55 sequences, 74,693 frames, ~1.5 h.
- `python tools/check_alignment.py`: GT frames line up with decoded frames (peak at offset 0).
- `python evaluate.py --configs baseline --group camera`:
  - classic baseline (centre point, simple line): **WAPE 92.4%**, 129 over-counts
  - feet point + band on the **same tracks**: **WAPE 27.6%**, 16 over-counts
  - remaining errors are mostly **missed people** (42 of 58), e.g. MOT17-02 recall 0.24 at 640.
- **Learned:** in angled views the box centre floats above the floor. People walking
  past a shop had their centre on the threshold line. The feet point is not optional.
- **Learned:** Python buffers stdout when piped to a file. Use `python -u` for background runs.

### Phase 2 status
- Done and committed. Next: Phase 3 grid (tracker × model × imgsz), MOT20, min-age tuning.

---

## Phase 3: Reliable counting and the grid

**Goal:** the best detector/tracker/counting settings, chosen on dev only, and an
honest before/after on unseen test cameras including busy scenes.

### 3.1 MOT20
- Measured the size first (4.8 GB zip), then `python tools/download_mot.py MOT20 --seqs ...`.
- All 4 training sequences are static (`check_camera_motion.py`: 0.09–0.55 px/frame).
  Lines were drawn from traffic plots: 1,066 true crossings, 46–195 people per frame.
- **Dev/test split by camera:** dev = CAVIAR corridor + MOT17; test = CAVIAR front + MOT20.

### 3.2 Counting sweep on baseline tracks
- `python evaluate.py --configs baseline --sweep`: 32 combinations. Wide bands hurt the
  shop-entrance camera badly (3× band: test 65.3%). Min age barely matters.

### 3.3 Automated grid (`benchmark.py --plan onefactor`)
- Image size → model → tracker, one factor at a time, scored on dev counting errors,
  with a tie rule of 1 error → the faster config wins. 7 configs × 55 sequences.
- Chosen: **YOLO26m, 960, BoT-SORT**. Dev errors 43 → 32. ReID did not improve counts.
- The fresh n640 run reproduced the Phase 2 baseline exactly (43 dev errors).
- Sweep on the final tracks: the default counting settings tie for best, so no extra filters.

### 3.4 MOT20 and the crowd profile
- m960 on MOT20: 74.2% WAPE. MOT20-03/05 almost entirely missed.
- **Diagnosis:** the detector finds about 34 of 195 people on MOT20-05 at 960 px, about 128 at 1920.
  NMS is not the cause (YOLO26 is NMS-free; `iou` and `max_det` changed nothing).
- After asking: added a per-camera **crowd profile** (`retailflow/profiles.py`, scene
  `"profile"` field), used at ≥ 15 people per frame. Checked on the dev crowd cameras first
  (no change in counts, better recall).
- Result: MOT20 914 (baseline) → 327 errors. MOT20-02 got worse (8 → 47, over-counting),
  reported and not tuned away.
- **Learned:** in crowds, counting accuracy is capped by detection recall. Measure
  detections before tuning the tracker or the line.

### 3.5 Code and tests
- `retailflow/profiles.py`; `run.py` reads the scene's profile (still overridable);
  `evaluate.py --configs final` scores each camera with its own profile (renamed `final_phase3` in Phase 6).
- `pytest -q` → 38 passed.

### Phase 3 status
- Done and committed. Open: MOT20-02 over-counting and MOT20-05 recall (known limitations);
  crowd-profile speed (83–112 ms/frame) is a Phase 6 topic.

---

## Phase 4: Zones, dwell, heatmaps, staff, blur

**Goal:** the "if possible" analytics (zones, dwell, heatmaps), scored against ground
truth like counting, plus staff filtering and head blurring.

### 4.1 Zones placed from traffic plots
- Drew ground-truth feet heatmaps over each camera and placed zones where people walk and stop
  (CAVIAR corridor: 3, shop front: 3, MOT17-09: 2). Saved in the scene files.

### 4.2 Modules
- `zones.py` (debounced visits, dwell, occupancy), `heatmap.py` (person-seconds grid, optional
  floor homography, median background), `staff.py`, `static_filter.py`, `privacy.py` (head pixelation).
- Pipeline emits `zone_enter`/`zone_exit`/`occupancy`/`staff_flag`/`static_flag` events and writes
  `heatmap.npy` + `heatmap.png`. `run.py --blur-heads`.
- **Learned:** track-level decisions (staff? mannequin?) are only possible when a track ends, so
  the pipeline buffers each track's heatmap cells until then.

### 4.3 Evaluation (`evaluate_zones.py`)
- Same zone/heatmap code on ground truth and on predictions.
- Debounce sweep on dev: short settings over-count visits by up to 3× (every ID break = a new visit).
  Chose 2 s / 2 s: dev visit WAPE 26.9%, with time-in-zone about the same (≈23%).
- **Learned:** dwell time is far more robust than visit counts; visit counts depend on ID stability.

### 4.4 Mannequins and staff
- Test-camera heatmap correlation was only 0.15: the hottest predicted spots were mannequins.
  Added a static-object filter (post-diagnosis, like the crowd profile). Checked on dev (corridor
  0.80 → 0.93) and on ground truth (it would remove 26% of real people on the MOT17-04 street), so it is a per-camera switch.
- CAVIAR has no staff labels. The staff rule (45 s in the shop) found the real assistant in 1 of 2
  clips; in the other the assistant's track broke into 8+ IDs. Reported as not reliable on this camera.
- Fixed my own evaluation mistake: the staff rule must be applied to ground truth too, otherwise
  removing real staff from predictions looks like an error.

### 4.5 Tests
- `pytest -q` → 58 passed.

### Phase 4 status
- Done and committed. Open: shop-front walkway under-detection (glass railing); staff rule depends on
  stable IDs; no floor-plan heatmaps because no dataset has floor measurements.

---

## Phase 5: Backend and dashboard

**Goal:** events from the pipeline into PostgreSQL through FastAPI, and a Next.js dashboard
showing counts, zones, dwell and heatmaps. Everything runs with Docker Compose.

### 5.1 Services
- `docker compose up -d --build`: Postgres 16, FastAPI (`backend/`), Next.js (`dashboard/`).
  Only the listed stack plus the packages it needs: `uvicorn` (server) and `psycopg` (driver).
  No ORM, no chart library.
- Host ports 5432 and 3000 were taken on this machine, so the stack uses 55432 and 3001.

### 5.2 Design decisions
- **Sessions:** each video keeps its own video-time axis; nothing is put on a fake clock.
- **Retroactive flags:** a `customer_events` view drops all events of tracks later flagged staff or static.
- **Occupancy from visits:** zone events now carry the real visit start/end, and the API rebuilds
  occupancy from them, so staff and mannequins can still be removed afterwards. The pipeline's 1 Hz
  occupancy samples were dropped for this reason.

### 5.3 Pipeline to API
- `run.py --api http://127.0.0.1:8000` (`retailflow/sender.py`, stdlib only): batches, retry after 5 s, local CSV always written.
- **Learned:** `localhost` cost 2 s per request (IPv6 tried first, Docker listens on IPv4 only).
- Demo data: 4 CAVIAR clips, MOT17-09 and MOT20-02 processed with `--api`.

### 5.4 Checks
- `pytest -q` → 70 passed, including the API integration test against the running stack.
- Dashboard checked in the browser: desktop and phone width (fixed a grid overflow), no console errors.

### Phase 5 status
- Done and committed. Not in scope: authentication, connection pooling, retention policy (doc 07 §7).

---

## Phase 6: Streaming and performance

**Goal:** real-time live streams, several at once, without losing accuracy.

### 6.1 Export (`tools/export_models.py`)
- ONNX and TensorRT FP16 engines with dynamic shapes (rectangular input, as with PyTorch).
- The 1920 engine ran out of memory in Ultralytics' FP16 step, so it was built directly with the TensorRT API
  from the size-independent FP16 ONNX.
- **Learned the hard way:** Ultralytics auto-installed packages and downgraded ONNX Runtime to a CUDA 12
  build, which silently ran on the CPU (200–1,100 ms per frame). Fixed, auto-install disabled, everything
  pinned in `requirements-export.txt`. Always check which device actually ran.

### 6.2 Speed and accuracy
- `tools/benchmark_speed.py`: TensorRT FP16 inference 2.6–3.5× faster than PyTorch FP32. ONNX Runtime FP32
  was slower than PyTorch.
- All 62 sequences re-run with TensorRT: dev errors 32 → 32, test 342 → 335. Same accuracy.
- Found the real bottleneck on 1080p: BoT-SORT's camera-motion compensation (CPU optical flow), useless for
  static cameras. Without it: dev errors 30, dev 12.3 ms/frame (Phase 3: 28.8). Adopted for both profiles;
  `--backend auto` uses the engine when built.

### 6.3 Live streams
- MediaMTX (Compose profile `streaming`, pinned 1.21.3) + `tools/simulate_rtsp.py` (FFmpeg loops videos as RTSP cameras).
- Real-time check: lag trend + MediaMTX drop warnings. Two measurement bugs found and fixed on the way:
  start-up backlog made lag look like it was shrinking, and server-side frame drops hid a growing lag.
  Model warm-up now happens before the stream is opened.
- `run_multi.py` results: 4 small cameras, 2 full-HD, or 1 crowd camera in real time on this laptop.
  At 4 × 1080p the CPU (decoding) is the limit.
- Live RTSP → API → dashboard checked in the browser (session marked live, counts updating).

### 6.4 Tests
- `pytest -q` → 78 passed (new: streaming helpers, lag check, exported-backend config names, profile backends).

### Phase 6 status
- Done and committed. Not done: GPU video decoding, batched multi-camera inference (doc 08 §6).

---

## Phase 7: Multi-camera

**Goal:** use CAVIAR's two simultaneous views (corridor + shop front) together; MMPTRACK if feasible.

### 7.1 MMPTRACK
- Official page checked: research use only, signed agreement e-mailed to the organisers, no public link or size.
  Not requested (that is the owner's decision; turnaround does not fit). Documented in `tools/download_data.md`.

### 7.2 Time sync from the burned-in clock
- `retailflow/osd_clock.py`: template matching on the 8 px bitmap clock, 31 labelled glyph shapes,
  strict accept rule; `fit_start` = robust (median) start per clip.
- All 72,518 frames readable, 0 outliers. Front view starts 3.04–3.12 s before the corridor view (25/26 clips).
- **Learned:** first template set (4 clips) misread a "0" as "8"; collecting from all clips + a margin rule fixed it.
- Independent check with ground truth: front threshold crossings coincide with corridor track starts/ends
  60% of the time at the clock sync (the peak), 14–28% when shifted by ±2 s, 19% at random moments.

### 7.3 Two cameras, one door
- The Promod door is visible at the corridor view's left edge. Door line placed from the ground-truth
  coincidences; site file `scenes/sites/caviar_promod.json`; `retailflow/multicam.py` fuses crossings by time.
- `evaluate_multicam.py`: front only 14 errors / 72; corridor only 61; union 16–22; intersection 63–69.
  **The second camera does not help here**: the door is at its image edge (recall 0.73 vs 0.92).

### 7.4 Tests
- `pytest -q` → 85 passed (clock reading on synthetic and real frames, robust start fit, fusion, site file).

### Phase 7 status
- Done and committed. Not built: dashboard site page on the recording clock; cross-camera re-identification
  (no cross-view identity labels to test it).

---

## Phase 8: Final docs and demo

### 8.1 Consistency pass
- Closed the last two TODOs: `check_gpu.py` in performance mode gives 16.3 ms/frame (was 55.4 while throttled).
- Re-ran `evaluate.py --configs n640_bytetrack final` and checked every number in the new headline table
  (doc 10 §0, README) against it.

### 8.2 Demo
- Annotated, head-blurred demo videos for a shop-front clip, a corridor clip and 30 s of MOT20-02
  (`outputs/demo/*.mp4`, not committed); one frame of each in `docs/img/`.
- `demo.ps1`: services + simulated RTSP camera + live pipeline → dashboard, tested end to end.
- **Learned:** in Windows PowerShell 5.1, `$ErrorActionPreference = "Stop"` makes Docker's normal stderr
  output fatal. The script checks exit codes instead.

### 8.3 README
- Rewritten: requirement-by-requirement evidence, headline before/after table, honest list of what is not
  solved, quick start, architecture, docs index, licences (including YOLO26's AGPL-3.0).

### Phase 8 status
- Done and committed. Project complete for this brief.

---

## After Phase 8: one hour of real footage (MEVA) and live heatmap overlay

- **Data:** MEVA (CC BY 4.0) camera G420, indoor entrance lobby, busiest continuous hour (914 MB).
  Camera picked from a 3 MB slice of every camera (`tools/meva_index.py`, `tools/download_data.md` §5).
- **New options:** `--heatmap-overlay` (heatmap so far tinted onto the saved video or preview) and
  `--duration SECONDS` (stops after that much video at the source's own fps; `demo.ps1` used to assume 25 fps).
- **One hour live with the API:** 108,000 frames, 89 in / 122 out, 336 zone visits, 22.0 ms per frame on
  average. It did **not** keep up through the rush minutes (real-time factor 0.962, 71 s lag at the end):
  details and fixes in doc 08 §9.
- **Fix:** live streams are read in a thread that keeps only the newest frame; skipped frames are counted
  and weighted. Same hour again, nothing else running: lag at most 3.1 s, 0 decode errors, 0 server drops,
  97.5% of frames analysed, 87 in / 123 out (doc 08 §9.1). A stalled stream now ends the run in at most 15 s.
- **Also fixed:** `demo.ps1` left the simulator's FFmpeg running (grandchild of the venv launcher).
- **Learned:** a camera that keeps up on average can still fall behind at peak traffic. Check the busiest
  minutes, not only the average frame time, and measure with nothing else on the machine.
- **Live preview:** `run.py --preview` / `demo.ps1 -Preview` serve the running pipeline's annotated frames
  (boxes, line, zones, heatmap) at http://127.0.0.1:8090 as MJPEG, drawn only while someone watches.
  Cost on the busiest 5 minutes: 2.3% → 2.5% skipped frames. **Learned:** the NumPy heatmap blend took
  64 ms per full-HD frame; `cv2.blendLinear` takes 1.4 ms.
- **Dashboard link and Stop button:** a live session run with `--preview` shows "Watch live video" on the
  dashboard; the preview page's "Stop run" ends the run cleanly (tested live: session closed,
  `stopped_by_user: true`, simulator stopped). The session length now comes from the stream position
  (`video_frames`), not from the number of analysed frames.
- **Zone-to-zone transitions** (item 3 of the original plan): per-track zone paths → transition matrix,
  in run outputs, API and dashboard. Error on steps: corridor 22.4% → 9.3%, shop front 91.5% → 56.6%
  (baseline → final, doc 05 §8). **Learned:** most errors are steps lost at track breaks.
- **Head blur in the demo:** the preview showed faces because `demo.ps1` never passed `--blur-heads`;
  it now always does.
- **Cross-camera IDs** (plan Version 2): SmartSpaces scene 071, 15 cameras tracked one by one, linked on the
  floor (`retailflow/mtmc.py`). Test (minutes 2–7): global IDF1 0.110 → 0.281. **Learned:** with perfect
  per-camera tracks the same linking gets 0.825, so the per-camera tracker is the bottleneck; appearance
  matching is next.
- **Colour matching across cameras:** torso/legs colour histograms (no new dependency, kept in memory only)
  join pieces geometry cannot. Test global IDF1 0.281 → 0.369 (doc 11 §7).
- **Store live on the dashboard:** `run_store.py` runs all 15 cameras at once (one shared detector, a tracker
  per camera, re-linking every 3 s with stable numbers) and the dashboard's Stores page shows a live floor
  map. Live IDs are weaker than offline (global IDF1 0.197 vs 0.369, minutes 2–7) because each camera gets
  only 3–4 analyses per second (doc 11 §8).
- **Batch engine, linking process, floor heatmap:** a batch-16 TensorRT engine and a separate linking process
  raised the live rate from 3.3 to 4.88 analyses per camera per second and live global IDF1 (minutes 2–7)
  from 0.197 to 0.294. The store page has a blueprint-style floor heatmap. **Learned:** a Python thread is
  no place for heavy pandas work next to a real-time loop; keep TensorRT batch sizes constant.
- **Live camera view of the store:** `run_store.py --preview` shows all 15 cameras (or one) with each box labelled
  by its store-wide number, sent back from the linking process; Stop button; "Watch live cameras" on the store page.
- **Store layout and zones:** a top-down floor plan stitched from the 15 calibrated cameras (photo and blueprint),
  drawn under the dashboard map, with 9 store zones in metres: people now, visits, stay times and zone-to-zone moves
  across the whole store (store-level movement from the original plan).
- **Store zones scored:** ground-truth floor positions through the same zone code. Offline: 0.85 people error per
  zone per second, time in zones 24.8%, moves 30.4%; live 1.40 / 42.9% / 42.7%; visits about 2× too high (split IDs).
  **Learned:** occupancy and time-in-zone survive ID splits; visit counts do not.
