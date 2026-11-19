# 01 · Setup and data

## 1. Machine

| Item | Value (checked on this machine) |
|---|---|
| OS | Windows 11 Home |
| CPU | Intel Core i7-13650HX |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB, driver 621.01 |
| Python | 3.12.9 in `.venv` (system default is 3.13; we pin 3.12 for library support) |
| Node.js | 22.20.0 (npm 11.4.2) |
| Docker | 29.9.2 |
| FFmpeg | 9.0.2 (gyan.dev full build) |
| Git | 2.55.0 |

## 2. Python environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# 1) PyTorch from the CUDA wheel index (the default PyPI wheel on Windows is CPU-only)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# 2) everything else
pip install -r requirements.txt
```

Installed versions: torch 2.14.1+cu130, ultralytics 8.4.215, OpenCV 5.1, NumPy 2.5.4,
pandas 3.1.1, motmetrics 1.4.0, pytest 9.2.0.

## 3. GPU check

```powershell
python tools/check_gpu.py
```

Result on this machine:

```
torch 2.14.1+cu130 | CUDA build 13.0
CUDA available: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU | 8.0 GB
YOLO26n on GPU: 55.4 ms/frame at 640 (dummy frame, includes pre/post-processing)
ultralytics 8.4.215 botsort.yaml -> ['gmc_method: sparseOptFlow', 'with_reid: False', 'model: auto']
```

**Note on speed.** 55 ms/frame (39 ms after 30 warm-ups; 34 ms of that is the
model itself) is far slower than an RTX 4060 should be. `nvidia-smi` showed the GPU
at 1170 MHz of a 3105 MHz maximum while plugged in, i.e. the laptop's power
profile was throttling it. See `TROUBLESHOOTING.md`. All speed numbers in later
docs are measured after switching to a performance power profile.

Re-run in performance mode (Phase 8): **16.3 ms/frame** for the same check (same versions, same
dummy frame). This quick check includes start-up effects; the proper per-backend measurements
are in `08_STREAMING_PERFORMANCE.md` (e.g. YOLO26m TensorRT FP16: 5 ms inference at 960 px).

**ReID support.** The installed Ultralytics BoT-SORT config has `with_reid`
(default `False`) and `model: auto`, so ReID can be enabled with a custom tracker
YAML. Done in Phase 3.

## 4. Docker

Docker Desktop 29.9.2 is installed. It runs PostgreSQL, the FastAPI backend and
the dashboard (Phase 5). The CV pipeline runs on the host so it can use the GPU directly.

## 5. Datasets

Exact download steps and licences: [`tools/download_data.md`](../tools/download_data.md).

### Why each dataset counts as angled / side-view

- **CAVIAR (Portugal shopping centre):** real shopping-centre CCTV. The
  *corridor* camera looks down the corridor at an angle, and the *front* camera
  looks across at the shop entrances from the side. Both are fixed, wall-mounted
  and neither is overhead, so this matches the client's cameras most closely.
- **MOT17-02 / 04 / 09:** fixed cameras looking down at a street or square at
  an oblique angle (04 is high and steep, 02 and 09 are near eye-level). People
  are small and often occluded, which stresses detection the same way a far
  aisle does.
- **MOT20 (Phase 3):** fixed, elevated, angled cameras over very dense crowds.
  This is the "busy scene" stress test.

### Moving cameras rejected

Camera motion was **measured**, not assumed: `tools/check_camera_motion.py`
matches ORB keypoints on the background only (person boxes masked out) between
consecutive frames and fits a similarity transform with RANSAC. The table is in
`tools/download_data.md`: 02, 04 and 09 move ≤0.05 px/frame (static), while
05, 10, 11 and 13 move 3.4–4.5 px/frame (handheld or vehicle). The four moving
ones were deleted from `data/`.

### Inventory

`python tools/inventory.py` writes `outputs/dataset_inventory.csv`.

| Dataset | Sequences | Resolution | FPS | Frames | Mean people / frame | Max people / frame |
|---|---|---|---|---|---|---|
| CAVIAR corridor | 26 | 384x288 | 25 | 295 – 3700 per clip | 1.05 – 8.61 | 12 |
| CAVIAR front | 26 | 384x288 | 25 | 295 – 3700 per clip | 0.26 – 3.02 | 7 |
| MOT17-09 | 1 | 1920x1080 | 30 | 525 | 10.14 | 13 |
| MOT17-02 | 1 | 1920x1080 | 30 | 600 | 30.97 | 36 |
| MOT17-04 | 1 | 1920x1080 | 30 | 1050 | 45.29 | 52 |

What the inventory showed:
1. No ground-truth frame runs past the end of its video, so frame indexing
   lines up. A visual overlay check follows in Phase 2.
2. Some CAVIAR **front-view** labels stop early or miss people (e.g.
   `OneLeaveShop2front`: 1 person labelled, GT ends at frame 289 of 1119). So
   **corridor views are the main counting benchmark**, and front views are used for multi-camera.
3. Paired views can differ in length (`TwoEnterShop3`: 1149 vs 1075 frames),
   confirming they must be time-aligned before multi-camera matching.

## 6. Common ground-truth format

`retailflow/groundtruth.py` converts both label formats into one table:

| Column | Meaning |
|---|---|
| `frame` | 0-based frame index (MOT files are 1-based and are shifted) |
| `track_id` | person ID within the sequence |
| `x1 y1 x2 y2` | box corners in pixels (CAVIAR's centre-based boxes are converted) |
| `feet_x feet_y` | bottom-centre of the box, where the person stands on the floor |
| `role` | CAVIAR role label (e.g. `walker`), empty for MOT |

Filtering rules:
- **CAVIAR:** only `<objectlist>` people. `<grouplist>` boxes cover several
  people and would double-count, so they are ignored.
- **MOT:** only rows with `consider=1` and class `1` (pedestrian), the official
  evaluation subset. Static people, vehicles, reflections etc. are excluded.

Tests: `tests/test_groundtruth.py` (2 tests, passing).
