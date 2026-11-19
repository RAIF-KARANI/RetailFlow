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
