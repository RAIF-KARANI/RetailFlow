# Troubleshooting log

Format: **problem → cause → fix**. Newest at the bottom.

## Phase 1

### `ModuleNotFoundError: No module named 'yaml'` in `tools/check_gpu.py`
- **Cause:** Ultralytics 8.4.215 no longer installs PyYAML as a dependency.
- **Fix:** the script reads the three BoT-SORT keys it needs from the YAML text
  directly, so no extra package is needed.

### YOLO26n takes ~40–55 ms/frame on an RTX 4060
- **Symptom:** `check_gpu.py` reported 55.4 ms/frame. After 30 warm-ups it was
  39.3 ms, with 34.0 ms in model inference alone, which is several times slower than expected.
- **Cause:** `nvidia-smi` showed the GPU at **1170 MHz of 3105 MHz max** and drawing 32 W
  while plugged in, so the laptop power profile was throttling the GPU.
- **Fix:** set Armoury Crate to *Performance/Turbo* and Windows power mode to
  *Best performance*, then re-run `python tools/check_gpu.py`.
  TODO: run and fill in the result after switching.
- **Lesson:** check clocks with `nvidia-smi --query-gpu=pstate,clocks.gr,clocks.max.gr --format=csv`
  before any benchmark, otherwise speed numbers are not comparable.

### The original `.gitignore` was a Flash/ActionScript template
- **Fix:** replaced it with Python + Node + data/outputs/weights rules.

## Phase 2

### `json.decoder.JSONDecodeError: Expecting value: line 1 column 1` loading a scene
- **Cause:** Windows PowerShell 5.1 `Out-File -Encoding utf8` writes a UTF-8
  **byte-order mark** (BOM). Python's `json.loads` rejects it.
- **Fix:** `load_scene` reads with `encoding="utf-8-sig"` (accepts BOM or no BOM);
  `save_scene` writes plain UTF-8. Test: `test_loads_file_with_utf8_bom`.

### `motmetrics.distances.iou_matrix` fails: `np.asfarray was removed in the NumPy 2.0 release`
- **Cause:** motmetrics 1.4.0 still uses a NumPy 1.x function.
- **Fix:** `retailflow/evaluate.py` computes the IoU distance matrix itself
  (`iou_distance`, unit-tested) and only uses motmetrics' accumulator and metric
  functions, which work fine on NumPy 2.5.

### GPU still power-limited during the baseline
- **Symptom:** pipeline at ~72 ms/frame on 384x288 CAVIAR. `nvidia-smi`: 92% utilisation
  but only 1125 MHz and ~30 W.
- **Effect:** speed only. Detection results do not depend on clock speed, so accuracy numbers are valid.
- **Action:** speed numbers are re-measured in Phase 6 in performance mode.

### Background runs show an empty log
- **Cause:** Python buffers stdout when it is piped to a file.
- **Fix:** run background jobs with `python -u`.

## Phase 3

### Packed crowds: almost nobody detected (MOT20-05)
- **Symptom:** 74% counting error on MOT20; about 19 tracks per frame for 195 people.
- **Diagnosis:** the problem is raw detection, not tracking: about 34 boxes with conf ≥ 0.1 at 960 px vs about 128 at 1920 px.
  `iou` and `max_det` changed nothing (YOLO26 is NMS-free).
- **Fix:** per-camera `crowd` profile at full resolution (doc 10 §2.3).

## Phase 4

### Seeking in CAVIAR MPEG-2 files returns wrong frames
- **Symptom:** `cv2.CAP_PROP_FRAME_COUNT` reported 18 frames for a 1,675-frame clip, and
  `cap.set(CAP_PROP_POS_FRAMES, n)` returned the wrong images.
- **Cause:** the MPEG-2 files have no reliable index for random access.
- **Fix:** always read CAVIAR videos sequentially (`FrameSource`), never seek.

### Mannequins counted as people in zones and heatmaps
- **Symptom:** the hottest predicted heatmap spots were inside the shop, where the ground truth has nobody;
  one "person" was present in every frame of `ShopAssistant1front`.
- **Fix:** static-object filter, switched on per camera (doc 05 §5).
