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

## Phase 5

### Docker: `failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine`
- **Cause:** Docker Desktop was not running.
- **Fix:** start Docker Desktop and wait until `docker info` answers.

### `ports are not available ... 127.0.0.1:5432` / `Bind for 127.0.0.1:3000 failed: port is already allocated`
- **Cause:** host ports 5432 and 3000 were already in use by other software on this machine.
- **Fix:** Postgres is published on host port 55432 and the dashboard on 3001 (`DASHBOARD_PORT`).
  Other programs were left untouched.

### Every API call from Python took ~2 s
- **Symptom:** `urllib` to `http://localhost:8000` took 2.1 s per call; `http://127.0.0.1:8000` took 0.014 s.
- **Cause:** `localhost` resolves to IPv6 `::1` first. Docker publishes the port on IPv4 `127.0.0.1`
  only, so each request waits for the IPv6 attempt to fail.
- **Fix:** the sender rewrites `localhost` to `127.0.0.1`; docs and tests use `127.0.0.1`. Browsers
  fall back quickly, so the dashboard is not affected.

### Dashboard panels overflowed the screen on phones
- **Cause:** grid items default to `min-width: auto`, so wide charts and tables forced columns wider than the screen.
- **Fix:** `.grid > * { min-width: 0 }` and `minmax(min(420px, 100%), 1fr)` columns.

## Phase 6

### Ultralytics installed packages by itself and downgraded ONNX Runtime
- **Symptom:** ONNX Runtime ran at 215–1,092 ms per frame (CPU speed). `pip list` showed
  `onnxruntime-gpu 1.22.0` although 1.32.0 had been installed, plus new packages
  (`nvidia-modelopt`, `polygraphy`, `onnx_graphsurgeon`, `onnxscript`...).
- **Cause:** during the TensorRT export, Ultralytics' requirement check pip-installed what its FP16
  path needs and pinned `onnxruntime-gpu` to 1.22, a **CUDA 12** build. This machine has CUDA 13
  (from torch), so the CUDA provider failed to load and ONNX Runtime **silently fell back to the CPU**.
- **Fix:** reinstalled `onnxruntime-gpu==1.32.0` (CUDA 13); `tools/export_models.py` sets
  `YOLO_AUTOINSTALL=false`; all export packages are pinned in `requirements-export.txt`.
- **Check:** `InferenceSession(...).get_providers()` must list `CUDAExecutionProvider` first.
  ORT 1.32 needs the CUDA 13 / cuDNN 9 DLLs, which are found once `torch` has been imported.
- **Lesson:** a "speed" result can be a silent CPU fallback. Always check which device actually ran.

### TensorRT 1920 export: `Failed to allocate memory` in ModelOpt AutoCast
- **Cause:** Ultralytics converts the ONNX to FP16 by running a CPU reference inference at full
  size. At 1920x1920 this ran out of memory.
- **Fix:** the FP16 ONNX written during the 960 export has dynamic axes, so the 1920 engine is built
  straight from it with the TensorRT API (`tools/export_models.py --from-fp16-onnx`), copying
  Ultralytics' metadata header from the 960 engine. TensorRT's "Skipping tactic ... insufficient memory"
  warnings during a build are normal on an 8 GB GPU (it skips candidate kernels that don't fit).

### Multi-stream: "reader is too slow, discarding N frames" even for one fast stream
- **Symptom:** MediaMTX logged discards for a single MOT17-09 stream processed at 20 ms per
  33 ms frame; `error while decoding MB` messages from OpenCV's H.264 decoder.
- **Cause:** each worker opened the RTSP stream and only then ran its first inference, which builds
  the TensorRT context (several seconds). Nobody read the stream meanwhile, MediaMTX's per-reader
  queue overflowed and it dropped data (the decoder then sees broken frames).
- **Fix:** workers warm the model up on a blank frame **before** opening the stream.
- **Also learned:** dropped frames make the lag check look *better* (the frame counter falls behind the
  real video), so `run_multi.py` now reads MediaMTX's drop warnings for its run and a stream only
  counts as real time if nothing was dropped. MediaMTX's "frames" in that message are RTP packets.

### Tools in `tools/` could not `import retailflow`
- **Cause:** a script's own folder (`tools/`) is first on `sys.path`, not the repo root.
- **Fix:** each tool inserts the repo root into `sys.path`, as the earlier tools already did.
