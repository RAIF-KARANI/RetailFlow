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
  Result after switching: `check_gpu.py` 16.3 ms/frame (was 55.4); the GPU reaches about 2,500 MHz
  under load (clock sampled during the Phase 6 stream tests).
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

### Tools in `tools/` could not `import retailflow` (Phase 6)
- **Cause:** a script's own folder (`tools/`) is first on `sys.path`, not the repo root.
- **Fix:** each tool inserts the repo root into `sys.path`, as the earlier tools already did.

## Phase 7

### On-screen clock read 8 minutes wrong in some clips
- **Symptom:** `tools/sync_caviar_views.py` reported hundreds of frames off by exactly 480 s or 300 s in
  5 clips; `10:20:00.037` was read as `10:28:00.037`.
- **Cause:** the digit templates came from 4 clips only. A minute "0" in a slightly different shape
  matched the "8" template within the 3-pixel tolerance.
- **Fix:** collected glyph shapes from all 52 videos (6 unseen shapes, labelled), and a digit is
  accepted only if it is ≤ 2 px from a template and ≥ 2 px better than any other digit.
  Result: 72,518 frames readable, 0 outliers. Four clips' start times changed by 5–8 minutes:
  the error had also reached the median start.

### Checking a sync by correlating people counts gave noise
- **Cause:** the two CAVIAR views watch different areas; the same person shows up seconds apart.
- **Instead:** shop-threshold crossings in one view vs track starts/ends in the other (docs/11 §2).

### My own sync check first paired frames with the wrong sign
- **Symptom:** the burned-in clocks in a "synchronised" pair read 10:07:52 vs 10:07:58.
- **Rule:** corridor frame = front frame + `offset_frames` (offset is negative: the front view starts
  earlier). Looking at the clocks in the rendered pairs caught it at once.

## Phase 8

### `demo.ps1` stopped at the first `docker compose` call with `NativeCommandError`
- **Cause:** in Windows PowerShell 5.1, `$ErrorActionPreference = "Stop"` turns any text a native
  program writes to stderr into a fatal error, and Docker writes its normal progress there.
- **Fix:** the script runs Docker through `cmd /c ... >NUL 2>&1` and checks `$LASTEXITCODE`.

## After Phase 8

### One-hour live MEVA run fell 71 s behind ("reader is too slow" in MediaMTX)
- **Symptom:** `keeps_up: false`, `realtime_factor` 0.962; MediaMTX discarded packets and the log has
  249 H.264 "error while decoding MB" lines.
- **Cause:** in the busiest minutes (many people at once) one 1920×1072 frame takes longer than its
  33 ms; measured offline 27.3 fps on a busy stretch. MediaMTX then drops packets for the slow reader.
- **Fix:** live streams are read in a thread that keeps only the newest frame (whole frames are skipped,
  not packets). Same hour: lag at most 3.1 s, 0 decode errors, 0 drops (doc 08 §9.1).

### Heatmap overlay was almost invisible
- **Cause:** colours were scaled to the maximum cell, and one spot where people stood still dominated.
- **Fix:** scale to the 99th percentile and give every visited cell at least 20% opacity.

### `demo.ps1` left FFmpeg encoders running after it ended
- **Cause:** a venv's `python.exe` is a launcher that starts the real interpreter, so the simulator's
  FFmpeg is a grandchild; the script only stopped direct children.
- **Fix:** `taskkill /PID <simulator> /T /F` stops the whole tree. Checked: 0 FFmpeg left after a run.

### Live run hung for 13 minutes, then crashed sending the summary
- **Cause:** Docker Desktop's engine stopped responding (every call returned 500), so the stream stalled;
  OpenCV waited on it with no timeout, and the final POST to the API timed out and raised.
- **Fix:** live captures open with a 15 s open/read timeout; a failed summary POST is a warning (the local
  outputs are already written). Docker Desktop had to be restarted.

### A live run took 64 ms per frame instead of 21
- **Cause:** a local LLM (Ollama) held 6.3 GB of the 8 GB GPU memory at the same time.
- **Fix:** none needed in RetailFlow; measure with nothing else on the GPU (`nvidia-smi`, `ollama ps`).

### Watching the live preview made the pipeline skip more frames
- **Cause:** the heatmap overlay was blended in NumPy: 64 ms per full-HD frame, done 10 times a second
  while the page was open.
- **Fix:** `cv2.blendLinear` does the same blend in 1.4 ms (results within 1 grey level).

### Ctrl+C left the session marked "live" on the dashboard
- **Cause:** Ctrl+C kills the run before it sends its summary to the API.
- **Fix:** use the preview page's **Stop run** button (`--preview`): the pipeline leaves its loop and
  finishes normally.

### A stopped live session showed less video time than it had reached
- **Cause:** the API computed the length as analysed frames / fps; with skipped frames that is too short
  (46.3 s shown for about 67 s of video).
- **Fix:** the summary also stores `video_frames` (stream position) and the API uses it.

### Heads were not blurred in the live preview
- **Cause:** `demo.ps1` did not pass `--blur-heads`; the blur itself works in the preview path.
- **Fix:** the demo always passes it.

### 15 cameras in one process: the TensorRT engine refused a batch
- **Cause:** the engines are exported with batch 1 (dynamic image size only): "input size (15, 3, 544, 960)
  is outside the TensorRT optimization profile".
- **Fix:** frames go through the shared detector one at a time. A batch engine is a possible next step.

### 15 cameras analysed at only 1.6 per second each
- **Cause:** each replay thread turned all 30 frames per second into images (450 per second), and creating
  those arrays in Python competes with the main loop for the GIL; per frame 25 ms instead of 10 ms.
  Hardware decoding did not help (1.77 per second).
- **Fix:** only every 3rd frame becomes an image (`keep_every`); the others are decoded with `grab()` to keep
  time. Result: 4.66 analyses per camera per second.

### Floor map labels drawn huge
- **Cause:** the SVG worked in metres, so text was under 1 unit tall, and the browser's minimum font size
  applies in SVG units.
- **Fix:** draw at 20 units per metre.

### Docker Desktop's engine stopped answering (500 errors) more than once
- **Fix:** quit and restart Docker Desktop; the stack then starts normally.

### Batched detection was slower than one frame at a time (611 ms per batch)
- **Cause:** cameras were picked one by one when due, so every call had a different batch size, and each new
  size made the TensorRT backend re-allocate its buffers.
- **Fix:** with a batch engine, all cameras are read on one tick; 179 ms per 15-frame batch.

### Linking in a thread starved detection
- **Cause:** the linker's pandas work held the GIL; detection hit "NMS time limit exceeded" and fell behind real time.
- **Fix:** linking runs in its own process and receives only new tracks.

### "hardware accelerator failed to decode picture" (0x8007000e) with 15 cameras
- **Cause:** 15 GPU video decoders plus the batch engine ran out of video memory.
- **Fix:** CPU decoding is the default (`--hw-decode` to try the GPU); it was not slower.
