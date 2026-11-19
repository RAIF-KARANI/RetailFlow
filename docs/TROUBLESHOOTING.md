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
