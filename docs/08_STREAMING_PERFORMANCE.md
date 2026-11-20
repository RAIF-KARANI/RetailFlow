# 08 · Streaming and performance

Goal: process **live CCTV streams** in real time, several at once on one laptop GPU,
**without losing the accuracy** measured in Phases 3–4.

Machine: RTX 4060 Laptop (8 GB), Windows 11, performance power mode (GPU clock checked
before every speed run, see TROUBLESHOOTING).

---

## 1. Inference backends

`python tools/export_models.py` exports YOLO26m to:
- **ONNX** (FP32), run by ONNX Runtime's CUDA provider;
- **TensorRT FP16** engine (`models/yolo26m_<imgsz>.engine`).

Both use **dynamic input shapes**. A fixed-shape engine forces every frame into an
imgsz × imgsz square, so a 16:9 camera would do nearly twice the work of PyTorch's
rectangular letterbox (1920 × 1088 instead of 1920 × 1920). Engines are built for this
GPU and TensorRT version, so they are rebuilt on each machine and never committed.

Notes from the export (details in TROUBLESHOOTING):
- The 1920 engine could not be built through Ultralytics: its FP16 conversion step ran out
  of memory on the CPU. It was built directly with the TensorRT API from the
  (size-independent) FP16 ONNX made during the 960 export.
- The exports use YOLO26's standard detection head plus NMS (`end2end: False` in the engine
  metadata), while the `.pt` model runs the NMS-free head. Together with FP16, that is why
  accuracy was re-checked on the whole dataset (§3) instead of assumed.
- Ultralytics silently pip-installed packages and downgraded ONNX Runtime during export,
  which made ONNX Runtime fall back to the CPU. This was fixed, and auto-install is now
  disabled (`YOLO_AUTOINSTALL=false`). Pinned list: `requirements-export.txt`.

## 2. Speed per backend

`python tools/benchmark_speed.py` → `outputs/speed.csv`.
**Model** = preprocess + inference + postprocess on frames already in memory (200 frames
after 20 warm-up). **Pipeline** = `avg_proc_ms` over the first 300 frames (detect + track +
count + zones + heatmap), measured with the Phase 3 tracker (BoT-SORT with GMC).

| Camera (input) | Backend | Inference ms | Model ms | Pipeline ms |
|---|---|---|---|---|
| CAVIAR front 384×288 @ 960 | PyTorch FP32 | 17.7 | 22.0 | 26.3 |
| | PyTorch FP16 | 10.6 | 15.0 | – |
| | ONNX Runtime FP32 (CUDA) | 22.9 | 27.2 | 32.1 |
| | **TensorRT FP16** | **5.2** | **9.1** | **13.6** |
| MOT17-09 1080p @ 960 | PyTorch FP32 | 12.8 | 18.5 | 35.9 |
| | PyTorch FP16 | 9.5 | 13.1 | – |
| | ONNX Runtime FP32 (CUDA) | 17.1 | 21.2 | 39.1 |
| | **TensorRT FP16** | **4.9** | **12.3** | 28.1 |
| MOT20-02 1080p @ 1920 (crowd) | PyTorch FP32 | 56.0 | 61.7 | 77.7 |
| | PyTorch FP16 | 29.0 | 33.9 | – |
| | ONNX Runtime FP32 (CUDA) | 68.6 | 73.9 | 94.8 |
| | **TensorRT FP16** | **15.9** | **33.5** | **37.4** |

(The pipeline has no FP16 switch for `.pt` files, so PyTorch FP16 is model-only.)

- TensorRT FP16 makes **inference 2.6–3.5× faster**. FP16 alone (PyTorch FP16) gives 1.3–1.9×;
  the rest comes from TensorRT's kernel fusion and tuning.
- ONNX Runtime FP32 is **slower than PyTorch** here: no FP16, and generic CUDA kernels.
- On 1080p cameras, the pipeline time is **much larger than the model time**
  (MOT17-09: 28.1 ms pipeline vs 12.3 ms model with TensorRT). §4 finds out why.

## 3. Does TensorRT FP16 change the results?

All 62 sequences were re-run with the engines (`benchmark.py --configs m960_botsort_trt
m1920_botsort_trt`), then scored exactly as in Phase 3:

| | Dev errors (138 crossings) | Test errors (1,138) | Dev IDF1 | Dev ms/frame |
|---|---|---|---|---|
| PyTorch FP32 (Phase 3 final) | 32 | 342 | 0.718 | 28.8 |
| TensorRT FP16 | 32 | 335 | 0.718 | **16.0** |

**Same accuracy, 1.8× faster end to end.** Per camera, the counts differ by 0–5 errors in
both directions (e.g. MOT20-05: 184 → 179, MOT20-03: 81 → 82). That is detection noise, not
a systematic loss.

## 4. The hidden bottleneck: camera-motion compensation

BoT-SORT's **GMC** (global motion compensation) tracks background points between frames
(sparse optical flow, on the CPU) to correct the tracker's motion predictions when the
**camera** moves. Every camera here is static (Phase 1: < 1 px per frame), so GMC only ever
measures "no motion", but it pays the full cost on every 1080p frame.

300-frame pipeline, TensorRT, with vs without GMC (`trackers/botsort_nogmc.yaml`, which is
the default BoT-SORT with only `gmc_method: none`):

| Camera | With GMC | Without GMC |
|---|---|---|
| MOT17-09 1080p | 45.2 ms | **14.1 ms** |
| MOT20-02 crowd 1920 | 40.8 ms | **26.0 ms** |
| CAVIAR front 384×288 | 13.7 ms | 11.3 ms |

(These 300-frame timings vary by a few ms between runs. MOT17-09 with GMC measured 28–45 ms.)

Turning GMC off changes the tracker, so it went through the same dev check as every
other choice. All 62 sequences, scored as in Phase 3:

| Config | Dev errors | Test errors | Dev IDF1 | Dev ms/frame | MOT20 ms/frame |
|---|---|---|---|---|---|
| PyTorch, BoT-SORT with GMC (Phase 3) | 32 | 342 | 0.718 | 28.8 | 100.0 |
| TensorRT, BoT-SORT with GMC | 32 | 335 | 0.718 | 16.0 | 46.3 |
| **TensorRT, BoT-SORT without GMC (final)** | **30** | **336** | **0.719** | **12.3** | **37.8** |

**Adopted:** both profiles now use `botsort_nogmc`, and `run.py` / `run_multi.py` use the
TensorRT engine when it has been built (`--backend auto`, the default; `pt` / `trt` to force).
`python evaluate.py --configs final_phase3 final_trt_gmc final` reproduces the table.

**Keep GMC on for PTZ cameras or cameras on an unstable mount.** The scene file decides
nothing here yet; that would be a per-camera tracker setting.

## 5. Live streams (RTSP)

```powershell
docker compose --profile streaming up -d mediamtx         # RTSP server, rtsp://127.0.0.1:8554
python tools/simulate_rtsp.py front=data/caviar/TwoEnterShop1front/TwoEnterShop1front.mpg
python run.py --source rtsp://127.0.0.1:8554/front --scene scenes/caviar_front.json --api http://127.0.0.1:8000
```

`tools/simulate_rtsp.py` turns dataset videos into "IP cameras": FFmpeg loops each file at
real-time speed (`-re -stream_loop -1`) and publishes H.264 (low-latency settings, 1 keyframe/s,
no B-frames) to MediaMTX over TCP. The pipeline reads the RTSP URL like any other source.

**Real-time check.** For a live source, every frame gets
`lag = wall-clock time since the stream opened − frame index / fps`.
A pipeline that keeps up has a flat lag. One that is too slow falls further behind every
second (frames queue up), so lag grows linearly. `summary.json["live"]` reports the lag
growth after a 5 s warm-up, and `realtime_factor` = video seconds processed per wall second
(1.0 = real time).

## 6. Several streams at once (`run_multi.py`)

`python run_multi.py --cams <video> --n 4 --seconds 60` publishes N looped copies, starts
**one pipeline process per stream** (each with its own model and tracker, all sharing the
GPU), runs for 60 s, and collects each stream's summary plus GPU utilisation and memory
(`nvidia-smi` once per second).

A stream counts as **real time** only if both hold:
1. its lag does not grow (least-squares slope over the second half of the run < 0.02 s per s);
2. MediaMTX never had to drop data for it (`reader is too slow, discarding`, read from its log).

The second check matters: when the server drops frames, the frame counter falls behind the
real video and the lag looks *better* than it is (TROUBLESHOOTING, Phase 6).

Final system (TensorRT FP16, BoT-SORT without GMC), 60 s per run, each stream a looped copy
of the same video. `ms/frame` = `avg_proc_ms` per stream under load; the budget is 1000 / fps.

| Camera (input, fps) | Streams | ms/frame per stream (budget) | GPU busy | GPU memory | Lag slope | Server drops | **Real time** |
|---|---|---|---|---|---|---|---|
| CAVIAR front (384×288, 25) | 1 | 18.1 (40) | 40% | 3.3 GB | 0.000 | 0 | **yes** |
| | 2 | 22.8 (40) | 43% | 3.5 GB | 0.000 | 0 | **yes** |
| | 4 | 29.7–34.6 (40) | 80% | 6.2 GB | 0.000 | 0 | **yes** |
| MOT17-09 (1080p, 30) | 1 | 17.7 (33) | 35% | 3.0 GB | 0.000 | 0 | **yes** |
| | 2 | 19.3–19.4 (33) | 40% | 3.3 GB | ≤ 0.003 | 0 | **yes** |
| | 4 | 32.9–33.0 (33) | 62% | 6.2 GB | 0.20 | yes, all 4 | **no** (0.83× real time) |
| MOT20-02 crowd (1080p @ 1920 px, 25) | 1 | 27.4 (40) | 45% | 2.6 GB | 0.003 | 0 | **yes** |
| | 2 | 47.9 (40) | 74% | 3.3 GB | 0.31 | yes, both | **no** (0.76× real time) |
| MOT17-09 (1080p, 30), **PyTorch** | 1 | 20.5 (33) | 44% | 2.4 GB | 0.000 | 0 | **yes** |

Raw results: `outputs/multi/<run>/multi_summary.json`. Lag at start-up (the first frames while
everything settles) was at most 0.3–1.1 s in the runs that kept up.

**Capacity of this laptop (RTX 4060 Laptop), measured:**
- **4 typical small retail cameras** (CAVIAR, 384×288 @ 25 fps) in real time, with headroom left in
  each frame's budget (30–35 of 40 ms).
- **2 full-HD cameras** at 30 fps in real time. At 4, each stream needs 33 ms for a 33 ms budget and
  falls behind. The GPU is only 62% busy then, so the limit is the **CPU**: four 1080p H.264
  decodes, plus the simulator's four encodes on the same CPU, plus pre-processing.
- **1 crowd camera** at 1920 px input in real time; 2 do not fit.
- With GMC off, even PyTorch handles one 1080p stream in real time (20.5 ms). TensorRT is what
  creates room for more streams.
- A 50 s live run from RTSP into the API was shown on the dashboard as a **live** session updating
  every 5 s (2 in, 2 out, real-time factor 1.0).

**How to scale further** (not done here): decode on the GPU (NVDEC) instead of the CPU, batch frames
from several cameras into one inference call, use YOLO26s on quiet cameras (Phase 3 grid, ByteTrack stage:
38 dev errors vs 34 for m at 960 px), or lower the crowd input from 1920 to 1280 px.

## 7. Run it

```powershell
pip install -r requirements-export.txt
python tools/export_models.py                                        # 960 engine + ONNX (~10 min)
python tools/export_models.py --imgsz 1920 --from-fp16-onnx models/yolo26m.fp16.onnx
python tools/benchmark_speed.py                                      # §2
python benchmark.py --configs m960_botsort_nogmc_trt m1920_botsort_nogmc_trt   # §3-4 accuracy
python evaluate.py --configs final_phase3 final_trt_gmc final
docker compose --profile streaming up -d mediamtx
python run_multi.py --n 4 --seconds 60 --cams data/caviar/TwoEnterShop1front/TwoEnterShop1front.mpg  # §6
```

(`models/yolo26m.fp16.onnx` is written by the first export; it is moved there from the repo root.)

## 8. Limits of these measurements

- The simulated cameras' H.264 **encoding also runs on this laptop's CPU** (FFmpeg); real
  cameras encode on the camera. Multi-stream results are therefore slightly pessimistic.
- Lag is measured from when the pipeline opens the stream, so the constant network/decoder
  delay is not included. Only whether the pipeline keeps up is measured.
- Speed depends on the laptop's power mode and temperature (see TROUBLESHOOTING, Phase 1).

## 9. One hour of real footage, live (MEVA)

After Phase 8 the system ran for a full hour on real CCTV-style footage: MEVA camera G420, an indoor
school entrance lobby filmed from above at an angle, 2018-03-07 10:05–11:05 (1920×1072, 30 fps; licence
and download in `tools/download_data.md` §5). The video was published as a live RTSP camera and processed
with `--api` into the dashboard:

```powershell
.\demo.ps1 -Seconds 3600 -Video data\meva\G420_2018-03-07_10-05_1h.mp4 -Scene scenes\meva_G420.json -Stream meva_g420
```

| Measure | Result |
|---|---|
| Frames processed | 108,000 (3,600 s of video), no crash, no restart |
| Counts (IN = into the building) | 89 in / 122 out (no staff or static tracks flagged) |
| Zone visits | 336: `stairs_top` 233 (median dwell 4.9 s), `lobby_right` 68 (4.2 s), `doormat` 35 (3.3 s) |
| Track IDs | 836 |
| Processing time | 22.0 ms per frame on average (budget 33.3 ms) |
| Real-time check | **did not keep up** over the whole hour: `realtime_factor` 0.962, lag at the end 70.9 s |
| API / dashboard | same counts and zones from `/api/sessions/<id>/summary` |

**Where it fell behind.** Lag stayed flat at 2.4 s for the first 20 minutes, then rose in two steps,
to about 22 s around minutes 24–28 and to about 71 s around minutes 40–48, and stayed flat again
after each step. MediaMTX logged "reader is too slow" in exactly those minutes (about 52,000
discarded RTP packets after the first minute), and these are the busiest minutes of the hour (up to 56
zone/line events per minute, against fewer than 15 in most minutes). Run offline on a 5-minute busy
stretch (minutes 40–45), the pipeline needs 28.1 ms per frame for processing and reaches **27.3 fps**
including decoding, below the camera's 30 fps. On a quiet stretch (minutes 10–15) it needs 23.3 ms
and reaches 31.9 fps: above 30, but with little margin. This camera costs more per frame than
MOT17-09 in §6 (17.7 ms), so the "2 full-HD cameras" capacity there does not hold for it.

So: **one full-HD camera keeps up on this laptop in normal traffic, only just, and not in rush minutes.** Frames
that arrive while it is behind are dropped by the RTSP server, so crossings in those minutes can be
missed (about 71 s of the hour, 2%). MEVA has no in/out ground truth, so the counts cannot be scored.

### 9.1 The fix: keep only the newest frame

Live streams are now read by a background thread (`retailflow/sources.py`, `LatestFrameReader`) that
decodes **every** frame and hands the pipeline only the newest one. When a frame takes longer than its
33 ms, the next frames are skipped **whole**, instead of the RTSP server dropping packets mid-frame.
`Frame.index` stays the frame's true position in the stream, so video time is unchanged, and zone
debounce, staff time and the heatmap give each analysed frame the weight of the frames skipped before it.
The real-time check now also reports `received`, `skipped` and `processed_frac`, and only says
`keeps_up` when at most 1% of frames were skipped (skipping keeps the lag flat, so lag alone would hide it).
A stream that stops sending frames ends the run after at most 15 s (it used to block for good).

The same hour, live with `--api`, before and after (the "after" run had nothing else on the GPU or in
the background; see §9.2):

| | Before (packet drops) | **After (latest frame)** |
|---|---|---|
| Lag behind the stream, max / at the end | 77.9 s / 70.9 s | **3.1 s / −0.7 s** |
| H.264 decode errors in the log | 249 | **0** |
| MediaMTX "reader is too slow" warnings | 262 | **0** |
| Frames analysed | 108,000 received, about 71 s of them lost in dropped packets | **105,253 of 108,000 (97.5%)**, 2,747 skipped whole |
| `keeps_up` (≤ 1% skipped, no lag growth) | no | no (2.5% skipped) |
| Processing time per frame | 22.0 ms | 21.2 ms |
| In / out | 89 / 122 | 87 / 123 |
| Zone visits (stairs / lobby right / doormat) | 336 (233 / 68 / 35) | 395 (280 / 84 / 31) |
| Track IDs | 836 | 733 |

**Result:** one full-HD camera now runs a whole hour live without falling behind; over the hour
2.5% of the frames are skipped instead of whole seconds being lost. By the 1% rule the laptop still has no
headroom for this camera at peak. The counts barely move. There are about 18% more zone visits and 12% fewer
track IDs, which fits frames no longer arriving corrupted (a broken track splits into short pieces that
never reach the 2 s visit threshold), but MEVA has no ground truth to check which run is closer.

Still open: decode on the GPU (NVDEC), or a smaller input size or YOLO26s on this camera, would create
headroom for the peak.

### 9.2 What spoiled two of the runs

- **Leftover simulated cameras.** `demo.ps1` stopped the simulator but not its FFmpeg encoder: with a
  venv, `python.exe` is a launcher that starts the real interpreter, so FFmpeg is a grandchild. Four
  encoders (two of them full HD) kept running and loaded the CPU during the second hour (6.9% skipped,
  27.2 ms per frame). Fixed: the script stops the whole process tree (`taskkill /T`).
- **Another program on the GPU.** A local LLM held 6.3 GB of the 8 GB GPU memory during the third run,
  which took 64 ms per frame. Docker Desktop's engine then stopped responding and the run ended after
  15.6 minutes. Measure on a GPU with nothing else loaded (`nvidia-smi` memory use).
