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
