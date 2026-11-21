# 12 · Demo walkthrough

Three ways to see RetailFlow working, from fastest to fullest. Everything runs on public
research videos. All numbers below come from runs on the development laptop (RTX 4060 Laptop).

Before you start: the setup steps in the README, and at least CAVIAR downloaded
(`python tools/download_caviar.py`, 581 MB).

---

## A. One video → annotated output (2 minutes)

```powershell
python run.py --source data/caviar/TwoEnterShop1front/TwoEnterShop1front.mpg `
              --scene scenes/caviar_front.json --save-video --blur-heads --out outputs/demo/front
```

Look at `outputs/demo/front/`:
- `annotated.mp4`: boxes, track IDs, the **feet point** (red dot) used for counting, the counting
  line with its **band** (thin yellow lines) and IN arrow, zones with live occupancy, and
  **heads pixelated** (`--blur-heads`);
- `events.csv`: every crossing, zone visit (with dwell) and staff/static flag;
- `summary.json`: counts, customer counts (staff removed), timing;
- `heatmap.png`: where people spent time, over a people-free background.
- `transitions.csv`: how many people moved from each zone to each other zone.

| Demo clip | Predicted in / out | Ground truth in / out |
|---|---|---|
| `TwoEnterShop1front` (shop entrance) | 2 / 1 | 2 / 2 |
| `OneShopOneWait1cor` (corridor) | 0 / 2 | 1 / 3 |
| `MOT20-02`, first 30 s (crowd, 1920 px profile) | 22 / 4 | 18 / 0 |

These are single clips. The reliable numbers are the group results in `10_RESULTS.md`
(e.g. MOT20-02 is the known over-counting case, doc 10 §2.5).

![Shop front, annotated and blurred](img/demo_front.jpg)
![Corridor](img/demo_cor.jpg)
![MOT20-02 crowd](img/demo_mot20.jpg)

**Heatmap on the playing video:** add `--heatmap-overlay` (with `--save-video` or `--show`).
The frame is tinted where people have spent time so far, building up as the video plays. Only
visited areas are coloured. For the live overlay everyone counts at once; the saved
`heatmap.png` and the dashboard still drop staff and mannequins.

**Your own video:** first draw the line and zones on one of its frames with
`python tools/setup_scene.py --source my_video.mp4 --camera my_cam` (writes `scenes/my_cam.json`),
then run `python run.py --source my_video.mp4 --scene scenes/my_cam.json`. `--duration 600` stops after 10 minutes.

## B. Live camera → dashboard (5 minutes)

```powershell
.\demo.ps1                     # needs Docker Desktop running
```

1. Starts PostgreSQL, the API (:8000), the dashboard (:3001) and the RTSP server (MediaMTX).
2. Publishes the shop-front video as a live RTSP camera (looped, real-time speed).
3. Runs the pipeline on the stream for 90 s, sending events to the API.
4. Opens <http://localhost:3001>. Choose `caviar_front`, then the session marked **live**:
   the KPIs, crossings chart and zone occupancy refresh every 5 s.

**Watch the live video too:** `.\demo.ps1 -Preview` (or `run.py ... --preview`) opens
<http://127.0.0.1:8090>: the running pipeline's own annotated frames (boxes, feet points, counting line,
zones, live heatmap, IN/OUT so far) as an MJPEG stream, about 10 frames per second at 960 px. No second
model runs. Frames are only drawn while the page is open, and the server listens on this computer only
(127.0.0.1), because the preview shows people. `demo.ps1` always pixelates heads (`--blur-heads`); with
`run.py --preview` add `--blur-heads` yourself.
On the dashboard, a live session started with the preview shows a **▶ Watch live video** button that opens
this page. The page's **■ Stop run** button ends the run cleanly: outputs are written, the summary goes to
the API and the session is closed on the dashboard (Ctrl+C in the terminal also stops it, but leaves the
session marked live).
Measured cost, busiest 5 minutes of the MEVA hour streamed live (full HD, 30 fps): 2.3% of frames skipped
without the page open, 2.5% with it (2,206 preview frames sent); counts identical (7 in / 34 out).

At the end the script prints the real-time check (`keeps_up: true`, `realtime_factor: 1.0`).
`docker compose --profile streaming down` stops the services.

What to point out on the dashboard:
- **Customers in / out** exclude staff and static objects (mannequins). "Excluded" shows how many
  crossings were removed and why.
- Time axes are **video time**: the data has no real business hours, so none are invented.
- The dwell table explains the 2 s visit rule. The heatmap appears when a session ends.

## D. A whole store live (15 cameras, after Phase 8)

Needs the SmartSpaces scene (`python tools/download_smartspaces.py`, 3.03 GB) and the services running.

```powershell
python run_store.py --store scenes/stores/smartspaces_071.json --seconds 300 --api http://127.0.0.1:8000 --preview
```

`--preview` adds the live camera view (all cameras, store-wide numbers, heads pixelated, Stop button), linked
from the store page as **Watch live cameras**.

Open <http://localhost:3001>, section **Stores** → `smartspaces_071`: people now on the floor map with
their store-wide number and 10 s trails, a blueprint-style floor **heatmap** (toggle), visitors so far, the
people table and camera-to-camera moves. With `models/yolo26m_960_b16.engine` built, all cameras go through the
detector as one batch (doc 11 §8).
The store is synthetic, and live IDs are far from perfect (doc 11 §8): one person often gets several numbers.

## C. Reproduce the evaluation (hours of GPU time)

```powershell
python tools/export_models.py                                         # TensorRT engines (doc 08)
python benchmark.py --configs baseline m960_botsort_nogmc_trt --datasets caviar mot17
python benchmark.py --configs baseline m1920_botsort_nogmc_trt --datasets mot17 mot20
python evaluate.py  --configs n640_bytetrack final --group camera     # counting before / after
python evaluate_zones.py                                              # zones, dwell, heatmaps
python tools/sync_caviar_views.py; python evaluate_multicam.py        # multi-camera
python run_multi.py --n 4 --seconds 60 --cams data/caviar/TwoEnterShop1front/TwoEnterShop1front.mpg
pytest -q                                                             # 120 tests
```

`docs/09_EVALUATION_METHOD.md` defines every metric; `docs/10_RESULTS.md` has all the tables.

## Demo videos

`outputs/demo/video_front.mp4`, `video_cor.mp4` and `video_mot20.mp4` (H.264, 1–15 MB, heads
pixelated) were made with mode A. They are not committed: `outputs/` is git-ignored, and the
videos are derived from the datasets, so they are regenerated rather than redistributed.
