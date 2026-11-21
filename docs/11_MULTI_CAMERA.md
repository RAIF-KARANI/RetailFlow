# 11 · Multi-camera

Goal from the brief: multi-camera with the **two CAVIAR views**, and MMPTRACK if feasible.

Each of the 26 CAVIAR clips was filmed by two static cameras at the same time:
- **corridor** (`caviar_cor`): looks down the corridor; the Promod shop door is on its left wall, at the image edge;
- **front** (`caviar_front`): looks across the balcony at the Promod shop front.

The views overlap very little. Both see the shop door: the front camera head-on, the
corridor camera at the very edge of its image.

---

## 1. MMPTRACK: not used

From the official dataset page (iccv2021-mmp.github.io), checked in Phase 7:
- research use only; access requires **signing a user agreement and e-mailing it** to the
  organisers, then waiting for a manual reply. There is no direct download;
- no download size is published (about 5 h of training and 1.5 h of validation video);
- it does include a retail environment, calibrated cameras and cross-camera identity labels.

Signing terms and sending the request is the project owner's decision, and the turnaround
does not fit the time budget, so Phase 7 uses CAVIAR only. If MMPTRACK becomes available,
its cross-camera identities would allow proper cross-camera re-identification experiments (§6).

## 2. Putting both cameras on one clock

The views **do not start at the same moment**. The dataset says to align them with the
time code burned into every frame (`10:07:33.997 20-01-2004`).

**Reading the clock** (`retailflow/osd_clock.py`, no OCR library):
1. The clock is a fixed bitmap font, 8 px tall, at fixed positions. Each of the 9 time digits
   (HH MM SS mmm) has a slot; its pixels are binarised and trimmed to the lit columns.
2. Each slot is compared with **digit templates** (`retailflow/osd_digits.json`). The 31 template
   shapes were collected from the frames and **labelled once by eye**. The same digit renders
   1 px narrower or wider depending on its slot, so several shapes map to one digit.
3. A digit is accepted only if it is within 2 px of a template **and** at least 2 px better than
   any other digit. Otherwise the frame is "unreadable" rather than guessed.

**Fitting each clip's start** (`fit_start`): start = median over all frames of
(clock − frame index / 25). The recording sometimes repeats a frame and then skips one
(clock steps of 0 and 80 ms instead of 40 ms), and the median is not affected by that.

```powershell
python tools/sync_caviar_views.py      # -> outputs/caviar_sync.csv
```

**Result:** all **72,518 frames of all 52 videos readable**, **0 outliers** (every frame within
0.2 s of its clip's fit; 95th-percentile residual 0.04 s, i.e. one frame). The front camera
starts **3.04–3.12 s (76–78 frames) before the corridor camera** in 25 of 26 clips;
`TwoEnterShop3` is the exception (0.16 s), and its two views also differ in length (1,149 vs 1,075 frames).

**A bug this caught.** The first template set came from only 4 clips. A minute digit "0" in a
shape that wasn't in the set was read as "8" (3 px off, inside the first, looser tolerance),
moving four clips' start times by 5–8 minutes. Collecting shapes from all 52 clips (6 new ones)
and adding the margin rule fixed it. See TROUBLESHOOTING.

### Independent check of the sync (ground truth, not the clock)

If the sync is right, a person crossing the shop threshold in the front view should appear or
disappear in the corridor view **at the same moment**. For all 72 true threshold crossings, the
share with a corridor ground-truth track starting or ending within ±12 frames (±0.5 s):

| Shift added to the clock sync (frames) | −50 | −30 | −10 | **0** | +10 | +30 | +50 |
|---|---|---|---|---|---|---|---|
| Crossings with a corridor track start/end within ±0.5 s | 0.28 | 0.53 | 0.53 | **0.60** | 0.53 | 0.12 | 0.14 |

At random moments the same measure is 0.19. The peak is exactly at the clock-based sync, so
the clocks are right to well under a second. (A first idea, correlating the *number of people*
in each view over time, gave meaningless results: the views watch different areas.)

## 3. One door, two cameras

Where corridor tracks start/end at the moment of a front-view crossing shows where the shop
door is in the corridor image: a tight cluster at x ≈ 54–61 px, y ≈ 150–195 px (the left wall).
A **door line** was placed there (x = 70 px, IN = to the left). Three positions were tried, and
the one whose **ground-truth** counts agreed best with the front camera's ground truth was kept
(35 in / 41 out vs 31 / 41). That is mild tuning on the test camera's labels, and it only fixes
where the line is, not any result below.

The site file `scenes/sites/caviar_promod.json` records which line on which camera watches the
door. `retailflow/multicam.py` fuses the two cameras' crossings on the shared clock: same
direction and within `match_window_s` (1 s) = the same person, counted once. No appearance or
identity is used, only time and direction.

## 4. Results: does a second camera make door counts more reliable?

`python evaluate_multicam.py` (final system's tracks; only the time span both views cover;
truth = front-view ground-truth threshold crossings, as in Phases 2–6; 72 crossings, 26 clips):

| Policy | Predicted in / out | Abs. errors | WAPE | Exact clips |
|---|---|---|---|---|
| **Front camera only** | 29 / 37 | **14** | **19.4%** | **21 / 26** |
| Corridor camera only (door line) | 10 / 5 | 61 | 84.7% | 4 / 26 |
| Union, 1 s window | 38 / 40 | 22 | 30.6% | 14 / 26 |
| Union, 2 s window | 33 / 39 | 16 | 22.2% | 18 / 26 |
| Intersection, 1 s window | 1 / 2 | 69 | 95.8% | 5 / 26 |

**No: here the second camera makes counts worse.** The corridor camera sees the door at the
very edge of its image, against the wall:
- detection recall there is **0.73**, against **0.92** for the rest of the corridor view;
- people are cut off by the wall or the image border, so their boxes are clipped and the feet
  point never gets clearly past the door line. The corridor camera found 15 of the 72 crossings.

Its few extra events add more false counts (union) than missed people they recover.
**The front camera alone is the right choice for this door.** In a real store, a second camera
only helps if it sees the entrance properly, which is an installation decision, not a software one.

## 5. What multi-camera support exists now

- `retailflow/osd_clock.py`: per-frame recording time from a burned-in clock (tested on all 52 videos).
- `tools/sync_caviar_views.py`: per-clip offset between the views (`outputs/caviar_sync.csv`).
- `retailflow/multicam.py` + `scenes/sites/*.json`: site definition, two-camera fusion of one entrance.
- `evaluate_multicam.py`: single camera vs union vs intersection against ground truth.

**Not built (time):** a site page in the dashboard showing several cameras on the real recording
clock. The pieces exist: each session would store its clock start, and the API would merge sessions
of one site. This is listed as future work in doc 10.

## 7. One ID per person across 15 cameras (synthetic store, after Phase 8)

CAVIAR cannot score cross-camera identities (§6), and MMPTRACK's terms do not fit this project (§1).
NVIDIA's **PhysicalAI-SmartSpaces** (CC BY 4.0, `tools/download_data.md` §6) has a **synthetic** retail
store with 15 synchronised, calibrated, angled cameras and ground truth where every person keeps one ID
in all cameras. Scene 071: 31 people, about 13 minutes. Synthetic people, so this measures whether the
**method** works, not real-world accuracy.

**Method** (`retailflow/mtmc.py`; colour matching added in step 5):
1. Each camera is tracked on its own with the deployed detector and tracker (`tools/track_multicam.py`).
2. Every box's feet point is put on the floor with the camera's homography (median error against the
   dataset's floor positions: 0.17 m; boxes cut off at the bottom edge are skipped).
3. Tracks of different cameras that are on the floor at the same place at the same time
   (median distance ≤ 0.5 m) are the same person. Joined closest first; never two tracks of one camera
   at the same moment.
4. A path that ends, followed within 4 s by one that starts nearby at walking speed, is the same person
   (hand-off between views, or a broken track re-joined).
5. Optional colour matching (below) joins pieces that position alone cannot.

**Dev / test split in time:** minutes 0–2 were used for every choice (link distance 0.5 m and hand-off
gap 4 s, from a 9-setting sweep where IDF1 moved only between 0.417 and 0.445); minutes 2–7 were scored once.

**Metric:** global IDF1, the AI City Challenge multi-camera measure: boxes matched per camera and frame
(IoU ≥ 0.5), and a person only counts as identified if they keep **one** ID across all cameras.
`python evaluate_mtmc.py --tracks outputs/mtmc/scene_071/tracks_7min --from-s 120 [--appearance]`.

| | Dev, minutes 0–2 | **Test, minutes 2–7** |
|---|---|---|
| Detection recall / precision | 0.933 / 0.905 | 0.927 / 0.891 |
| Mean per-camera IDF1 (each camera alone) | 0.571 | 0.415 |
| Global IDF1, **no linking** (every camera track a separate person) | 0.169 | 0.110 |
| Global IDF1, **floor linking** | 0.445 | 0.281 |
| Global IDF1, **floor linking + colour** | 0.509 | **0.369** |
| Predicted people seen ≥ 10 s (true: 31), floor / + colour | 118 / 91 | 238 / 169 |

**What limits it, measured on dev:**
- Given the **ground-truth** per-camera tracks, the same linking reaches global IDF1 **0.825** and never
  puts two people under one ID; it still splits the 31 people into 71, mainly where a camera never sees
  the person's feet at the same moment as the others, or projects them metres off (legs hidden behind
  shelves, or camera_0641, whose projections are often far off).
- The real per-camera tracker breaks people into many pieces (971 camera tracks in 2 minutes for 31
  people) and swaps IDs, so linking starts from much worse input: 0.825 → 0.445.
- **Colour matching** (`retailflow/appearance.py`) was added for that: hue/saturation histograms of the
  torso and the legs, from up to 6 clear views per track (boxes hidden behind another person are skipped).
  Same person in another camera vs different people: median distance 0.32 vs 0.63, AUC 0.94 (dev,
  ground-truth boxes). It allows overlap links up to 2 m apart and hand-offs up to 30 s when colours
  match, and vetoes position links when colours clearly differ. Chosen on dev from 12 settings
  (match ≤ 0.30, veto > 0.6, gap ≤ 30 s; IDF1 0.463–0.509). **Privacy:** descriptors are computed
  during the run and kept in memory only, never written to disk (project rule for appearance
  features); they describe clothing colours, not faces.
- Still open: a steadier per-camera tracker (it breaks people into many pieces) and a learned
  re-identification model instead of colours (it would add a dependency). Unique-visitor counts are
  not reliable yet (169 people seen ≥ 10 s predicted vs 31 true over 5 minutes).

## 8. The store live: all cameras at once, on the dashboard

`python run_store.py --store scenes/stores/smartspaces_071.json --seconds 300 --api http://127.0.0.1:8000`,
then the dashboard's **Stores** section → `smartspaces_071`: people now on a floor map (store-wide
number, last 10 s), visitors so far, a people table (cameras in the order they saw each person) and
camera-to-camera moves, refreshed every 2 s.

**How it runs on one laptop:**
- Each camera's video is replayed at real-time speed in its own thread; the pipeline takes the newest
  frame (as from a live camera). Only every 3rd frame is turned into an image: converting all 450
  frames per second of 15 full-HD cameras starved the main loop (25 ms instead of 10 ms per analysed frame).
- One detector serves all cameras (15 TensorRT copies do not fit in 8 GB); each camera has its own
  tracker. The engine is built for batch 1, so frames go through one at a time.
- Every 3 s the tracks of the last 2 minutes are re-linked (floor + colour); older people keep their number.
- No RTSP: re-encoding 15 full-HD streams would need the CPU the pipeline needs.

**Measured** (15 cameras, target 5 analyses per camera per second, scored like §7 but only on the frames
each camera actually analysed):

| | Offline, every frame (§7) | Live, 2 min | Live, 7 min |
|---|---|---|---|
| Analyses per camera per second | 30 | 3.8 | 3.3 |
| Global IDF1, no linking | 0.169 (min 0–2) / 0.110 (min 2–7) | 0.118 (min 0–2) | 0.060 (min 2–7) |
| Global IDF1, linked (floor + colour) | 0.509 / 0.369 | 0.306 | 0.197 (min 2–7) |
| Re-link time, mean / max | – | 1.1 s / 2.2 s | 1.8 s / 5.5 s |

**Batch engine and a linking process** (added afterwards):
- `models/yolo26m_960_b16.engine` (`python tools/export_models.py --from-fp16-onnx models/yolo26m.fp16.onnx
  --imgsz 960 --opt-hw 544 960 --max-batch 16`, 300 s to build) takes up to 16 frames per call. On 15
  full-HD frames: 232 ms one at a time vs 171 ms as one batch (15.4 → 11.4 ms per frame), same 68 people found.
  All cameras are processed on one tick so the batch size stays the same: picking cameras one by one gave a
  different batch size every call and 611 ms per batch.
- Linking runs in a **separate process** that receives only new tracks every 3 s. A thread was tried first;
  its pandas work held the GIL and starved detection (NMS time-outs, video falling behind).
- Videos are decoded on the CPU: 15 GPU decoders next to the batch engine ran out of video memory.

| 15 cameras, 7 minutes live | Before | **Batch + linking process** |
|---|---|---|
| Analyses per camera per second (target 5) | 3.3 | **4.88** |
| Detector time per 15-frame batch | – | 179 ms |
| Global IDF1, minutes 0–2 | 0.306 (2-min run) | 0.413 |
| Global IDF1, minutes 2–7 | 0.197 | **0.294** (offline: 0.369) |

**Floor heatmap** (store page, "Heatmap"): person-seconds per 0.5 m cell, counted once per store-wide
person per second (the median of the cameras that see their feet), drawn as a blueprint. The shape shows
where people walk; the totals are inflated where one person has several numbers (298 person-minutes in 7
minutes, while 31 people can give at most 217).

**Live camera view** (`run_store.py --preview`): http://127.0.0.1:8090 shows all cameras as a mosaic (or one camera
at full size, picked in the page), each box labelled with the person's **store-wide number** and coloured by it,
so the same person can be followed from camera to camera; "?" marks a track the linker has not numbered yet
(it is numbered within one 3 s re-link). The numbers come back from the linking process. Heads are pixelated
(`--no-blur` to switch off), frames are drawn only while the page is open, and the page has the Stop button;
the store page on the dashboard shows "Watch live cameras" while such a run is live.

**Floor plan and store zones.** The store has no plan, so `tools/build_floor_plan.py` makes one from the cameras:
each camera's people-free background (median of 15 frames) is projected onto the floor with its homography, and
the 15 views are stitched, each floor spot taken from the nearest camera. The result shows the floor as seen from
above (checkouts, shelf bases, freezers, the storage room's blue floor); things standing on the floor are smeared
away from the camera, and floor no camera sees stays black. A blueprint version keeps its outlines as light lines.
The dashboard draws either under the map (Blueprint / Floor photo / Grid).
Nine store zones were drawn on it in floor metres (`scenes/stores/smartspaces_071.json`; editor:
`tools/setup_store_zones.py`; `tools/register_store.py` sends zones and floor images to the API). Per zone, from
each store-wide person's floor position once per second and the same 2 s rule as camera zones: people now, visits,
mean/median stay, and **zone-to-zone moves across the whole store**.

**Scored against ground truth** (`python evaluate_store_zones.py`): the dataset gives every person's floor position,
so the true people (one ID across cameras, median position once per second) go through exactly the same zone code
(`store_live.store_zone_analysis`). Zones were drawn from the floor photo, not tuned on these numbers. Test minutes 2–7:

| | Offline (every frame, floor + colour) | Live run (15 cameras, 4.9 analyses/s) |
|---|---|---|
| People in a zone, error per second (zones hold 2.9 on average) | **0.85** | 1.40 |
| Time in zones (person-seconds), WAPE | **24.8%** | 42.9% |
| Zone-to-zone moves (382 true), WAPE | **30.4%** (462 found) | 42.7% (503 found) |
| Visits (480 true) | 940 | 1,091 |

- Occupancy and time in zone are not hurt by one person getting several numbers one after another, and they are
  the most reliable store figures; the storage room (0.2% time error offline) and snack aisles are close, the
  entrance and freezers are too high. Where they are too high, one person is counted twice at the same moment:
  two cameras' views of them were not joined.
- Visits are about twice the truth, with shorter stays: a split person turns one visit into several. Visit counts
  and stay times per zone should not be quoted until the cross-camera IDs improve.
- The move matrix is closer than visits (30% offline) because a split rarely changes which zone comes next.

**What this meant before the batch engine:** the live store view worked end to end, but at 3–4 analyses per second per camera the
per-camera tracks break more, so store-wide IDs are clearly worse than offline, and "visitors so far"
over-counts heavily (939 store-wide numbers seen ≥ 2 s in 7 minutes for 31 people). Re-linking also
still slows down as a run gets longer, which lowers the analysis rate. Next steps: a batch-enabled
engine (one detector call for all cameras), re-linking off the main loop, and the per-camera tracker.

## 6. Limits

- **No cross-camera identities on real footage.** (A synthetic store is scored in §7.) CAVIAR numbers people separately in each view (e.g. `TwoEnterShop1`:
  12 IDs in the corridor view, 10 in the front view), so cross-camera re-identification (the same
  person in both views by appearance) **cannot be scored** here. It was not built, rather than built
  untested. MMPTRACK (§1) would allow it.
- The door-line choice used ground-truth agreement on all 26 clips (§3).
- Two cameras with little overlap is a hard setup for fusion. The negative result is specific to this
  geometry, not a claim that fusion never helps.
