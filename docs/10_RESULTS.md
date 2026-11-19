# 10 · Results

All numbers come from runs on the development machine (RTX 4060 Laptop) and can
be reproduced with the commands shown. Raw per-sequence rows: `outputs/results.csv`.
Method and metric definitions: [`09_EVALUATION_METHOD.md`](09_EVALUATION_METHOD.md).

---

## 1. Baseline ("before"), Phase 2

**Detector/tracker:** YOLO26n, ByteTrack, `imgsz=640`, `conf=0.1` (config `baseline`).
**Data:** 55 sequences (26 CAVIAR corridor, 26 CAVIAR front, MOT17-02/04/09), 74,693 frames, 210 true crossings.

```powershell
python benchmark.py --configs baseline
python evaluate.py --configs baseline --group camera
```

### 1.1 Counting variants on the same tracks

All rows use **identical tracks**, so differences are caused only by the counting logic.

| Variant | Point | Band | WAPE (all) | Abs. errors | Over / under | Exact sequences |
|---|---|---|---|---|---|---|
| `centre_simple` (classic baseline) | centre | no | **92.4%** | 194 | 129 / 65 | 9 / 55 |
| `feet_simple` | feet | no | 64.3% | 135 | | 19 / 55 |
| `feet_band` | feet | 5 px / 15 px | **27.6%** | 58 | 16 / 42 | 27 / 55 |
| `feet_band_age` (min age 10) | feet | yes | 27.6% | 58 | | 27 / 55 |
| `feet_band_age_once` | feet | yes | 26.7% | 56 | | 27 / 55 |

### 1.2 By camera (WAPE)

| Camera | True crossings | `centre_simple` | `feet_simple` | `feet_band` | `feet_band_age_once` | Mean IDF1 |
|---|---|---|---|---|---|---|
| caviar_cor (26 clips) | 95 | 83.2% | 97.9% | 34.7% | 31.6% | 0.720 |
| caviar_front (26 clips) | 72 | 133.3% | 33.3% | 20.8% | 22.2% | 0.490 |
| MOT17-02 | 18 | 44.4% | 44.4% | 44.4% | 44.4% | 0.320 |
| MOT17-04 | 13 | 0.0% | 61.5% | 0.0% | 0.0% | 0.351 |
| MOT17-09 | 12 | 91.7% | 16.7% | 16.7% | 16.7% | 0.519 |

### 1.3 By density (WAPE)

| Density | Sequences | True crossings | `centre_simple` | `feet_simple` | `feet_band` | `feet_band_age_once` |
|---|---|---|---|---|---|---|
| quiet (< 2.5 people/frame) | 31 | 66 | 163.6% | 66.7% | 24.2% | 25.8% |
| medium (2.5 – 15) | 22 | 113 | 69.0% | 66.4% | 30.1% | 27.4% |
| busy (≥ 15) | 2 | 31 | 25.8% | 51.6% | 25.8% | 25.8% |

The busy group has only 2 sequences (MOT17-02, -04); MOT20 is added in Phase 3.

### 1.4 What the baseline shows

1. **The centre point is wrong for angled cameras.** In the CAVIAR front view, people
   walking *past* the shop have their feet at y≈150–190, but their box centre sits at
   y≈120–150, right on the shop-threshold line (y=142). The centre point "crosses"
   while the person never approaches the door. `centre_simple` over-counts the shop
   entrance by **133%**; using the feet point alone brings it to 33%.
2. **The band removes jitter double counts.** On the corridor camera, the feet point
   with a simple line is *worse* (97.9%) because a feet point dithering on the line
   flips side repeatedly. Adding the ±5 px band brings it to 34.7%. Over-counts across
   all sequences drop from 129 (`centre_simple`) to 16 (`feet_band`).
3. **What remains is mostly missed people (detection/tracking), not counting logic.**
   With `feet_band`, 42 of 58 errors are **under-counts**. MOT17-02 has detection
   **recall 0.24** at `imgsz=640` (76% of labelled boxes missed), so it loses 8 of 18
   crossings with every counting variant. Some CAVIAR front clips have recall ≈0.2.
   ID switches are high in the busier CAVIAR clips (30–48 per clip).
   → Phase 3 targets these with larger input sizes, larger models and BoT-SORT + ReID.
4. **Min age and once-per-direction barely matter yet** (27.6% → 26.7%). They target
   ID-switch double counts, which become more visible once recall improves.
5. **MOT17-04 at 0% error should not be over-read.** It is one sequence with 13
   crossings, and its tracking is weak (IDF1 0.351; mean best IoU vs GT ~0.27, see 09 §7).
   A correct total on so few crossings is not evidence that busy-scene counting is solved.

### 1.5 Speed during the baseline

Not reported as a result. The GPU was power-limited and its clock changed during the
run (corridor clips averaged 29.0 ms/frame, front clips 14.5 ms/frame at the same
384x288 resolution). Speed is measured properly in Phase 6.

---

## 2. Reliable counting: final configuration (Phase 3)

**Rule:** every choice was made on the **dev split** (CAVIAR corridor, MOT17-02/04/09).
The **test split** (CAVIAR front, MOT20) was only scored afterwards. The one exception,
the crowd profile, is labelled as such in §2.3.

### 2.1 Detector/tracker grid (dev only)

`python benchmark.py --plan onefactor --datasets caviar mot17` changes one factor
per stage and keeps the winner. Score = dev counting errors with `feet_band`. Configs
within 1 error of the best count as tied, and the faster one wins. Selections are saved in
`outputs/grid_selection.json`.

| Stage | Config | Dev WAPE | Dev errors | Dev IDF1 | ms/frame | |
|---|---|---|---|---|---|---|
| 1 image size | n640 ByteTrack (baseline) | 31.2% | 43 | 0.687 | 11.9 | |
| | **n960 ByteTrack** | 26.1% | 36 | 0.709 | 12.0 | chosen |
| | n1280 ByteTrack | 26.8% | 37 | 0.672 | 13.7 | tied, slower |
| 2 model | n960 ByteTrack | 26.1% | 36 | 0.709 | 12.0 | |
| | s960 ByteTrack | 27.5% | 38 | 0.725 | 12.2 | |
| | **m960 ByteTrack** | 24.6% | 34 | 0.714 | 21.9 | chosen |
| 3 tracker | m960 ByteTrack | 24.6% | 34 | 0.714 | 21.9 | |
| | **m960 BoT-SORT** | **23.2%** | **32** | 0.716 | 25.2 | **chosen** |
| | m960 BoT-SORT + ReID | 23.9% | 33 | 0.720 | 26.3 | tied, slower |

ms/frame here is the mean `avg_proc_ms` over dev sequences (detect + track + count),
with the GPU in performance mode. Proper speed measurements come in Phase 6.

Observations:
- The fresh n640 run reproduced the Phase 2 baseline exactly (43 dev errors).
- 640 → 960 is the biggest single gain (−7 errors), and costs almost nothing.
- ReID raises IDF1 slightly (0.716 → 0.720) but does not change the counts.
- The differences between the stage-2 and stage-3 rows are 1–4 errors on 138 crossings.
  The direction is consistent, but this is a small dev set.

### 2.2 Counting settings on the final tracks (dev only)

`python evaluate.py --configs m960_botsort --sweep` tests 32 combinations of band scale
(0.5/1/2/3×), min track age (0/5/10/20) and once-per-direction. On dev, the **default
settings are tied for best**: 1× band, min age 0, no once-per-direction, 32 errors (the same
as min age 5). The best filtered variant (`0.5× band, once per direction`) makes 33. So
the final counting settings are **feet point + 1× band, no extra filters**. On the
stronger tracks, the filters that helped the baseline (§5.2 in doc 04) no longer pay off.

### 2.3 Crowd profile (decided after a test-set diagnosis)

The grid winner (m960) failed on the two densest MOT20 videos (§2.4). Looking at
raw detections showed why: on MOT20-05 the detector finds about 34 boxes with
conf ≥ 0.1 per frame at 960 px, for 195 labelled people. At the full 1920 px it finds
about 128 (sample of 6 frames, `yolo26m`).

That diagnosis used test data, so the fix was kept **as generic as possible**:
- A second **profile** (`retailflow/profiles.py`): same model and tracker, run on the
  full-resolution frame (`imgsz=1920`). No other setting was changed or tuned.
- **Which cameras use it** follows a rule fixed in Phase 2: mean people per frame ≥ 15
  (the existing "busy" threshold). It is stored as `"profile": "crowd"` in the scene file.
  In a real installation the installer sets it from how busy the camera usually is.
- The rule also selects two **dev** cameras (MOT17-02, -04). The profile was checked on
  those first:

| Dev crowd camera | Errors m960 → m1920 | Recall | IDF1 |
|---|---|---|---|
| MOT17-02 | 8 → 8 | 0.416 → 0.436 | 0.402 → 0.398 |
| MOT17-04 | 0 → 0 | 0.533 → 0.701 | 0.560 → 0.616 |

On dev it does no harm to counts and clearly improves detection. The cost is speed:
about 85 ms/frame instead of about 32 ms/frame on these 1080p scenes.

**The "final" system** = `standard` profile (m960 BoT-SORT) on CAVIAR and MOT17-09,
plus the `crowd` profile (m1920 BoT-SORT) on MOT17-02/04 and all MOT20 cameras.
`python evaluate.py --configs n640_bytetrack final` compares it with the baseline.

### 2.4 Before / after (feet point + band)

```powershell
python benchmark.py --configs n640_bytetrack m960_botsort m1920_botsort --datasets caviar mot17 mot20
python evaluate.py --configs n640_bytetrack m960_botsort m1920_botsort final --variants centre_simple feet_band
```

| Camera | Split | Crossings | Classic baseline (centre, n640) | Baseline feet+band (n640) | m960 everywhere | **Final** |
|---|---|---|---|---|---|---|
| caviar_cor (26 clips) | dev | 95 | 83.2% | 34.7% (33) | 23.2% (22) | **23.2% (22)** |
| MOT17-02 | dev | 18 | 44.4% | 44.4% (8) | 44.4% (8) | 44.4% (8) |
| MOT17-04 | dev | 13 | 0.0% | 0.0% (0) | 0.0% (0) | 0.0% (0) |
| MOT17-09 | dev | 12 | 91.7% | 16.7% (2) | 16.7% (2) | 16.7% (2) |
| **Dev total** | | **138** | | **31.2% (43)** | **23.2% (32)** | **23.2% (32)** |
| caviar_front (26 clips) | test | 72 | 133.3% | 20.8% (15) | 20.8% (15) | **20.8% (15)** |
| MOT20-01 | test | 29 | 55.2% | 79.3% (23) | 65.5% (19) | **51.7% (15)** |
| MOT20-02 | test | 114 | 11.4% | 29.8% (34) | **7.0% (8)** | 41.2% (47) |
| MOT20-03 | test | 385 | 90.1% | 91.9% (354) | 68.3% (263) | **21.0% (81)** |
| MOT20-05 (extreme) | test | 538 | 91.6% | 93.5% (503) | 93.1% (501) | **34.2% (184)** |
| **MOT20 total** | test | **1066** | 81.5% | **85.7% (914)** | 74.2% (791) | **30.7% (327)** |

WAPE (absolute errors in brackets). "Final" is the per-camera profile system described in §2.3.
MOT20 is kept separate so its 1,066 crossings don't drown the 210 from the other cameras.

By density (CAVIAR + MOT17, `feet_band`): quiet 24.2% → 19.7%, medium 30.1% → 23.0%,
busy 25.8% → 25.8% (baseline → final). Mean IDF1: dev 0.687 → 0.718; MOT20 0.220 → 0.494.

### 2.5 What the results show

1. **On normal retail-density cameras the final system cuts dev errors by a quarter**
   (43 → 32), mainly on the corridor camera (33 → 22). The shop-entrance test camera
   is unchanged at 15 errors (20 of 26 clips exactly right, against 17 before).
2. **Crowds were a detection problem, and input resolution was the lever.** On MOT20,
   recall rises from 0.04–0.25 (baseline) to 0.38–0.65 with the crowd profile, and
   errors fall from 914 to 327. MOT20-03 (night, 130 people per frame) improves from
   92% to 21%.
3. **MOT20-05 is still not solved.** 195 people per frame, recall 0.38: the system still
   misses about a third of the crossings (336 OUT counted against 514 real). Packed
   crowds of this kind would need a detector trained on crowd data, which is outside this project.
4. **MOT20-02 got worse with the crowd profile** (8 → 47 errors). Here it **over-counts**
   (111 IN against 86, 50 OUT against 28), while every other MOT20 camera under-counts.
   More detected people means more tracks, and fragmented tracks can cross the line more
   than once. Once-per-direction might help, but choosing it from this test result
   would be tuning on test, so it was not done. This is reported as a known weakness.
5. **Centre point on MOT20 (observation, not used for any choice):** with the final tracks,
   `centre_simple` scores 26.6% on MOT20 against 30.7% for `feet_band`. In dense crowds
   the lower body is often hidden, so the bottom of the box is less reliable. On the
   angled retail cameras, feet+band remains much better (shop front 116.7% vs 20.8%).
6. **Speed cost:** standard profile about 25 ms/frame on CAVIAR; crowd profile 83–112 ms/frame
   on 1080p crowds (about 9–12 fps). Real-time on crowd cameras needs the Phase 6
   export (TensorRT / FP16) or a smaller crowd input size.

## 3. Zones, dwell, heatmaps (Phase 4)

TODO: run and fill in.

## 4. Streaming and performance (Phase 6)

TODO: run and fill in.

## 5. Multi-camera (Phase 7)

TODO: run and fill in.
