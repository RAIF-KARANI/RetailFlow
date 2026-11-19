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

TODO: run and fill in (tracker × model × imgsz grid, MOT20, best configuration vs baseline).

## 3. Zones, dwell, heatmaps (Phase 4)

TODO: run and fill in.

## 4. Streaming and performance (Phase 6)

TODO: run and fill in.

## 5. Multi-camera (Phase 7)

TODO: run and fill in.
