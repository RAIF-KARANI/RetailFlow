# 09 · Evaluation method

The aim is to prove **"counts people in and out reliably, including in busier
scenes"** with numbers that anyone can reproduce from public data.

## 1. Data used

| Camera (scene file) | Sequences | Counting line | True crossings |
|---|---|---|---|
| `caviar_cor` | 26 CAVIAR corridor clips | across the corridor floor; IN = towards camera | 95 (49 in / 46 out) |
| `caviar_front` | 26 CAVIAR front clips | Promod shop threshold; IN = entering the shop | 72 (31 in / 41 out) |
| `MOT17-02` | 1 | vertical line across the square; IN = rightwards | 18 (15 / 3) |
| `MOT17-04` | 1 | horizontal line across the street; IN = towards camera | 13 (8 / 5) |
| `MOT17-09` | 1 | vertical line on the shop-front pavement; IN = rightwards | 12 (10 / 2) |
| **Subtotal (retail-density cameras)** | **55** | | **210 (113 in / 97 out)** |
| `MOT20-01` | 1 | station concourse; vertical line across the main flow; IN = rightwards | 29 (21 / 8) |
| `MOT20-02` | 1 | same station, other camera; horizontal line; IN = towards camera | 114 (86 / 28) |
| `MOT20-03` | 1 | night street, two walkways; horizontal line; IN = towards camera | 385 (9 / 376) |
| `MOT20-05` | 1 | packed crowd, no dominant flow (extreme, not store-like); IN = towards camera | 538 (24 / 514) |
| **MOT20 subtotal** | **4** | | **1,066 (140 / 926)** |

MOT20 lines were also placed from traffic plots before any system output was seen.
MOT20 is always **reported separately and per camera**. Otherwise its 1,066 crossings
would outweigh the other 210 and hide how the system does at normal retail density.

### Dev / test split

| Split | Cameras | Crossings | Used for |
|---|---|---|---|
| **dev** | `caviar_cor`, MOT17-02/04/09 | 138 | every choice: image size, model, tracker, band, min age |
| **test** | `caviar_front`, MOT20-01/02/03/05 | 72 + 1,066 | reporting only |

The split is by **camera**, so no test camera's view or lines influenced a choice.
One documented exception: the crowd profile was added after a diagnosis on MOT20
(see `10_RESULTS.md` §2.3 for how that was kept honest).

10 sequences have zero true crossings (e.g. `WalkByShop1front`: people walk
past the shop but nobody enters). They still matter: every crossing the system
reports there is a false count.

**Sanity check of the shop-entrance truth:** the true counts match the
CAVIAR clip names. `TwoEnterShop*` gives INs, `TwoLeaveShop*` / `OneLeaveShop*` give
only OUTs, and `WalkByShop1` / `ThreePastShop1-2` give zero. So the line
separates "entered the shop" from "walked past", as intended.

## 2. True counts from ground truth

Ground truth gives boxes per person per frame, not IN/OUT events. We derive the
true counts by applying **the same line logic** to the ground-truth **feet
points** (bottom-centre of each labelled box):

```
TRUTH_CFG = feet point + scene band (5 px CAVIAR, 15 px MOT17) + every crossing counts
```

- The band is used for truth too, because hand-drawn boxes also jitter by a few pixels.
- No min-age or once-per-direction filter: GT IDs are clean, and a real person
  who enters, leaves and re-enters really did cross 3 times.
- **This definition is fixed.** Every experiment is scored against the same truth.

## 3. Predicted counts

Each run saves `tracks.csv` (all tracked boxes). `evaluate.py` recounts them
under each counting variant:

| Variant | Point | Band | Min age | Once per direction |
|---|---|---|---|---|
| `centre_simple` (baseline) | box centre | no | 0 | no |
| `feet_simple` | feet | no | 0 | no |
| `feet_band` | feet | yes | 0 | no |
| `feet_band_age` | feet | yes | 10 frames | no |
| `feet_band_age_once` | feet | yes | 10 frames | yes |

The detector/tracker config (model, image size, tracker) is a separate
dimension: see `benchmark.py`.

## 4. Metrics

### Counting (the main result)
Per sequence:
- `true_in`, `true_out`, `pred_in`, `pred_out`
- `abs_err = |pred_in − true_in| + |pred_out − true_out|`
- `pct_err = abs_err / (true_in + true_out)` (undefined when there are no true crossings)

Per group (camera, dataset, density):
- **WAPE** = Σ abs_err / Σ true crossings. Weighted absolute percentage error.
  It does **not** let errors cancel. Example: 3 missed INs and 3 extra OUTs
  give a perfect net total but WAPE = 60% on 10 true crossings. A unit test checks this.
- **Exact sequences**: how many sequences have exactly the right IN and OUT.

### Tracking quality
Computed with `motmetrics`, matching boxes at IoU ≥ 0.5:
- **IDF1**: share of detections assigned to the correct identity over the
  whole sequence. The metric most related to counting, since ID switches
  cause double counts and lost counts.
- **MOTA**: 1 − (misses + false positives + ID switches) / GT boxes.
- **ID switches**, precision, recall.

**HOTA is not reported.** It needs TrackEval, which is not in the agreed tech stack.
IDF1 already measures identity consistency, which is what matters most for counting.
TrackEval can be added later if a HOTA number is needed.

### Speed
- `avg_proc_ms`: detect + track + count per frame (no decoding/drawing).
- `throughput_fps`: whole loop including decoding and writing.
Speed numbers are only comparable with the GPU in performance mode (see TROUBLESHOOTING).

## 5. Density groups

By mean ground-truth people per frame:

| Group | Mean people / frame | Sequences here |
|---|---|---|
| quiet | < 2.5 | most CAVIAR front, some corridor |
| medium | 2.5 – 15 | busier CAVIAR corridor, MOT17-09 |
| busy | ≥ 15 | MOT17-02, MOT17-04, and all MOT20 (MOT20 reported separately) |

The same ≥ 15 threshold selects which cameras use the `crowd` detector profile.

## 6. Known limitations of this evaluation

- **Small number of crossings per sequence** (0–18), so one error is a large
  percentage. Group WAPE over many sequences is more meaningful.
- **CAVIAR labels:** some front-view GT is incomplete (see `01_SETUP_AND_DATA.md`),
  which can make a correct prediction look wrong. Affected clips are flagged in results.
- **Unlabelled people:** CAVIAR does not label every distant person, so a correct
  detection of an unlabelled person counts as a false positive in MOTA/IDF1.
- **MOT17 counting lines are our choice.** The line positions were chosen from
  traffic plots before seeing any system output, and are not changed after.
- **Frame alignment:** GT frame *i* is assumed to be decoded frame *i*. Verified
  in section 7.

## 7. Frame-alignment check

`python tools/check_alignment.py --config baseline` compares GT boxes at frame
*f* with the baseline's tracked boxes at frame *f + k* (k = −3…+3), using the mean
best IoU per GT box.

| Peak offset | Sequences |
|---|---|
| k = 0 | 43 |
| k = +1 | 11 |
| k = −1 | 1 |

In every non-zero case, the k = 0 score is within 0.005 IoU of the peak (e.g.
`EnterExitCrossingPaths1front`: 0.663 at k=0 vs 0.665 at k=+1), and the curves are
symmetric around 0. A real offset would shift the whole curve. The slight lean
to +1 fits tracker boxes lagging a fast-moving person by a fraction of a frame.
**Conclusion: GT frame i = decoded frame i; no offset correction.**

Side finding: MOT17-02 and MOT17-04 have a mean best IoU of only ~0.27 at k=0 (vs
0.6–0.87 for most CAVIAR corridor clips), because many labelled people have no
detection at all at `imgsz=640`.

## 8. Reproduce

```powershell
python benchmark.py --configs baseline      # detection + tracking -> outputs/runs/
python evaluate.py --group camera           # -> outputs/results.csv + summary tables
python evaluate.py --group density
python benchmark.py --plan onefactor --datasets caviar mot17       # Phase 3 grid (dev)
python benchmark.py --configs n640_bytetrack m960_botsort m1920_botsort --datasets mot17 mot20
python evaluate.py --configs m960_botsort --sweep                  # counting settings (dev)
python evaluate.py --configs n640_bytetrack final --group camera   # before / after
```
