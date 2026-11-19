# 06 · Staff filtering and privacy

## 1. Staff filtering (`retailflow/staff.py`)

Footfall should count **customers**. Employees walk in and out all day and would
inflate every number.

### Rule

```json
"staff": {"zones": ["shop_inside"], "min_seconds": 45}
```

A track is **staff** once it has spent 45 s in total inside any of the listed zones.
Customers pass through these areas; employees stay there. In a real store the zone
would be behind the till or a stock-room door. On the CAVIAR shop-front camera, the
only area where an assistant spends long periods is the inside of the shop.

**Why a rule, not a model:** none of the public datasets label who is staff (CAVIAR's
roles are only `walker`, `browser`, `meeters`), so a classifier could not be trained or
tested. The rule uses only positions and time, so nothing about appearance is stored.

**Why 45 s:** measured on ground truth for this camera, customers spend at most 26 s
inside the shop (`OneStopMoveEnter2front`). The people present all clip long in the two
`ShopAssistant` clips spend 57–115 s. The value was read from that distribution, so it is a
**per-store setting**, not a tuned result.

### Online behaviour

A track is only known to be staff after 45 s. The pipeline then emits a `staff_flag`
event. Every consumer drops **all** events of that track ID, including earlier ones, and
the summary reports `counts` (everyone) and `counts_customers` (staff removed).
Heatmap cells of each track are held until the track ends (doc 05 §5), so staff never
reach the heatmap.

### What it actually does (checked against ground truth)

| Clip | Ground truth: who the rule should flag | Predicted tracks flagged | Match |
|---|---|---|---|
| `ShopAssistant2front` | GT 0 (assistant), GT 6 | 228, 221 | **228 = GT 0 for 1,277 frames: correct.** 221 is no labelled person (likely a mannequin). GT 6 missed. |
| `ShopAssistant1front` | GT 0 (assistant) | 2 | **Wrong.** 2 is a mannequin (present in every frame). The real assistant's track broke into 8+ IDs (longest 189 frames, 7.6 s) and never reached 45 s. |

So the rule finds a staff member **when tracking keeps one ID**, and fails when the ID
breaks. On the shop-front camera tracking is weak (IDF1 0.466), so it found 1 of the 3 people
it should have flagged. Applying it changed zone accuracy very little (doc 05 §6.1).

**Honest status: implemented, unit-tested, works in principle, not reliable on this camera.**
Improvements for real use, none built here:
- a uniform-colour check: a staff shirt colour range in HSV, compared on the torso box;
- keeping a track's staff time across ID switches (needs ReID, which stays in memory only);
- an explicit staff-only door with a counting line.

## 2. Privacy

| Principle | How |
|---|---|
| No identities | Track IDs are integers that restart every session and mean nothing outside it. |
| No face images | Nothing saves crops. Events hold camera, time, track ID, position, zone, dwell, flags. |
| No appearance stored | ReID embeddings (if BoT-SORT+ReID is used) stay in tracker memory and are never written. The default tracker does not use them. |
| Aggregated outputs | Heatmaps hold person-seconds per cell; the rendered background is a median image (people removed). |
| Blurred video | `--blur-heads` pixelates heads in any saved or previewed video. |

### Head blur (`retailflow/privacy.py`)

`python run.py ... --save-video --blur-heads`

The detector finds bodies, not faces, and in angled CCTV a face is often tiny, turned away
or hidden, so a face detector would miss many. Instead, **the top 22% of every person box**
(plus 15% width on each side and 5% above) is **pixelated** into 6 × 6 blocks:
- an adult's head is about 1/7–1/8 of body height (≈ 0.13–0.14), and 0.22 leaves room for loose boxes, bent heads and hair;
- pixelation cannot be undone by sharpening, unlike a light blur.

Blurring is applied to the **output image only**. Detection and tracking always see the
original frame, so accuracy is unchanged. Checked visually on the `ShopAssistant1front`
demo video: all visible heads are pixelated, bodies and boxes are not.

**Limitation:** a person who is not detected is not blurred. For recordings that leave
the premises, the whole frame should be treated as personal data.

## 3. Tests

`tests/test_heatmap_staff_privacy.py`: staff flagged after `min_seconds` and not before,
no rule = no flags, offline staff detection, head box geometry, pixelation changes only
the head region.
