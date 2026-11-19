# 04 · Counting reliability

Client requirements 1 and 2: count people **in and out reliably** on **angled /
side-view** cameras, **including busier scenes**.

A count is only as good as three things: **where** we measure a person
(the anchor point), **how** we decide a crossing happened (line logic), and
**whether the person keeps one ID** (tracking). This doc covers all three.

---

## 1. Feet point, not box centre

In an overhead camera the box centre is directly above the feet, so either
point works. In an **angled** camera the box is tall and the centre floats
half a body height **above the floor**. The counting line is drawn **on the
floor**, so only the feet point (bottom-centre of the box) is in the same
plane as the line.

**Measured example (CAVIAR front view, shop threshold line at y = 142):**
the median labelled person here is 44 px tall (`TwoEnterShop1front`). Someone
walking *past* the shop on the walkway has their feet at y ≈ 165, so their
box centre is at ≈ 165 − 22 = **143, right on the line**. As they walk, the
centre wobbles around y = 142 and "crosses" the shop entrance again and again,
although the person never goes near the door.

Result on identical tracks (baseline, `10_RESULTS.md` §1.2): shop-entrance WAPE
**133.3%** with the centre point vs **33.3%** with the feet point (both simple line).

## 2. Band / hysteresis

The feet point jitters by a few pixels between frames (detector noise, legs
moving). A simple line counts every time the sign of the distance flips, so a
person standing on the line can produce several counts.

**Rule:** each track remembers the last side it was **clearly** on, meaning
further than `band_px` from the line. Points inside the band change nothing.
A crossing is counted only on a change from clearly one side to clearly the other.

### Worked numeric example (`scenes/caviar_cor.json`)

Line from (45, 190) to (265, 190), IN point (150, 250), so IN is **below** the
line (towards the camera). `band_px = 5`. A shopper walks down the corridor at
x = 150 and hesitates on the line. Feet y per frame, and signed distance
d = y − 190 (positive = IN side):

| Frame | feet y | d (px) | Simple line (band 0) | With band ±5 |
|---|---|---|---|---|
| 0 | 180 | −10 | side = OUT | side = OUT (clear) |
| 1 | 189 | −1 | — | inside band, ignore |
| 2 | 191 | +1 | **flip → IN +1** | inside band, ignore |
| 3 | 188 | −2 | **flip → OUT +1** | inside band, ignore |
| 4 | 192 | +2 | **flip → IN +1** | inside band, ignore |
| 5 | 197 | +7 | — | clear IN side, **flip → IN +1** |
| **Total** | | | **IN 2, OUT 1** (wrong) | **IN 1** (correct) |

How d is computed: for line A→B and point P, the cross product
(Bx−Ax)(Py−Ay) − (By−Ay)(Px−Ax) divided by |AB| gives the signed distance.
For frame 5: (265−45)(197−190) − 0 = 1540, divided by 220 = **+7 px**. Its sign is
flipped if needed so the IN point is positive.

The crossing must also lie **within the segment** (projection parameter t ∈ [0, 1]),
so someone walking around the end of the line is not counted.

**Band width matters in both directions.** Too small: jitter leaks through. Too
large: real crossings are missed, because a person must travel `band_px` beyond
the line before it counts, and someone stepping through a doorway and
disappearing inside the shop may never get "clearly" past a wide band. The band
width is tuned in §5 on dev data only.

## 3. Minimum track age

A track that appears and disappears within a few frames is often a false
detection (a reflection, a mannequin, a fragment of a person) or a duplicate ID
created after an occlusion. `min_track_age = N` means a track must have
existed for N frames before it can count. The track's side is still updated
while it is young, so a real crossing later is not lost.

## 4. One count per direction per ID

If a track flickers back and forth across the line *beyond* the band (e.g.
a person loitering at a shop door), it can still count IN, OUT, IN, OUT.
`once_per_direction` caps each track ID at one IN and one OUT.
**Trade-off:** someone who genuinely enters, leaves and re-enters within the
same track is under-counted (IN 1 instead of 2). In a store that is rare within
one continuous track (they usually leave the camera view and get a new ID).

## 5. Experiments

All counting variants are evaluated on the **same saved tracks**, so their
effect is separated from detector randomness. Parameters are **chosen on the
dev split** (CAVIAR corridor + MOT17) and then reported on the **test split**
(CAVIAR front + MOT20), which was never used for any choice.

### 5.1 Baseline counting variants (Phase 2 tracks: YOLO26n, ByteTrack, 640)

| Variant | WAPE (all 210 crossings) |
|---|---|
| centre point, simple line (classic) | 92.4% |
| feet point, simple line | 64.3% |
| feet point, band | 27.6% |
| feet point, band, min age 10, once per direction | 26.7% |

### 5.2 Band / min age / once-per-direction sweep on baseline tracks

`python evaluate.py --configs baseline --sweep` (32 combinations). Selected rows:

| Band scale | Min age | Once | Dev WAPE | Test WAPE |
|---|---|---|---|---|
| 0.5× | 10 | yes | 25.4% | 20.8% |
| 1× | 0 | yes | 29.0% | 22.2% |
| 1× | 0 | no | 31.2% | 20.8% |
| 2× | 0 | no | 31.9% | 48.6% |
| 3× | 0 | no | 35.5% | 65.3% |

Findings: wide bands cost a lot on the shop-entrance camera (test); once-per-direction
helps on dev; min age has little effect. The final choice is made on the
**final detector/tracker's** tracks in §5.4.

### 5.3 Detector / tracker grid

A one-factor grid on dev (`10_RESULTS.md` §2.1) chose **YOLO26m, 960 px, BoT-SORT**
(no ReID). Dev errors fell from 43 to 32. Most of the gain comes from 640 → 960 px
and the larger model, which find more of the small and partly hidden people.

### 5.4 Final counting settings

The 32-combination sweep on the final tracks (dev) puts the **defaults** at the top:
feet point, band 1× the scene's `band_px`, min age 0, no once-per-direction (32 errors).
With better tracks, the extra filters no longer pay off on dev, so they stay **off**.
They remain available as options (`--min-age`, `--once-per-direction`).

### 5.5 Busy scenes: crowd profile

In packed crowds (MOT20), counting failed because the detector **missed most people**.
No line logic can count a person who is never detected. Running the detector on the
full 1920 px frame instead of 960 px more than doubled recall. This became a
per-camera **profile** (`"profile": "crowd"` in the scene file), used when the camera
averages 15 or more people per frame. Full story, including what got worse: `10_RESULTS.md` §2.3–2.5.

| MOT20 (test, 1,066 crossings) | WAPE |
|---|---|
| baseline (n640, ByteTrack) | 85.7% |
| standard profile (m960, BoT-SORT) | 74.2% |
| **crowd profile (m1920, BoT-SORT)** | **30.7%** |

## 6. Unit tests

`tests/test_counting.py` covers: signed distance and side, IN-point orientation,
segment-end check, feet/centre anchor, IN/OUT direction, no count without
crossing, simple line counting jitter many times, band ignoring jitter, band
counting once after a real crossing, band scale, min track age, once per
direction, independent tracks, offline table counting.
