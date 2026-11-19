# Downloading the datasets

All data lives in `data/` (git-ignored). Every dataset here is **public and for
research / non-commercial use**. Nothing in this repo redistributes the data:
the scripts download it from the original source.

Run all commands from the repo root with the venv active (`.\.venv\Scripts\Activate.ps1`).

## Selection rule

Only **static cameras mounted at an angle or from the side**, like typical
retail CCTV. No overhead (top-down) views and no moving cameras.

| Dataset | Used for | Camera | Size downloaded | Licence |
|---|---|---|---|---|
| CAVIAR (Portugal shopping centre) | counting, zones, dwell, multi-camera | 2 static views per scene: angled down the corridor ("cor") and facing the shop fronts ("front") | 581 MB (52 videos + 52 GT files) | Creative Commons BY-SA; acknowledge the EC-funded CAVIAR project (IST 2001 37540) |
| MOT17 (static train sequences 02, 04, 09) | counting and tracking in medium/busy scenes | static, elevated, angled street/square views | 5.6 GB zip, ~1 GB extracted for 3 sequences | CC BY-NC-SA 3.0 (MOTChallenge) |
| MOT20 (train sequences) | very busy scenes | static, elevated, angled | 4.8 GB zip (Phase 3) | CC BY-NC-SA 3.0 (MOTChallenge) |
| MMPTRACK | multi-camera (only if feasible) | calibrated multi-view, includes retail | TODO: check size in Phase 7 | Research use; TODO: confirm terms in Phase 7 |

## 1. CAVIAR

Source: <https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/> (files are hosted under `.../rbf/CAVIARDATA2/`).

```powershell
python tools/download_caviar.py               # both views, all 26 clips
python tools/download_caviar.py --views cor   # corridor view only (~290 MB)
```

Result: `data/caviar/<Clip><view>/<Clip><view>.mpg` and `<c|f><code>gt.xml`.
Video is 384x288 at 25 fps (half-resolution PAL, MPEG-2). Ground truth is CVML
XML with centre-based boxes (`<box xc yc w h>`) per `<object id>` per `<frame number>`.

Caveats (from the dataset page and our own inventory):
- The two views are **not guaranteed frame-aligned**: each may start at a
  slightly different time. Use the on-screen time code to align (Phase 7).
- Some front-view ground truth is incomplete (e.g. `OneLeaveShop2front` is only
  labelled up to frame 289 of 1119).

## 2. MOT17 (static sequences only)

Source: <https://motchallenge.net/data/MOT17/>. Only whole-benchmark zips are offered.

```powershell
python tools/download_mot.py MOT17 --seqs MOT17-02-FRCNN MOT17-04-FRCNN MOT17-09-FRCNN
```

The zip is kept in `data/_downloads/` so sequences can be re-extracted without
re-downloading; add `--delete-zip` to remove it. MOT17 ships each sequence three
times (`-DPM`, `-FRCNN`, `-SDP`) with **identical frames and ground truth**; only
the public detections differ, and we don't use those, so one copy is enough.

Why 02, 04 and 09: we measured camera motion on all 7 training sequences
(`python tools/check_camera_motion.py`, background-only ORB + RANSAC):

| Sequence | Median shift (px/frame) | 90th pct | Verdict |
|---|---|---|---|
| MOT17-02 | 0.05 | 0.17 | static |
| MOT17-04 | 0.02 | 0.03 | static |
| MOT17-09 | 0.02 | 0.05 | static |
| MOT17-05 | 3.37 | 9.74 | moving |
| MOT17-10 | 3.78 | 7.42 | moving |
| MOT17-11 | 3.93 | 6.74 | moving |
| MOT17-13 | 4.54 | 28.30 | moving |

## 3. MOT20 (Phase 3)

```powershell
python tools/download_mot.py MOT20 --seqs <chosen sequences> --delete-zip
```

TODO: run in Phase 3, then record chosen sequences and camera-motion check here.

## 4. MMPTRACK (Phase 7, optional)

TODO: check download size, camera angles and licence before downloading.

## Verify

```powershell
python tools/inventory.py   # frames, GT coverage and people-per-frame per sequence
```
