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
| MEVA (one camera, one hour) | long real-footage test | static elevated indoor entrance camera | 914 MB (12 clips) | CC BY 4.0 (Kitware Inc. and IARPA) |
| MMPTRACK | multi-camera (only if feasible) | calibrated multi-view, includes retail | not downloaded (no public link; size not published) | Research use only; signed user agreement e-mailed to the organisers |

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
- The two views are **not frame-aligned**: measured from the on-screen time code, the front
  view starts 3.04–3.12 s earlier in 25 of 26 clips (`python tools/sync_caviar_views.py`, docs/11).
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
python tools/download_mot.py MOT20 --seqs MOT20-01 MOT20-02 MOT20-03 MOT20-05
```

All four training sequences were extracted (the zip has to be downloaded whole anyway).
Camera motion (`tools/check_camera_motion.py`) and density:

| Sequence | Size | Frames | Median shift (px/frame) | Mean / max people per frame | Use |
|---|---|---|---|---|---|
| MOT20-01 | 1920x1080 | 429 | 0.55 (static) | 46 / 55 | busy test |
| MOT20-02 | 1920x1080 | 2782 | 0.14 (static) | 56 / 97 | busy test |
| MOT20-03 | 1173x880 | 2405 | 0.09 (static) | 130 / 199 | busy test (small people, night) |
| MOT20-05 | 1654x1080 | 3315 | 0.11 (static) | 195 / 220 | **extreme** stress test (packed, no dominant flow) |

MOT20-01 moves slightly more (0.55 px) because with ~50 people per frame little
background is left to match; it is still well under the 1 px static threshold.

## 4. MMPTRACK (Phase 7, optional): not downloaded

Checked on the official page (<https://iccv2021-mmp.github.io/subpage/dataset.html>):
research use only. Access requires downloading a user agreement, signing it and e-mailing it to the
organisers; there is no direct download link and no published size (about 5 h train + 1.5 h validation
video, 640x320 at 15 fps, 5 environments including retail, calibrated cameras, cross-camera IDs).
Signing and sending the agreement is the project owner's decision, so it was not requested
(docs/11_MULTI_CAMERA.md §1).

## 5. MEVA (long real-footage test, after Phase 8)

Source: <https://mevadata.org> (Kitware Inc. and IARPA), **CC BY 4.0**. Public S3 bucket
`mevadata-public-01`, no account needed. 3,956 ground-camera clips of 5 minutes (476 GB in total),
1920x1072 or 1920x1080 H.264 at 30 fps, recorded at a training site with hired actors.
There are no in/out counts in the main release, so it is used for **long-run** testing (stability,
real time, plausibility), not for accuracy.

```powershell
python tools/meva_index.py      # file names + sizes only -> outputs/meva_index.csv (no video)
```

Downloaded for the test: camera **school G420** (indoor entrance lobby, elevated angled view),
2018-03-07 10:05-11:05, the busiest continuous hour by file size: 12 clips, **914 MB**, joined
without re-encoding into `data/meva/G420_2018-03-07_10-05_1h.mp4` (FFmpeg concat, `-c copy`).
Clips were fetched over HTTPS from `https://mevadata-public-01.s3.amazonaws.com/<key>`.
Picking the camera used one 3 MB slice of a midday clip from each camera (about 75 MB in total).

Attribution: MEVA by Kitware Inc. and IARPA, CC BY 4.0. Corona et al., "MEVA: A Large-Scale
Multiview, Multimodal Video Dataset for Activity Detection", WACV 2021.

## Verify

```powershell
python tools/inventory.py   # frames, GT coverage and people-per-frame per sequence
```
