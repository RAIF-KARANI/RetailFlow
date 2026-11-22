"""Run a whole store live: every camera at once, one store-wide ID per person.

    python run_store.py --store scenes/stores/smartspaces_071.json --seconds 300 --api http://127.0.0.1:8000
    python run_store.py --store ... --cams camera_0639 camera_0643 camera_0644 camera_0645   # fewer cameras

What it does:
  1. every camera's video is replayed at real-time speed in its own thread, and the
     pipeline always takes the NEWEST frame (as from a live camera; frames it has no
     time for are skipped, and counted);
  2. each camera is analysed up to `--analysis-fps` times per second: the due cameras'
     newest frames go through ONE detector call when a batch engine exists
     (models/yolo26m_960_b16.engine, tools/export_models.py --max-batch 16), then each
     camera's own tracker (retailflow/detector.py MultiCameraTracker);
  3. every `--link-every` seconds the new tracks go to a separate LINKING PROCESS, which
     links them into store-wide people (floor position + clothing colours, colours in
     memory only) with stable person numbers and sends the store state (incl. a floor
     heatmap) to the API / dashboard. A thread was tried first: its pandas work shared
     the GIL with detection and starved it. Detection never waits for linking.

With --preview, http://127.0.0.1:8090 shows the cameras live (all as a mosaic, or one),
each box labelled with the person's store-wide number, heads pixelated, plus a Stop button;
the store page on the dashboard links to it.

No RTSP here: re-encoding 15 full-HD streams would need the CPU the pipeline needs
(single cameras over RTSP: run.py / demo.ps1). Outputs in outputs/store/<store>/<tag>/:
per-camera tracks.csv (video frame numbers, so they can be scored), links.csv
(camera, track, person) and summary.json. No colour descriptors are written.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import queue
import time
from pathlib import Path

import cv2
import numpy as np

from retailflow.detector import MultiCameraTracker
from retailflow.profiles import get_profile
from retailflow.sender import StoreApiClient
from retailflow.sources import LatestFrameReader
from retailflow.store_live import StoreLinker, link_worker, load_store, take_delta
from retailflow.store_preview import mosaic, single


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--store", required=True, help="scenes/stores/<store>.json")
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--analysis-fps", type=float, default=5.0, help="analyses per camera per second (target)")
    p.add_argument("--link-every", type=float, default=3.0, help="seconds between store-wide re-links")
    p.add_argument("--cams", nargs="*", default=None, help="only these cameras")
    p.add_argument("--no-colour", action="store_true", help="link on floor position only")
    p.add_argument("--hw-decode", action="store_true", help="decode videos on the GPU (default: CPU)")
    p.add_argument("--api", default=None, metavar="URL")
    p.add_argument("--profile", default="standard")
    p.add_argument("--preview", nargs="?", const=8090, type=int, default=None, metavar="PORT",
                   help="live camera view at http://127.0.0.1:PORT/ (default 8090)")
    p.add_argument("--no-blur", action="store_true", help="do not pixelate heads in the preview")
    p.add_argument("--batch", type=int, default=None,
                   help="images per detector call (default: 16 if models/yolo26m_960_b16.engine exists, else 1)")
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    store = load_store(args.store)
    cams = [c for c in store.cameras if not args.cams or c in args.cams]
    store.cameras = cams
    tag = args.tag or f"{len(cams)}cams_{int(args.seconds)}s_{args.analysis_fps:g}fps"
    out = Path("outputs/store") / store.store_id / tag
    prof = get_profile(args.profile)
    weights = prof.weights("auto")
    batch_engine = Path(f"models/yolo26m_{prof.imgsz}_b16.engine")
    batch = args.batch if args.batch is not None else (16 if batch_engine.exists() else 1)
    if batch > 1 and str(weights).endswith(".engine"):
        weights = str(batch_engine)
    mct = MultiCameraTracker(weights, prof.tracker, prof.imgsz, prof.conf, 0, cams, batch=batch)
    linker = StoreLinker(store, appearance=not args.no_colour)

    # Warm up with a full batch before the cameras start (TensorRT context, CUDA kernels).
    blank = np.zeros((store.frame_h, 1920, 3), np.uint8)
    for _ in range(3):
        mct.model.predict([blank] * min(batch, len(cams)) if batch > 1 else blank,
                          imgsz=prof.imgsz, device=0, verbose=False)

    preview = None
    if args.preview is not None:
        from retailflow.preview import PreviewServer
        # 1920 px wide: the mosaic has 4 tiles across, halving it would make boxes vanish.
        preview = PreviewServer(store.store_id, port=args.preview, views=["all", *cams], width=1920)
        print(f"Live cameras: {preview.url}", flush=True)
    run_info = {"cameras": len(cams), "batch": batch, "decode": "gpu" if args.hw_decode else "cpu", "analysis_fps_target": args.analysis_fps, "model": weights,
                "tracker": prof.tracker, "imgsz": prof.imgsz, "colour": not args.no_colour,
                "source": "video files replayed at real-time speed"}
    if preview:
        run_info["preview_url"] = preview.url
    api = StoreApiClient(args.api, store.to_api(), run_info, store.scene_dir) if args.api else None
    if api:
        print(f"API store run {api.run_id}", flush=True)

    max_frames = int(args.seconds * store.fps)
    readers = {}
    for c in cams:
        # CPU decoding by default. GPU decoding did not speed the run up (the cost was turning
        # frames into Python images, now limited by keep_every), and 15 GPU decoders next to
        # the batch engine ran out of video memory ("hardware accelerator failed", 0x8007000e).
        hw = [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY] if args.hw_decode else []
        cap = cv2.VideoCapture(str(store.scene_dir / c / "video.mp4"), cv2.CAP_FFMPEG, hw)
        # Only every n-th frame becomes an image: at most ~2x the analysis rate is ever needed.
        keep = max(1, int(store.fps / (2 * args.analysis_fps)))
        readers[c] = LatestFrameReader(cap, max_frames, pace_fps=store.fps, keep_every=keep)

    files = {}
    for c in cams:
        (out / c).mkdir(parents=True, exist_ok=True)
        files[c] = open(out / c / "tracks.csv", "w", newline="")
    writers = {c: csv.writer(f) for c, f in files.items()}
    for w in writers.values():
        w.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "conf"])

    interval = 1.0 / args.analysis_fps
    due = {c: 0.0 for c in cams}
    analysed = {c: 0 for c in cams}
    analysed_frames = {c: [] for c in cams}
    batch_ms, batches = [], 0
    ctx = mp.get_context("spawn")
    inbox, outbox = ctx.Queue(), ctx.Queue()
    worker = ctx.Process(target=link_worker, daemon=True,
                         args=(args.store, cams, not args.no_colour, args.api, api.run_id if api else None,
                               inbox, outbox))
    worker.start()
    sent: dict = {}
    numbers: dict = {}                                 # (camera, track) -> store-wide person, for the preview
    last_img: dict = {}
    last_people: dict = {}
    preview_frames = 0
    stopped_by_user = False
    t0 = time.perf_counter()
    next_link = t0 + args.link_every
    result = None
    try:
        try:
            while not all(r.finished for r in readers.values()):
                if preview is not None and preview.stop_requested:
                    stopped_by_user = True                   # Stop button on the preview page
                    break
                now = time.perf_counter()
                frames, index = {}, {}
                if batch > 1:
                    # One tick for all cameras: the batch keeps the same size from call to call.
                    # (Cameras picked one by one gave a different batch size every call, and each
                    # new size made the TensorRT backend re-allocate: 611 ms per batch instead of ~170.)
                    if now >= due[cams[0]]:
                        for c in cams:
                            got = readers[c].newest()
                            if got is not None:
                                index[c], frames[c] = got
                        due[cams[0]] = max(due[cams[0]] + interval, now)
                else:
                    for c in cams:
                        if now >= due[c]:
                            got = readers[c].newest()
                            if got is not None:
                                index[c], frames[c] = got
                                due[c] = max(due[c] + interval, now)     # keep the rate, never bunch up
                if not frames:
                    time.sleep(0.003)
                    continue
                tb = time.perf_counter()
                results = mct.update(frames)
                batch_ms.append((time.perf_counter() - tb) * 1000)
                batches += 1
                for c, people in results.items():
                    analysed[c] += 1
                    analysed_frames[c].append(index[c])
                    for q in people:
                        writers[c].writerow([index[c], q.track_id, f"{q.x1:.1f}", f"{q.y1:.1f}", f"{q.x2:.1f}",
                                             f"{q.y2:.1f}", f"{q.conf:.3f}"])
                    linker.add(c, index[c], people, frames[c])
                    last_img[c], last_people[c] = frames[c], people
                if preview is not None:
                    while True:                              # newest store-wide numbers, without waiting
                        try:
                            kind, payload = outbox.get_nowait()
                        except queue.Empty:
                            break
                        if kind == "numbers":
                            numbers = payload
                    if preview.wants_frame():
                        t_s = max(index.values()) / store.fps
                        blur = not args.no_blur
                        if preview.focus in last_img:
                            view = single(last_img[preview.focus], last_people[preview.focus], numbers,
                                          preview.focus, t_s, blur)
                        else:
                            view = mosaic(last_img, last_people, numbers, cams, t_s, blur=blur)
                        preview.publish(view)
                        preview_frames += 1
                if time.perf_counter() >= next_link:          # hand the new tracks to the linking process
                    inbox.put(("delta", *take_delta(linker, sent)))
                    next_link += args.link_every
        except KeyboardInterrupt:
            # Ctrl+C ends the run like the Stop button: link what was seen and save it.
            # (The linking process ignores Ctrl+C, so it is still there to do the final link.)
            stopped_by_user = True
            print("Stopping: saving the results so far...", flush=True)
        wall = time.perf_counter() - t0
        inbox.put(("final", *take_delta(linker, sent)))
        while True:                                      # skip "numbers" messages until the final one
            kind, payload = outbox.get(timeout=600)
            if kind == "final":
                result = payload
                break
    except BaseException as e:
        # Close the run on the dashboard even when it fails (else it stays "live" forever).
        if api:
            try:
                api.finish({}, {"error": f"{type(e).__name__}: {e}"})
            except Exception:
                pass
        raise
    finally:
        if worker.is_alive():
            worker.terminate()
        if preview is not None:
            preview.close()
        for r in readers.values():
            r.stop()
        for f in files.values():
            f.close()

    state, links, link_ms = result["state"], result["links"], result["link_ms"]
    links.to_csv(out / "links.csv", index=False)
    per_cam = {c: {"analysed": analysed[c], "analysed_fps": round(analysed[c] / wall, 2),
                   "frames_skipped_by_reader": readers[c].skipped} for c in cams}
    summary = {**run_info, "seconds": args.seconds, "wall_s": round(wall, 1),
               "analysed_fps_mean": round(float(np.mean([v["analysed_fps"] for v in per_cam.values()])), 2),
               "batch_ms_mean": round(float(np.mean(batch_ms)), 1) if batch_ms else None,
               "link_ms_mean": round(float(np.mean(link_ms)), 1) if link_ms else None,
               "link_ms_max": round(float(np.max(link_ms)), 1) if link_ms else None,
               "people_now_end": len(state["people_now"]), "unique_people": state["unique_people"],
               "preview_frames": preview_frames, "stopped_by_user": stopped_by_user,
               "per_camera": per_cam}
    for c in cams:
        (out / c / "summary.json").write_text(json.dumps({"frames": max_frames, "fps": store.fps,
                                                          "analysed": analysed[c], "live": True}))
        (out / c / "analysed_frames.txt").write_text(" ".join(map(str, analysed_frames[c])))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if api:
        api.finish(state, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_camera"}, indent=2))
    print(f"Outputs in {out}")


if __name__ == "__main__":
    main()
