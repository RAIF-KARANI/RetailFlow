"""Environment sanity check: CUDA, GPU, YOLO26 on GPU, tracker ReID support.

Run:  python tools/check_gpu.py
Expected: "CUDA available: True", the GPU name, and a YOLO26n inference time.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO


def main() -> None:
    print(f"torch {torch.__version__} | CUDA build {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("!! PyTorch cannot see the GPU. Reinstall torch from the CUDA index (see docs/01).")
        return
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | {props.total_memory / 2**30:.1f} GB")

    # Load the smallest YOLO26 model (auto-downloads weights on first run).
    model = YOLO("yolo26n.pt")
    frame = np.zeros((640, 640, 3), dtype=np.uint8)  # dummy black frame
    model.predict(frame, device=0, verbose=False)     # warm-up (CUDA init, kernel selection)
    t0 = time.perf_counter()
    n = 20
    for _ in range(n):
        model.predict(frame, device=0, verbose=False)
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"YOLO26n on GPU: {ms:.1f} ms/frame at 640 (dummy frame, includes pre/post-processing)")

    # Show the default BoT-SORT config shipped with this Ultralytics version,
    # to confirm the ReID option exists and what it defaults to.
    cfg = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "botsort.yaml"
    # Plain-text scan instead of a YAML parser: we only need three flat keys,
    # and this avoids an extra dependency (PyYAML is not installed).
    wanted = ("with_reid:", "model:", "gmc_method:")
    lines = [ln.split("#")[0].strip() for ln in cfg.read_text().splitlines()]
    found = [ln for ln in lines if ln.startswith(wanted)]
    print(f"ultralytics {ultralytics.__version__} botsort.yaml -> {found}")


if __name__ == "__main__":
    main()
