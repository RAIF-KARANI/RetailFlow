"""Export the detector to ONNX and TensorRT (FP16) for the speed comparison (docs/08).

    python tools/export_models.py                      # yolo26m at 960 and 1920
    python tools/export_models.py --model yolo26n.pt --imgsz 640

Outputs go to models/ (git-ignored), named <stem>_<imgsz>.<onnx|engine>, e.g.
models/yolo26m_960.engine. Pass that path to run.py / benchmark.py with --model.

Both formats are exported with **dynamic** input shapes. A fixed-shape engine
would force every frame into an imgsz x imgsz square, so a 16:9 camera would do
almost twice the work of PyTorch's rectangular letterbox (e.g. 960x544).
`imgsz` is the largest side the engine is optimised for. TensorRT builds are
specific to this GPU and TensorRT version, so they are rebuilt, never shared.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

# Never let Ultralytics pip-install packages by itself during export: it downgraded
# onnxruntime-gpu once (docs/TROUBLESHOOTING.md). Install requirements-export.txt instead.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from ultralytics import YOLO  # noqa: E402  (must come after the environment setting)

OUT = Path("models")


def export(model: str, imgsz: int, fmt: str) -> Path:
    stem = Path(model).stem
    target = OUT / f"{stem}_{imgsz}.{'engine' if fmt == 'engine' else 'onnx'}"
    if target.exists():
        print(f"[skip] {target} exists")
        return target
    t0 = time.perf_counter()
    kwargs = dict(format=fmt, imgsz=imgsz, dynamic=True, batch=1)
    if fmt == "engine":
        kwargs.update(half=True, workspace=4)     # FP16, 4 GB build workspace
    else:
        kwargs.update(simplify=True)
    produced = Path(YOLO(model).export(**kwargs))
    OUT.mkdir(exist_ok=True)
    shutil.move(str(produced), target)
    # The engine export leaves its intermediate ONNX files next to the .pt. Keep the FP16
    # one (it builds engines for other sizes, see engine_from_fp16_onnx) and drop the rest.
    fp16 = Path(model).with_suffix(".fp16.onnx")
    if fmt == "engine" and fp16.exists():
        shutil.move(str(fp16), OUT / fp16.name)
    if fmt == "engine" and Path(model).with_suffix(".onnx").exists():
        Path(model).with_suffix(".onnx").unlink()
    print(f"[ok] {target} ({target.stat().st_size / 1e6:.0f} MB, {time.perf_counter() - t0:.0f} s)")
    return target


def engine_from_fp16_onnx(fp16_onnx: Path, template_engine: Path, imgsz: int, opt_hw=(1088, 1920)) -> Path:
    """Build a TensorRT engine straight from an FP16 ONNX file (fallback, docs/TROUBLESHOOTING).

    Ultralytics' engine export first converts the ONNX to FP16 with ModelOpt AutoCast, which
    runs a reference inference on the CPU at full size. At 1920 px that ran out of memory.
    The FP16 ONNX does not depend on the input size (dynamic axes), so the one made during
    the 960 export is reused, and only the TensorRT build is repeated with a larger shape
    range. The Ultralytics metadata header (class names, stride...) is copied from an
    existing engine, so Ultralytics loads both engines the same way.
    """
    import json

    import tensorrt as trt

    target = OUT / f"yolo26m_{imgsz}.engine"
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(fp16_onnx)):
        raise RuntimeError([parser.get_error(i) for i in range(parser.num_errors)])
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    profile = builder.create_optimization_profile()
    name = network.get_input(0).name
    profile.set_shape(name, (1, 3, 32, 32), (1, 3, *opt_hw), (1, 3, imgsz, imgsz))
    config.add_optimization_profile(profile)
    t0 = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT build failed")
    with open(template_engine, "rb") as f:
        n = int.from_bytes(f.read(4), byteorder="little")
        meta = json.loads(f.read(n).decode("utf-8"))
    meta["imgsz"] = [imgsz, imgsz]
    blob = json.dumps(meta).encode("utf-8")
    with open(target, "wb") as f:
        f.write(len(blob).to_bytes(4, byteorder="little"))
        f.write(blob)
        f.write(plan)
    print(f"[ok] {target} ({target.stat().st_size / 1e6:.0f} MB, {time.perf_counter() - t0:.0f} s, built from {fp16_onnx})")
    return target


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="yolo26m.pt")
    p.add_argument("--imgsz", type=int, nargs="+", default=[960, 1920])
    p.add_argument("--formats", nargs="+", default=["onnx", "engine"], choices=["onnx", "engine"])
    p.add_argument("--from-fp16-onnx", default=None, metavar="FILE",
                   help="build the engine directly from this FP16 ONNX (e.g. yolo26m.fp16.onnx), "
                        "copying metadata from models/yolo26m_960.engine")
    args = p.parse_args()
    if args.from_fp16_onnx:
        for s in args.imgsz:
            engine_from_fp16_onnx(Path(args.from_fp16_onnx), OUT / "yolo26m_960.engine", s)
        return
    for s in args.imgsz:
        for f in args.formats:
            export(args.model, s, f)


if __name__ == "__main__":
    main()
