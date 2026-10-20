#!/usr/bin/env python3
"""Check dependencies/models; optionally execute each model on synthetic tensors."""
import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=os.environ.get("CAMEX_OFFICE_MODELS"))
    parser.add_argument("--provider", default=os.environ.get("CAMEX_OFFICE_PROVIDER", "CUDAExecutionProvider"),
                        choices=["CUDAExecutionProvider", "CPUExecutionProvider"])
    parser.add_argument("--exercise-models", action="store_true",
                        help="Run one blank tensor per model; assert CUDA nodes for GPU mode")
    args = parser.parse_args()
    os.umask(0o077)
    if args.models is None:
        parser.error("Set CAMEX_OFFICE_MODELS or --models to the supplied, extracted model directory")
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "research_demo"), str(root / "office_demo")]
    modules = ("numpy", "cv2", "PIL", "onnx", "onnxruntime", "skimage", "scipy", "lap", "cython_bbox",
               "fastapi", "uvicorn", "multipart", "qdrant_client", "core", "vision.head_tracker", "vision_backend")
    for name in modules:
        importlib.import_module(name)
    import numpy as np
    import onnxruntime as ort
    from vision.networks import MODEL_HASHES, sha256
    if not np.__version__.startswith("1."):
        raise RuntimeError("Use the NumPy 1.26.4 pin, not NumPy 2")
    if ort.__version__ != "1.18.0":
        raise RuntimeError("Expected the pinned ONNX Runtime 1.18.0")
    if args.provider not in ort.get_available_providers():
        raise RuntimeError(f"Requested provider unavailable: {args.provider}")
    for executable in ("ffmpeg", "ffprobe"):
        if not shutil.which(executable):
            raise RuntimeError(f"Required executable not found: {executable}")
    models = args.models.expanduser().resolve()
    result = {"python": platform.python_version(), "machine": platform.machine(),
              "numpy": np.__version__, "onnxruntime": ort.__version__,
              "requested_provider": args.provider, "available_providers": ort.get_available_providers(),
              "synthetic_execution": args.exercise_models, "models": []}
    for name, expected in MODEL_HASHES.items():
        model = models / name
        if not model.is_file() or sha256(model) != expected:
            raise RuntimeError(f"Missing or checksum-mismatched supplied model: {name}")
        entry = {"file": name, "sha256_verified": True}
        if args.exercise_models:
            # Profiling distinguishes real CUDA node execution from provider registration.
            with tempfile.TemporaryDirectory(prefix="office-ort-check-") as temp:
                options = ort.SessionOptions()
                options.enable_profiling = True
                options.profile_file_prefix = str(Path(temp) / "profile")
                options.intra_op_num_threads = 2
                options.inter_op_num_threads = 1
                session = ort.InferenceSession(str(model), sess_options=options, providers=[args.provider])
                if session.get_providers()[0] != args.provider:
                    raise RuntimeError(f"Provider initialization fell back for {name}")
                tensors = {}
                for value in session.get_inputs():
                    if value.type != "tensor(float)":
                        raise RuntimeError(f"Unexpected input type in supplied model: {name}")
                    shape = [n if isinstance(n, int) and n > 0 else 1 for n in value.shape]
                    tensors[value.name] = np.zeros(shape, dtype=np.float32)
                session.run(None, tensors)
                events = json.loads(Path(session.end_profiling()).read_text())
                providers = sorted({event.get("args", {}).get("provider") for event in events
                                    if event.get("args", {}).get("provider")})
                entry["executed_providers"] = providers
                if args.provider not in providers:
                    raise RuntimeError(f"No profiled {args.provider} execution for {name}")
                del session
        result["models"].append(entry)
    print(json.dumps(result, indent=2))
    if not args.exercise_models:
        print("Imports and hashes passed; actual GPU execution has NOT been tested.")


if __name__ == "__main__":
    main()
