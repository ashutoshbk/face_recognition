from __future__ import annotations

import argparse
from pathlib import Path

import torch

from camex_mct.models import YoloXPersonDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Export static-batch YOLOX TorchScript")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        print(f"reusing existing TorchScript model: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)

    detector = YoloXPersonDetector(
        args.experiment,
        args.checkpoint,
        input_size=args.input_size,
    )
    dummy = torch.zeros(
        (args.batch_size, 3, args.input_size, args.input_size),
        dtype=torch.float32,
        device="cuda",
    )
    with torch.inference_mode():
        traced = torch.jit.trace(detector.model, dummy, strict=False)
        traced = torch.jit.freeze(traced.eval())
    temporary = output.with_suffix(output.suffix + ".part")
    torch.jit.save(traced, temporary)
    temporary.replace(output)
    detector.close()
    print(f"saved TorchScript model: {output}")


if __name__ == "__main__":
    main()
