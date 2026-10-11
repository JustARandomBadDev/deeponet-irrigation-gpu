from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.transformers.onnx_model import OnnxModel

from deeponet_irrigation.project_paths import REPOSITORY_ROOT


DEFAULT_INPUT = REPOSITORY_ROOT / "models" / "deeponet_reference.onnx"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "models" / "deeponet_reference_fp16.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the frozen FP32 ONNX model to internal FP16"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"FP32 ONNX model does not exist: {args.input}")

    model = onnx.load(args.input)
    converted = convert_float_to_float16(model, keep_io_types=True)
    OnnxModel(converted).topological_sort(is_deterministic=True)
    onnx.checker.check_model(converted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(converted, args.output)
    print(f"FP32 source: {args.input} ({sha256(args.input)})")
    print(f"FP16 model: {args.output} ({sha256(args.output)})")


if __name__ == "__main__":
    main()
