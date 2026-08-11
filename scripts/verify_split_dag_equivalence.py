#!/usr/bin/env python3
"""Check that a learned ONNX DAG split preserves the full-model output.

This is a graph/data-path equivalence check. It is intentionally separate from
task accuracy: a real dataset label manifest and reference/candidate
production traces are still required for an application claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _static_shape(value: Any) -> tuple[int, ...] | None:
    """Return a concrete ONNX shape, or None when a dimension is dynamic."""
    if not isinstance(value, (list, tuple)) or not value:
        return None
    dimensions: list[int] = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            return None
        dimensions.append(dimension)
    return tuple(dimensions)


def _infer_input_shape(value: Any) -> tuple[int, ...] | None:
    """Infer a concrete sample shape when only the batch dimension is dynamic."""
    if not isinstance(value, (list, tuple)) or not value:
        return None
    dimensions: list[int] = []
    for index, dimension in enumerate(value):
        if isinstance(dimension, int) and not isinstance(dimension, bool) and dimension > 0:
            dimensions.append(dimension)
        elif index == 0 and dimension in (None, "batch", "batch_size"):
            dimensions.append(1)
        else:
            return None
    return tuple(dimensions)


def verify(full_model: Path, producer_model: Path, head_model: Path,
           input_npy: Path, *, atol: float = 1e-5, rtol: float = 1e-5,
           input_shape: tuple[int, ...] | None = None) -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("split equivalence requires onnxruntime") from error
    paths = {"full_model": full_model, "producer_model": producer_model,
             "head_model": head_model, "input": input_npy}
    for name, path in paths.items():
        if not path.resolve().is_file():
            raise ValueError(f"{name} is not a regular file: {path}")
    for name, value in (("atol", atol), ("rtol", rtol)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    full_session = ort.InferenceSession(str(full_model.resolve()), providers=["CPUExecutionProvider"])
    producer_session = ort.InferenceSession(str(producer_model.resolve()), providers=["CPUExecutionProvider"])
    head_session = ort.InferenceSession(str(head_model.resolve()), providers=["CPUExecutionProvider"])
    full_input = full_session.get_inputs()[0].name
    producer_input = producer_session.get_inputs()[0].name
    full_raw_shape = full_session.get_inputs()[0].shape
    producer_raw_shape = producer_session.get_inputs()[0].shape
    full_expected = _infer_input_shape(full_raw_shape)
    producer_expected = _infer_input_shape(producer_raw_shape)
    if full_expected is not None and producer_expected is not None and full_expected != producer_expected:
        raise ValueError("full and producer input shapes differ")
    expected_shape = input_shape or full_expected or producer_expected
    if expected_shape is None:
        raise ValueError("dynamic model input requires --input-shape")
    sample = np.load(input_npy.resolve(), allow_pickle=False)
    if sample.dtype != np.float32 or tuple(sample.shape) != tuple(expected_shape):
        shape_text = ",".join(str(value) for value in expected_shape)
        raise ValueError(f"input must be float32 with shape [{shape_text}]")
    producer_output = producer_session.get_outputs()[0].name
    head_input = head_session.get_inputs()[0].name
    full_output = full_session.get_outputs()[0].name
    full_value = full_session.run([full_output], {full_input: sample})[0]
    activation = producer_session.run([producer_output], {producer_input: sample})[0]
    split_value = head_session.run([head_session.get_outputs()[0].name], {head_input: activation})[0]
    if full_value.shape != split_value.shape:
        raise ValueError("full and split outputs have different shapes")
    difference = np.abs(full_value - split_value)
    max_abs = float(np.max(difference))
    mean_abs = float(np.mean(difference))
    if not np.allclose(full_value, split_value, rtol=float(rtol), atol=float(atol)):
        raise ValueError(f"split output differs: max_abs={max_abs}")
    return {
        "schema_version": 1,
        "kind": "p9-learned-dag-split-equivalence",
        "status": "passed",
        "task_accuracy_claim": False,
        "full_model_sha256": sha256(full_model.resolve()),
        "producer_model_sha256": sha256(producer_model.resolve()),
        "head_model_sha256": sha256(head_model.resolve()),
        "input_path": str(input_npy.resolve()),
        "input_sha256": sha256(input_npy.resolve()),
        "input_shape": list(sample.shape),
        "output_shape": list(split_value.shape),
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "argmax_match": bool(np.argmax(full_value) == np.argmax(split_value)),
        "atol": float(atol),
        "rtol": float(rtol),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-model", type=Path, required=True)
    parser.add_argument("--producer-model", type=Path, required=True)
    parser.add_argument("--head-model", type=Path, required=True)
    parser.add_argument("--input-npy", type=Path, required=True)
    parser.add_argument(
        "--input-shape",
        type=lambda value: tuple(int(item) for item in value.split(",") if item),
        help="explicit NCHW shape for models with dynamic input dimensions",
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.full_model, args.producer_model, args.head_model,
                    args.input_npy, atol=args.atol, rtol=args.rtol,
                    input_shape=args.input_shape)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
