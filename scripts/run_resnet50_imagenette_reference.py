#!/usr/bin/env python3
"""Run the composed learned ResNet-50/ImageNette CPU reference.

The reference consumes the same fixed NCHW tensors as the split TensorRT
runner.  It emits the ordinary ``JDGOUT1`` post-completion container and a
pipeline-compatible CSV, so the accuracy gate can bind every prediction to
the input SHA rather than comparing an independently prepared image list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


INPUT_MAGIC = b"JDGINT1\x00"
INPUT_HEADER = struct.Struct("<IIQ")
INPUT_PREFIX = struct.Struct("<I64s")
OUTPUT_MAGIC = b"JDGOUT1\x00"
OUTPUT_HEADER = struct.Struct("<I")
OUTPUT_SIZE = struct.Struct("<Q")
OUTPUT_ITERATION = struct.Struct("<I")
INPUT_BYTES = 1 * 3 * 224 * 224 * 4
OUTPUT_BYTES = 10 * 4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def read_input_trace(path: Path) -> list[tuple[int, str, np.ndarray]]:
    raw = path.resolve().read_bytes()
    prefix = len(INPUT_MAGIC)
    if len(raw) < prefix + INPUT_HEADER.size or raw[:prefix] != INPUT_MAGIC:
        raise ValueError("producer input trace magic or header differs")
    schema, count, sample_bytes = INPUT_HEADER.unpack_from(raw, prefix)
    if schema != 1 or count <= 0 or sample_bytes != INPUT_BYTES:
        raise ValueError("producer input trace shape differs from ResNet-50")
    offset = prefix + INPUT_HEADER.size
    rows: list[tuple[int, str, np.ndarray]] = []
    for expected in range(count):
        end = offset + INPUT_PREFIX.size + sample_bytes
        if end > len(raw):
            raise ValueError("producer input trace record is truncated")
        iteration, digest_bytes = INPUT_PREFIX.unpack_from(raw, offset)
        if iteration != expected:
            raise ValueError("producer input trace iterations are not dense")
        try:
            digest = digest_bytes.decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("producer input trace hash is not ASCII") from error
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("producer input trace hash is invalid")
        payload = raw[offset + INPUT_PREFIX.size:end]
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"producer input trace payload hash differs at {expected}")
        tensor = np.frombuffer(payload, dtype="<f4").copy().reshape(1, 3, 224, 224)
        if not np.isfinite(tensor).all():
            raise ValueError(f"producer input tensor is non-finite at {expected}")
        rows.append((iteration, digest, tensor))
        offset = end
    if offset != len(raw):
        raise ValueError("producer input trace has trailing bytes")
    return rows


def write_output_trace(path: Path, outputs: list[tuple[int, bytes]]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(OUTPUT_MAGIC)
        stream.write(OUTPUT_HEADER.pack(1))
        stream.write(OUTPUT_SIZE.pack(OUTPUT_BYTES))
        for iteration, payload in outputs:
            if len(payload) != OUTPUT_BYTES:
                raise ValueError("reference output tensor has an unexpected size")
            stream.write(OUTPUT_ITERATION.pack(iteration))
            stream.write(payload)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.warmup < 0 or args.iterations <= 0 or args.deadline_us <= 0:
        raise ValueError("warmup, iterations, or deadline is invalid")
    rows = read_input_trace(args.input_trace)
    expected = args.warmup + args.iterations
    if len(rows) != expected:
        raise ValueError("input trace count differs from warmup plus iterations")
    session = ort.InferenceSession(
        str(args.model.resolve()), providers=["CPUExecutionProvider"]
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("reference model must have one input and one output")
    input_name = inputs[0].name
    output_name = outputs[0].name
    output_records: list[tuple[int, bytes]] = []
    timing: list[dict[str, Any]] = []
    for iteration, digest, tensor in rows:
        start = time.monotonic_ns()
        value = np.asarray(session.run([output_name], {input_name: tensor})[0])
        done = time.monotonic_ns()
        if value.dtype != np.float32 or value.size != 10 or not np.isfinite(value).all():
            raise ValueError(f"reference output differs from FP32 [1,10] at {iteration}")
        payload = np.asarray(value, dtype="<f4").reshape(-1).tobytes()
        wall_us = (done - start) / 1000.0
        output_records.append((iteration, payload))
        timing.append({
            "request": iteration,
            "input_sha256": digest,
            "wall_end_to_end_us": wall_us,
            "deadline_miss": wall_us > args.deadline_us,
        })
    write_output_trace(args.output_trace, output_records)
    args.trace_csv.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.trace_csv.resolve().open("w", encoding="utf-8") as stream:
        stream.write("request,input_sha256,wall_end_to_end_us,deadline_miss\n")
        for row in timing:
            stream.write(
                f"{row['request']},{row['input_sha256']},{row['wall_end_to_end_us']},"
                f"{int(row['deadline_miss'])}\n"
            )
    return {
        "schema_version": 1,
        "status": "ok",
        "task": "imagenette-classification",
        "reference": "resnet50-imagenette-composed-onnx-cpu",
        "model": {"path": str(args.model.resolve()), "sha256": sha256(args.model)},
        "input_trace": {"path": str(args.input_trace.resolve()), "sha256": sha256(args.input_trace)},
        "warmup": args.warmup,
        "iterations": args.iterations,
        "deadline_us": args.deadline_us,
        "deadline_misses": sum(int(row["deadline_miss"]) for row in timing),
        "output_trace": str(args.output_trace.resolve()),
        "trace_csv": str(args.trace_csv.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-trace", type=Path, required=True)
    parser.add_argument("--output-trace", type=Path, required=True)
    parser.add_argument("--trace-csv", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--deadline-us", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
