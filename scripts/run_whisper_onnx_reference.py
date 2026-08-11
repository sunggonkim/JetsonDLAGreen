#!/usr/bin/env python3
"""Run the pinned Whisper ONNX reference on the same prepared sample list.

This is a real application reference, not a transport projection.  It uses
the same mel features, prompt, greedy token policy, and post-completion
JDGASR1 container as the split TensorRT runner so the formal gate can bind
inputs, outputs, and timing without translating between output formats.
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


EOS = 50257
PROMPT = np.asarray([[50258, 50259, 50359, 50363]], dtype=np.int64)
MAX_TOKENS = 128


def sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def read_samples(path: Path) -> list[dict[str, Any]]:
    raw = path.resolve().read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("sample list is not newline-complete")
    rows = [json.loads(line) for line in raw.splitlines()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("sample list is invalid")
    if [row.get("iteration") for row in rows] != list(range(len(rows))):
        raise ValueError("sample list iterations are not dense")
    for row in rows:
        if not isinstance(row.get("path"), str) or not Path(row["path"]).is_file():
            raise ValueError("sample feature path is missing")
        if not isinstance(row.get("input_sha256"), str) or len(row["input_sha256"]) != 64:
            raise ValueError("sample input hash is invalid")
    return rows


def _greedy_decode(
    initial: ort.InferenceSession,
    with_past: ort.InferenceSession,
    hidden: np.ndarray,
    max_tokens: int,
) -> list[int]:
    outputs = initial.run(None, {
        "input_ids": PROMPT,
        "encoder_hidden_states": hidden,
    })
    initial_values = dict(zip((item.name for item in initial.get_outputs()), outputs))
    logits = np.asarray(initial_values["logits"])
    token = int(np.argmax(logits[0, -1]))
    tokens: list[int] = []
    if token != EOS:
        tokens.append(token)
    past_values = initial_values
    encoder_values = initial_values
    while token != EOS and len(tokens) < max_tokens:
        inputs: dict[str, np.ndarray] = {"input_ids": np.asarray([[token]], dtype=np.int64)}
        for layer in range(4):
            inputs[f"past_key_values.{layer}.decoder.key"] = past_values[
                f"present.{layer}.decoder.key"
            ]
            inputs[f"past_key_values.{layer}.decoder.value"] = past_values[
                f"present.{layer}.decoder.value"
            ]
            inputs[f"past_key_values.{layer}.encoder.key"] = encoder_values[
                f"present.{layer}.encoder.key"
            ]
            inputs[f"past_key_values.{layer}.encoder.value"] = encoder_values[
                f"present.{layer}.encoder.value"
            ]
        outputs = with_past.run(None, inputs)
        past_values = dict(zip((item.name for item in with_past.get_outputs()), outputs))
        token = int(np.argmax(np.asarray(past_values["logits"])[0, -1]))
        if token != EOS:
            tokens.append(token)
    if len(tokens) > max_tokens:
        raise ValueError("reference token output exceeds max_tokens")
    return tokens


def write_trace(path: Path, records: list[tuple[int, list[int]]]) -> None:
    output = bytearray(b"JDGASR1\x00")
    output += struct.pack("<II", 1, len(records))
    for iteration, tokens in records:
        output += struct.pack("<II", iteration, len(tokens))
        output += struct.pack("<" + "I" * len(tokens), *tokens)
    path.resolve().parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(output)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.warmup < 0 or args.iterations <= 0 or args.max_tokens <= 0 or args.max_tokens > MAX_TOKENS:
        raise ValueError("warmup, iterations, or max_tokens is invalid")
    samples = read_samples(args.sample_list)
    total = args.warmup + args.iterations
    if len(samples) != total:
        raise ValueError("sample count differs from warmup plus iterations")
    encoder = ort.InferenceSession(str(args.encoder.resolve()), providers=["CPUExecutionProvider"])
    initial = ort.InferenceSession(str(args.decoder_initial.resolve()), providers=["CPUExecutionProvider"])
    with_past = ort.InferenceSession(str(args.decoder_with_past.resolve()), providers=["CPUExecutionProvider"])
    records: list[tuple[int, list[int]]] = []
    timing: list[dict[str, Any]] = []
    for row in samples:
        features = np.fromfile(row["path"], dtype=np.float32)
        if features.size != 80 * 3000 or not np.isfinite(features).all():
            raise ValueError("prepared feature tensor is invalid")
        start = time.monotonic_ns()
        hidden = encoder.run(["last_hidden_state"], {"input_features": features.reshape(1, 80, 3000)})[0]
        tokens = _greedy_decode(initial, with_past, hidden, args.max_tokens)
        done = time.monotonic_ns()
        wall_us = (done - start) / 1000.0
        records.append((int(row["iteration"]), tokens))
        timing.append({
            "request": int(row["iteration"]),
            "input_sha256": row["input_sha256"],
            "wall_end_to_end_us": wall_us,
            "deadline_miss": wall_us > args.deadline_us,
        })
    write_trace(args.output_trace, records)
    args.trace_csv.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.trace_csv.resolve().open("w", encoding="utf-8") as stream:
        stream.write("request,input_sha256,wall_end_to_end_us,deadline_miss\n")
        for row in timing:
            stream.write(
                f"{row['request']},{row['input_sha256']},{row['wall_end_to_end_us']},"
                f"{int(row['deadline_miss'])}\n"
            )
    result = {
        "schema_version": 1,
        "status": "ok",
        "task": "asr",
        "reference": "whisper-tiny-onnx-cpu",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "max_tokens": args.max_tokens,
        "deadline_us": args.deadline_us,
        "deadline_misses": sum(int(row["deadline_miss"]) for row in timing),
        "encoder": {"path": str(args.encoder.resolve()), "sha256": sha256(args.encoder)},
        "decoder_initial": {"path": str(args.decoder_initial.resolve()), "sha256": sha256(args.decoder_initial)},
        "decoder_with_past": {"path": str(args.decoder_with_past.resolve()), "sha256": sha256(args.decoder_with_past)},
        "output_trace": str(args.output_trace.resolve()),
        "trace_csv": str(args.trace_csv.resolve()),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder-initial", type=Path, required=True)
    parser.add_argument("--decoder-with-past", type=Path, required=True)
    parser.add_argument("--sample-list", type=Path, required=True)
    parser.add_argument("--output-trace", type=Path, required=True)
    parser.add_argument("--trace-csv", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--deadline-us", type=float, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
