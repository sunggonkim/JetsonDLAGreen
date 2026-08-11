#!/usr/bin/env python3
"""Profile one-in-flight TensorRT drain envelopes for QUIET modalities."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
from typing import Any


MODELS = {
    "language": "distilbert-sst2",
    "audio": "whisper-tiny-encoder",
}


def envelope_ms(p999_values: list[float]) -> float:
    if not p999_values or any(value <= 0.0 or not math.isfinite(value) for value in p999_values):
        raise ValueError("drain samples must be finite and positive")
    return float(math.ceil(max(p999_values)))


def run_profile(
    bench: pathlib.Path,
    engine: pathlib.Path,
    model: str,
    samples: int,
    warmup: int,
    cpu: str,
    environment: dict[str, str],
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "taskset",
            "--cpu-list",
            cpu,
            str(bench),
            "--engine",
            str(engine),
            "--model-name",
            model,
            "--role",
            "benchmark",
            "--samples",
            str(samples),
            "--warmup",
            str(warmup),
            "--include-transfers",
            "true",
            "--priority",
            "default",
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120.0,
    )
    result = json.loads(completed.stdout)
    if result.get("schema_version") != 1 or result.get("model") != model:
        raise RuntimeError(f"invalid drain profile result for {model}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=pathlib.Path, required=True)
    parser.add_argument("--engine-root", type=pathlib.Path, required=True)
    parser.add_argument("--mps-pipe", type=pathlib.Path, required=True)
    parser.add_argument("--mps-log", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--quota", type=int, default=25)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu", default="11")
    args = parser.parse_args()
    if (
        not args.bench.is_file()
        or args.quota <= 0
        or args.quota > 100
        or args.samples <= 0
        or args.warmup < 0
        or args.repeats <= 0
    ):
        raise SystemExit("invalid drain profiler configuration")

    environment = os.environ.copy() | {
        "CUDA_VISIBLE_DEVICES": "0",
        "CUDA_MPS_PIPE_DIRECTORY": str(args.mps_pipe),
        "CUDA_MPS_LOG_DIRECTORY": str(args.mps_log),
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(args.quota),
    }
    environment["LD_LIBRARY_PATH"] = "/usr/local/cuda-13.2/lib64:" + environment.get(
        "LD_LIBRARY_PATH", ""
    )
    modalities: dict[str, Any] = {}
    for modality, model in MODELS.items():
        engine = args.engine_root / f"full-q{args.quota}" / f"{model}.engine"
        if not engine.is_file():
            raise SystemExit(f"missing TensorRT engine: {engine}")
        trials = [
            run_profile(
                args.bench,
                engine,
                model,
                args.samples,
                args.warmup,
                args.cpu,
                environment,
            )
            for _ in range(args.repeats)
        ]
        p999 = [float(trial["gpu_service"]["p999_ms"]) for trial in trials]
        modalities[modality] = {
            "model": model,
            "gpu_service_p999_ms": p999,
            "envelope_ms": envelope_ms(p999),
            "trials": trials,
        }

    output = {
        "schema_version": 1,
        "config": {
            "quota_percent": args.quota,
            "samples": args.samples,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "includes_transfers": True,
            "outstanding_depth": 1,
        },
        "modalities": modalities,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
