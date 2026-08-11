#!/usr/bin/env python3
"""Run one balanced-order large-edge transport ablation sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable


TREATMENTS = (
    "cross-mig-registered",
    "cross-mig-pinned",
    "cross-mig-pageable",
    "same-instance-registered",
)


def williams_orders() -> tuple[tuple[str, ...], ...]:
    base = (0, 1, 3, 2)
    return tuple(tuple(TREATMENTS[(item + shift) % 4] for item in base) for shift in range(4))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or not value or key in result:
            raise ValueError("invalid MIG environment")
        result[key] = value
    required = {"JDG_MIG_SMALL_UUID", "JDG_MIG_BIG_UUID", "JDG_MPS_PIPE_DIRECTORY"}
    if not required <= result.keys():
        raise ValueError("MIG environment is incomplete")
    return result


def treatment_args(name: str, mig: dict[str, str]) -> list[str]:
    if name == "cross-mig-registered":
        return ["--consumer", mig["JDG_MIG_BIG_UUID"], "--transport", "registered-direct"]
    if name == "cross-mig-pinned":
        return ["--consumer", mig["JDG_MIG_BIG_UUID"], "--transport", "pinned-bounce"]
    if name == "cross-mig-pageable":
        return ["--consumer", mig["JDG_MIG_BIG_UUID"], "--transport", "pageable-bounce"]
    if name == "same-instance-registered":
        return [
            "--consumer", mig["JDG_MIG_SMALL_UUID"],
            "--consumer-mps-pipe", mig["JDG_MPS_PIPE_DIRECTORY"],
            "--transport", "registered-direct",
        ]
    raise ValueError(f"unknown treatment: {name}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 0 <= args.sequence_index < 4:
        raise ValueError("sequence-index must be in [0, 3]")
    mig = load_env(args.mig_env)
    order = williams_orders()[args.sequence_index]
    args.result_dir.mkdir(parents=True, exist_ok=False)
    (args.result_dir / "mig.env").write_bytes(args.mig_env.read_bytes())
    environment = os.environ.copy()
    environment.update(mig)
    rows: list[dict[str, Any]] = []
    for position, treatment in enumerate(order):
        directory = args.result_dir / f"{position + 1:02d}-{treatment}"
        directory.mkdir()
        trace = directory / "pipeline.csv"
        output_trace = directory / "application-outputs.bin"
        command = [
            "taskset", "--cpu-list", str(args.control_cpu), str(args.benchmark),
            "--producer-engine", str(args.producer_engine),
            *(["--consumer-engine", str(args.consumer_engine)] if args.consumer_engine else []),
            "--consumer-input-tensor", args.consumer_input_tensor,
            "--producer", mig["JDG_MIG_SMALL_UUID"],
            "--producer-mps-pipe", mig["JDG_MPS_PIPE_DIRECTORY"],
            "--workload", "whisper-projection",
            "--deadline-mode", "wall",
            "--warmup", str(args.warmup), "--iterations", str(args.iterations),
            "--trace-csv", str(trace),
            "--application-output-trace", str(output_trace),
            *treatment_args(treatment, mig),
        ]
        completed = subprocess.run(
            command, cwd=args.repo, env=environment, check=True,
            capture_output=True, text=True, timeout=args.timeout_seconds,
        )
        result = json.loads(completed.stdout)
        if (
            result.get("status") != "ok"
            or result.get("payload_bytes") != 2_304_000
            or result.get("checksum_failures") != 0
            or result.get("iterations") != args.iterations
        ):
            raise RuntimeError(f"invalid result for {treatment}")
        if not output_trace.is_file() or output_trace.stat().st_size <= 8:
            raise RuntimeError(f"missing application output trace for {treatment}")
        result_path = directory / "pipeline.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        (directory / "pipeline.stderr").write_text(completed.stderr, encoding="utf-8")
        rows.append({
            "treatment": treatment,
            "position": position,
            "result": {"path": str(result_path.resolve()), "sha256": sha256(result_path)},
            "trace": {"path": str(trace.resolve()), "sha256": sha256(trace)},
            "application_output_trace": {
                "path": str(output_trace.resolve()),
                "sha256": sha256(output_trace),
                "capture_boundary": "post-completion",
            },
        })
    output = {
        "schema_version": 1,
        "kind": "p9-whisper-transport-williams-sequence",
        "sequence_index": args.sequence_index,
        "execution_order": list(order),
        "iterations_per_treatment": args.iterations,
        "payload_bytes": 2_304_000,
        "workload": "whisper-last-hidden-state-to-projection-mlp",
        "production_wall_definition": "arrival-to-consumer-completion-excludes-correctness-validation",
        "correctness_validation_placement": "post-completion",
        "artifacts": {
            "benchmark": {"path": str(args.benchmark.resolve()), "sha256": sha256(args.benchmark)},
            "producer_engine": {"path": str(args.producer_engine.resolve()), "sha256": sha256(args.producer_engine)},
            "consumer_engine": (
                {"path": str(args.consumer_engine.resolve()), "sha256": sha256(args.consumer_engine)}
                if args.consumer_engine else None
            ),
            "consumer_input_tensor": args.consumer_input_tensor,
            "mig_env": {"path": str(args.mig_env.resolve()), "sha256": sha256(args.mig_env)},
        },
        "runs": rows,
    }
    summary = args.result_dir / "summary.json"
    summary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def main(argv: Iterable[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=root)
    parser.add_argument("--mig-env", type=Path, default=Path("/tmp/jdg-mps-1g/mig.env"))
    parser.add_argument("--benchmark", type=Path, default=root / "build-r39/jdg-mig-trt-pipeline")
    parser.add_argument("--producer-engine", type=Path, default=root / "models/engines/mig-1g-q100/whisper-tiny-encoder.engine")
    parser.add_argument("--consumer-engine", type=Path,
                        help="external trained downstream TensorRT engine")
    parser.add_argument("--consumer-input-tensor", default="features")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--sequence-index", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--control-cpu", type=int, default=13)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    args = parser.parse_args(argv)
    if args.iterations <= 0 or args.warmup < 0 or args.timeout_seconds <= 0:
        parser.error("iterations/timeout must be positive and warmup nonnegative")
    args.repo = args.repo.resolve()
    args.mig_env = args.mig_env.resolve()
    args.benchmark = args.benchmark.resolve()
    args.producer_engine = args.producer_engine.resolve()
    if args.consumer_engine is not None:
        args.consumer_engine = args.consumer_engine.resolve()
        if not args.consumer_engine.is_file():
            parser.error("--consumer-engine must be a regular file")
    args.result_dir = args.result_dir.resolve()
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
