#!/usr/bin/env python3
"""Collect independent dependent-pipeline blocks for deadline calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import time
from typing import Any


PLACEMENTS = {
    "fixed-1g-producer-2g-consumer": {
        "producer_profile": "mig-1g-q100",
        "consumer_profile": "mig-2g-q100",
        "producer_role": "1g",
        "consumer_role": "2g",
    },
    "fixed-2g-producer-1g-consumer": {
        "producer_profile": "mig-2g-q100",
        "consumer_profile": "mig-1g-q100",
        "producer_role": "2g",
        "consumer_role": "1g",
    },
}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_env(path: pathlib.Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or not value or key in result:
            raise ValueError("invalid MIG environment")
        result[key] = value
    for key in ("JDG_MIG_SMALL_UUID", "JDG_MIG_BIG_UUID", "JDG_MPS_PIPE_DIRECTORY"):
        if key not in result:
            raise ValueError(f"MIG environment lacks {key}")
    return result


def process_state(pid: int) -> str:
    text = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    end = text.rfind(")")
    if end < 0 or end + 2 >= len(text):
        raise RuntimeError("malformed process state")
    return text[end + 2]


def wait_paused(process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"background exited {process.returncode}: {stdout} {stderr}"
            )
        if process_state(process.pid) in {"T", "t"}:
            return
        time.sleep(0.02)
    raise TimeoutError("background start barrier timed out")


def stop_background(process: subprocess.Popen[str]) -> tuple[str, str]:
    if process.poll() is None:
        for number in (signal.SIGCONT, signal.SIGINT):
            try:
                os.kill(process.pid, number)
            except ProcessLookupError:
                break
    try:
        return process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.communicate()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--mig-env", type=pathlib.Path, default=pathlib.Path("/tmp/jdg-mps-1g/mig.env"))
    parser.add_argument("--result-dir", type=pathlib.Path, required=True)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--slo-factor", type=float, default=1.10)
    parser.add_argument(
        "--workload",
        choices=("whisper-projection", "resnet-control", "resnet-detection-head", "resnet50-classification"),
        default="whisper-projection",
    )
    parser.add_argument(
        "--consumer-engine",
        type=pathlib.Path,
        help="external trained downstream TensorRT engine; omission uses generated control policy",
    )
    parser.add_argument(
        "--producer-engine", type=pathlib.Path,
        help="explicit producer TensorRT engine; omission uses the workload default",
    )
    parser.add_argument(
        "--producer-input-trace", type=pathlib.Path,
        help="optional JDGINT1 trace shared by every calibration block",
    )
    parser.add_argument(
        "--operational-arrival-trace", type=pathlib.Path,
        help="optional JDGARR1 trace shared by every calibration block",
    )
    parser.add_argument(
        "--consumer-input-tensor", default="features",
        help="input tensor name for --consumer-engine",
    )
    parser.add_argument(
        "--placement-variant",
        choices=tuple(PLACEMENTS),
        default="fixed-1g-producer-2g-consumer",
        help="fixed producer/consumer MIG direction to calibrate",
    )
    parser.add_argument(
        "--background-period-ms", type=float, default=0.0,
        help="best-effort pressure period; zero preserves isolated calibration",
    )
    parser.add_argument(
        "--background-quota", type=int, default=100,
        choices=(10, 25, 50, 75, 90, 100),
        help="MPS quota for the recorded best-effort pressure engine",
    )
    args = parser.parse_args()
    if args.blocks < 2 or args.samples < 100 or args.warmup < 0 or args.slo_factor != 1.10:
        raise ValueError("calibration requires >=2 blocks, >=100 samples, and factor 1.10")
    if args.workload == "resnet50-classification" and args.placement_variant != "fixed-1g-producer-2g-consumer":
        raise ValueError("resnet50-classification currently requires the profiled 1g-producer/2g-consumer split")
    if args.workload == "resnet50-classification" and args.consumer_engine is None:
        raise ValueError("resnet50-classification requires --consumer-engine")
    if args.operational_arrival_trace is not None and args.producer_input_trace is None:
        raise ValueError("operational arrival trace requires a producer input trace")
    if args.background_period_ms < 0.0:
        raise ValueError("background period must be nonnegative")
    if args.workload == "resnet50-classification" and args.consumer_input_tensor == "features":
        args.consumer_input_tensor = "gpu_0/res4_5_branch2c_bn_2"

    repo = args.repo.resolve()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=False)
    mig = load_env(args.mig_env)
    workload = {
        "whisper-projection": {
            "model": "whisper-tiny-encoder",
            "payload_bytes": 2304000,
            "deadline_mode": "wall",
        },
        "resnet-control": {
            "model": "resnet10-detection",
            "payload_bytes": 14720,
            "deadline_mode": "wall",
        },
        "resnet-detection-head": {
            "model": "resnet10-backbone",
            "payload_bytes": 1884160,
            "deadline_mode": "wall",
        },
        "resnet50-classification": {
            "model": "resnet50-backbone",
            "payload_bytes": 802816,
            "deadline_mode": "wall",
        },
    }[args.workload]
    placement = PLACEMENTS[args.placement_variant]
    engine = (
        args.producer_engine.resolve() if args.producer_engine is not None
        else repo / f"models/engines/{placement['producer_profile']}/{workload['model']}.engine"
    )
    if not engine.is_file():
        raise ValueError(f"producer engine is not a regular file: {engine}")
    background_engine = (
        repo / f"models/engines/mig-1g-q{args.background_quota}/distilbert-sst2.engine"
    )
    if args.background_period_ms > 0.0 and not background_engine.is_file():
        raise ValueError(f"background engine is not a regular file: {background_engine}")
    consumer_engine = args.consumer_engine.resolve() if args.consumer_engine else None
    if consumer_engine is not None and not consumer_engine.is_file():
        raise ValueError(f"--consumer-engine is not a regular file: {consumer_engine}")
    producer_input_trace = args.producer_input_trace.resolve() if args.producer_input_trace else None
    operational_arrival_trace = (
        args.operational_arrival_trace.resolve() if args.operational_arrival_trace else None
    )
    if producer_input_trace is not None and not producer_input_trace.is_file():
        raise ValueError("--producer-input-trace is not a regular file")
    if operational_arrival_trace is not None and not operational_arrival_trace.is_file():
        raise ValueError("--operational-arrival-trace is not a regular file")
    binary = repo / "build-r39/jdg-mig-trt-pipeline"
    source = repo / "benchmarks/mig_trt_pipeline.cpp"
    blocks: list[dict[str, Any]] = []
    for index in range(args.blocks):
        block_dir = result_dir / f"block-{index:02d}"
        block_dir.mkdir()
        trace = block_dir / "pipeline.csv"
        command = [
                "taskset", "--cpu-list", "13", str(binary),
        "--producer-engine", str(engine),
        *(["--consumer-engine", str(consumer_engine)] if consumer_engine else []),
        "--consumer-input-tensor", args.consumer_input_tensor,
                "--producer", mig[
                    "JDG_MIG_SMALL_UUID"
                    if args.placement_variant == "fixed-1g-producer-2g-consumer"
                    else "JDG_MIG_BIG_UUID"
                ],
                "--consumer", mig[
                    "JDG_MIG_BIG_UUID"
                    if args.placement_variant == "fixed-1g-producer-2g-consumer"
                    else "JDG_MIG_SMALL_UUID"
                ],
                "--producer-quota", "100",
                "--transport", "registered-direct",
                "--workload", args.workload,
                "--deadline-mode", str(workload["deadline_mode"]),
                "--warmup", str(args.warmup),
                "--iterations", str(args.samples),
                "--trace-csv", str(trace),
            ]
        if args.placement_variant == "fixed-1g-producer-2g-consumer":
            command.extend(("--producer-mps-pipe", mig["JDG_MPS_PIPE_DIRECTORY"]))
        else:
            command.extend(("--consumer-mps-pipe", mig["JDG_MPS_PIPE_DIRECTORY"]))
        if producer_input_trace is not None:
            command.extend(("--producer-input-trace", str(producer_input_trace)))
        if operational_arrival_trace is not None:
            command.extend(("--arrival-trace", str(operational_arrival_trace)))
        background: subprocess.Popen[str] | None = None
        background_stdout, background_stderr = "", ""
        try:
            if args.background_period_ms > 0.0:
                background_env = dict(
                    os.environ,
                    **mig,
                    CUDA_VISIBLE_DEVICES=mig["JDG_MIG_SMALL_UUID"],
                    CUDA_MPS_PIPE_DIRECTORY=mig["JDG_MPS_PIPE_DIRECTORY"],
                    CUDA_MPS_LOG_DIRECTORY=mig.get(
                        "JDG_MPS_LOG_DIRECTORY", "/tmp/jdg-mps-1g/log"
                    ),
                    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(args.background_quota),
                )
                background = subprocess.Popen(
                    [
                        "taskset", "--cpu-list", "0", str(repo / "build-r39/jdg-trt-bench"),
                        "--engine", str(background_engine),
                        "--model-name", "distilbert-sst2", "--role", "pressure",
                        "--duration-seconds", "3600", "--burst-size", "1",
                        "--period-ms", str(args.background_period_ms),
                        "--warmup", "20", "--include-transfers", "true",
                        "--priority", "default", "--start-paused", "true",
                    ],
                    cwd=repo,
                    env=background_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                wait_paused(background)
                os.kill(background.pid, signal.SIGCONT)
            try:
                completed = subprocess.run(
                    command,
                    cwd=repo,
                    env=dict(os.environ, **mig),
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
            except subprocess.CalledProcessError as error:
                (block_dir / "failure.json").write_text(
                    json.dumps(
                        {
                            "returncode": error.returncode,
                            "stdout": error.stdout,
                            "stderr": error.stderr,
                            "command": command,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                raise
        finally:
            if background is not None:
                background_stdout, background_stderr = stop_background(background)
        pipeline = json.loads(completed.stdout)
        if pipeline.get("status") != "ok" or pipeline.get("checksum_failures") != 0:
            raise RuntimeError("isolated pipeline calibration failed correctness")
        result = block_dir / "pipeline.json"
        result.write_text(json.dumps(pipeline, indent=2) + "\n", encoding="utf-8")
        (block_dir / "pipeline.stderr").write_text(completed.stderr, encoding="utf-8")
        (block_dir / "background.stdout").write_text(background_stdout, encoding="utf-8")
        (block_dir / "background.stderr").write_text(background_stderr, encoding="utf-8")
        blocks.append(
            {
                "index": index,
                "result_path": str(result.relative_to(result_dir)),
                "result_sha256": sha256(result),
                "trace_path": str(trace.relative_to(result_dir)),
                "trace_sha256": sha256(trace),
            }
        )

    summary = {
        "schema_version": 1,
        "kind": "p9-dependent-pipeline-deadline-calibration",
        "config": {
            "workload": args.workload,
            "placement_variant": args.placement_variant,
            "producer_profile": placement["producer_profile"],
            "consumer_profile": placement["consumer_profile"],
            "producer_role": placement["producer_role"],
            "consumer_role": placement["consumer_role"],
            "payload_bytes": workload["payload_bytes"],
            "transport": "registered-shared-sysmem-direct-binding",
            "deadline_mode": workload["deadline_mode"],
            "production_wall_definition": (
                "arrival-to-consumer-completion-excludes-correctness-validation"
                if workload["deadline_mode"] == "wall"
                else "legacy-validation-excluded"
            ),
            "correctness_validation_placement": "post-completion",
            "blocks": args.blocks,
            "samples_per_block": args.samples,
            "warmup": args.warmup,
            "slo_factor": args.slo_factor,
            "producer_quota_percent": 100,
            "background_period_ms": args.background_period_ms,
            "background_quota_percent": args.background_quota,
            "consumer_engine_mode": (
                "external-trained-engine" if consumer_engine is not None
                else "generated-control-policy"
            ),
            "consumer_input_tensor": args.consumer_input_tensor,
            "producer_input_trace": (
                {"path": str(producer_input_trace), "sha256": sha256(producer_input_trace)}
                if producer_input_trace is not None else None
            ),
            "operational_arrival_trace": (
                {"path": str(operational_arrival_trace), "sha256": sha256(operational_arrival_trace)}
                if operational_arrival_trace is not None else None
            ),
            "producer_uuid": mig[
                "JDG_MIG_SMALL_UUID"
                if args.placement_variant == "fixed-1g-producer-2g-consumer"
                else "JDG_MIG_BIG_UUID"
            ],
            "consumer_uuid": mig[
                "JDG_MIG_BIG_UUID"
                if args.placement_variant == "fixed-1g-producer-2g-consumer"
                else "JDG_MIG_SMALL_UUID"
            ],
            "cpu": 13,
        },
        "artifacts": {
            "binary": {"path": str(binary), "sha256": sha256(binary)},
            "source": {"path": str(source), "sha256": sha256(source)},
            "engine": {"path": str(engine), "sha256": sha256(engine)},
            "consumer_engine": (
                {"path": str(consumer_engine), "sha256": sha256(consumer_engine)}
                if consumer_engine is not None else None
            ),
            "background_engine": (
                {"path": str(background_engine), "sha256": sha256(background_engine)}
                if args.background_period_ms > 0.0 else None
            ),
            "producer_input_trace": (
                {"path": str(producer_input_trace), "sha256": sha256(producer_input_trace)}
                if producer_input_trace is not None else None
            ),
            "operational_arrival_trace": (
                {"path": str(operational_arrival_trace), "sha256": sha256(operational_arrival_trace)}
                if operational_arrival_trace is not None else None
            ),
        },
        "blocks": blocks,
    }
    output = result_dir / "calibration.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
