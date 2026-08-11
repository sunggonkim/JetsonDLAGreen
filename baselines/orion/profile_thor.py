#!/usr/bin/env python3
"""Regenerate Orion per-operation profiles on a fixed Thor MIG instance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable


UPSTREAM_COMMIT = "20f9469764fb96d94ce23a8e70615196e9ce4ba1"
MODES = ("isolated", "compute", "memory")
CLASSIFICATION_MARGIN = 0.05


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or not value or key in values:
            raise ValueError("invalid MIG environment file")
        values[key] = value
    required = {
        "JDG_MIG_SMALL_UUID",
        "JDG_MPS_PIPE_DIRECTORY",
        "JDG_MPS_LOG_DIRECTORY",
    }
    if not required <= values.keys():
        raise ValueError("MIG environment lacks the fixed 1g MPS contract")
    return values


def load_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{path} is not newline complete")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{path} is not newline complete")
    rows = [json.loads(line) for line in raw.splitlines()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} has no operation records")
    return rows


def signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("api"),
        tuple(row.get("grid", ())),
        tuple(row.get("block", ())),
        row.get("shared_mem_bytes"),
    )


def replay_mode(
    result_path: Path,
    trace_path: Path,
    *,
    model: str,
    samples: int,
    warmup: int,
) -> dict[str, Any]:
    result = load_json(result_path)
    if (
        result.get("kind") != "orion-thor-operation-profile-raw"
        or result.get("upstream_commit") != UPSTREAM_COMMIT
        or result.get("model") != model
        or result.get("samples") != samples
        or result.get("warmup") != warmup
        or result.get("numeric_comparison_allowed") is not False
        or result.get("client", {}).get("completed_requests") != samples
        or result.get("client", {}).get("gpu", {}).get("multiprocessors") != 8
    ):
        raise ValueError("Orion raw profile result differs from the requested contract")
    rows = load_jsonl(trace_path)
    total_requests = warmup + samples
    if len(rows) % total_requests != 0:
        raise ValueError("operation count is not constant per inference")
    per_request = len(rows) // total_requests
    if per_request <= 0:
        raise ValueError("profile has no operations per inference")
    expected_keys = {
        "schema_version", "client_id", "operation_index", "api", "grid",
        "block", "grid_blocks", "block_threads", "shared_mem_bytes",
        "active_blocks_per_sm", "device_sms", "estimated_sms",
        "kernel_duration_us",
    }
    for index, row in enumerate(rows):
        if set(row) != expected_keys or row["schema_version"] != 1:
            raise ValueError(f"operation {index} schema differs")
        if row["client_id"] != 1 or row["operation_index"] != index:
            raise ValueError(f"operation {index} identity differs")
        for key in ("kernel_duration_us",):
            value = row[key]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"operation {index} has invalid {key}")
        for key in (
            "grid_blocks", "block_threads", "active_blocks_per_sm",
            "device_sms", "estimated_sms",
        ):
            if not isinstance(row[key], int) or isinstance(row[key], bool) or row[key] <= 0:
                raise ValueError(f"operation {index} has invalid {key}")
        if row["device_sms"] != 8 or row["estimated_sms"] > 8:
            raise ValueError(f"operation {index} used the wrong MIG width")

    measured = rows[warmup * per_request :]
    positions: list[dict[str, Any]] = []
    for position in range(per_request):
        instances = measured[position::per_request]
        signatures = {signature(row) for row in instances}
        if len(instances) != samples or len(signatures) != 1:
            raise ValueError(f"operation position {position} is not repeatable")
        positions.append(
            {
                "position": position,
                "signature": instances[0],
                "duration_us": statistics.median(
                    float(row["kernel_duration_us"]) for row in instances
                ),
                "estimated_sms": max(row["estimated_sms"] for row in instances),
            }
        )
    return {
        "operations_per_inference": per_request,
        "positions": positions,
        "result_sha256": sha256(result_path),
        "trace_sha256": sha256(trace_path),
    }


def classify_profiles(replays: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if set(replays) != set(MODES):
        raise ValueError("all Orion profiling modes are required")
    count = replays["isolated"]["operations_per_inference"]
    if any(replay["operations_per_inference"] != count for replay in replays.values()):
        raise ValueError("profiling modes expose different operation counts")
    profiles: list[dict[str, Any]] = []
    for position in range(count):
        rows = {mode: replays[mode]["positions"][position] for mode in MODES}
        signatures = {signature(row["signature"]) for row in rows.values()}
        if len(signatures) != 1:
            raise ValueError(f"operation {position} signature differs across modes")
        isolated = rows["isolated"]["duration_us"]
        compute_slowdown = rows["compute"]["duration_us"] / isolated
        memory_slowdown = rows["memory"]["duration_us"] / isolated
        if compute_slowdown > memory_slowdown * (1.0 + CLASSIFICATION_MARGIN):
            resource_class = "compute"
            upstream_profile = 1
        elif memory_slowdown > compute_slowdown * (1.0 + CLASSIFICATION_MARGIN):
            resource_class = "memory"
            upstream_profile = 0
        else:
            resource_class = "unclear"
            upstream_profile = -1
        source = rows["isolated"]["signature"]
        profiles.append(
            {
                "position": position,
                "api": source["api"],
                "grid": source["grid"],
                "block": source["block"],
                "shared_mem_bytes": source["shared_mem_bytes"],
                "profile": upstream_profile,
                "resource_class": resource_class,
                "sm_used": max(row["estimated_sms"] for row in rows.values()),
                "duration_us": isolated,
                "compute_pressure_slowdown": compute_slowdown,
                "memory_pressure_slowdown": memory_slowdown,
            }
        )
    return profiles


def wait_ready(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if path.is_file() and path.read_text(encoding="utf-8") == "ready\n":
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(f"Orion pressure exited early: {stdout} {stderr}")
        time.sleep(0.02)
    raise TimeoutError("Orion pressure readiness timed out")


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise ValueError("refusing an existing Orion profile output directory")
    if args.samples <= 0 or args.warmup < 1:
        raise ValueError("positive samples and at least one warmup are required")
    mig = load_env(args.mig_env)
    args.output.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": mig["JDG_MIG_SMALL_UUID"],
            "CUDA_MPS_PIPE_DIRECTORY": mig["JDG_MPS_PIPE_DIRECTORY"],
            "CUDA_MPS_LOG_DIRECTORY": mig["JDG_MPS_LOG_DIRECTORY"],
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "100",
        }
    )
    replays: dict[str, dict[str, Any]] = {}
    raw_inputs: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        directory = args.output / mode
        profile_trace = directory / "operator-profile.jsonl"
        pressure: subprocess.Popen[str] | None = None
        if mode != "isolated":
            ready = directory.parent / f"{mode}-pressure.ready"
            pressure = subprocess.Popen(
                [
                    "taskset", "--cpu-list", str(args.pressure_cpu),
                    str(args.pressure_binary), "--mode", mode,
                    "--duration-seconds", str(args.pressure_seconds),
                    "--ready-file", str(ready),
                ],
                cwd=args.repo,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            wait_ready(ready, pressure)
        target_env = environment | {
            "LD_PRELOAD": str(args.capture_library),
            "ORION_TRT_PROFILE_TRACE": str(profile_trace),
        }
        completed = subprocess.run(
            [
                "taskset", "--cpu-list", str(args.target_cpu),
                str(args.profile_binary), "--engine", str(args.engine),
                "--model-name", args.model, "--samples", str(args.samples),
                "--warmup", str(args.warmup), "--output-dir", str(directory),
                "--single-client-profile",
            ],
            cwd=args.repo,
            env=target_env,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        (directory / "target.stdout").write_text(completed.stdout, encoding="utf-8")
        (directory / "target.stderr").write_text(completed.stderr, encoding="utf-8")
        pressure_evidence = None
        if pressure is not None:
            pressure_stdout, pressure_stderr = pressure.communicate(
                timeout=args.pressure_seconds + 10.0
            )
            if pressure.returncode != 0:
                raise RuntimeError(f"{mode} pressure failed: {pressure_stderr}")
            (directory / "pressure.json").write_text(pressure_stdout, encoding="utf-8")
            (directory / "pressure.stderr").write_text(
                pressure_stderr, encoding="utf-8"
            )
            pressure_value = json.loads(pressure_stdout)
            if (
                pressure_value.get("mode") != mode
                or pressure_value.get("multiprocessors") != 8
                or pressure_value.get("completed_launches", 0) <= 0
            ):
                raise ValueError(f"{mode} pressure evidence is invalid")
            pressure_evidence = {
                "path": str((directory / "pressure.json").resolve()),
                "sha256": sha256(directory / "pressure.json"),
            }
        replay = replay_mode(
            directory / "result.json", profile_trace,
            model=args.model, samples=args.samples, warmup=args.warmup,
        )
        replays[mode] = replay
        raw_inputs[mode] = {
            "result": {
                "path": str((directory / "result.json").resolve()),
                "sha256": replay["result_sha256"],
            },
            "operator_trace": {
                "path": str(profile_trace.resolve()),
                "sha256": replay["trace_sha256"],
            },
            "pressure": pressure_evidence,
        }
    profiles = classify_profiles(replays)
    scheduler_profile = args.output / "scheduler-profile.tsv"
    scheduler_profile.write_text(
        "orion-thor-profile-v1\n"
        "position\tapi\tgrid_x\tgrid_y\tgrid_z\tblock_x\tblock_y\tblock_z\t"
        "shared_mem_bytes\tprofile\tsm_used\tduration_us\n"
        + "".join(
            f"{row['position']}\t{row['api']}\t"
            f"{row['grid'][0]}\t{row['grid'][1]}\t{row['grid'][2]}\t"
            f"{row['block'][0]}\t{row['block'][1]}\t{row['block'][2]}\t"
            f"{row['shared_mem_bytes']}\t{row['profile']}\t{row['sm_used']}\t"
            f"{row['duration_us']:.9g}\n"
            for row in profiles
        ),
        encoding="ascii",
    )
    result = {
        "schema_version": 1,
        "kind": "orion-thor-operation-profile",
        "upstream_commit": UPSTREAM_COMMIT,
        "numeric_comparison_allowed": False,
        "model": args.model,
        "engine": {"path": str(args.engine.resolve()), "sha256": sha256(args.engine)},
        "mig": {"uuid": mig["JDG_MIG_SMALL_UUID"], "multiprocessors": 8},
        "profiling": {
            "samples": args.samples,
            "warmup": args.warmup,
            "classification": "relative-compute-vs-memory-pressure-sensitivity",
            "classification_margin": CLASSIFICATION_MARGIN,
            "sm_demand": "occupancy-based-grid-coverage",
        },
        "operations_per_inference": len(profiles),
        "resource_class_counts": {
            label: sum(row["resource_class"] == label for row in profiles)
            for label in ("compute", "memory", "unclear")
        },
        "operations": profiles,
        "scheduler_profile": {
            "path": str(scheduler_profile.resolve()),
            "sha256": sha256(scheduler_profile),
            "schema": "orion-thor-profile-v1",
        },
        "raw_inputs": raw_inputs,
    }
    (args.output / "profile.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=root)
    parser.add_argument("--mig-env", type=Path, default=Path("/tmp/jdg-mps-1g/mig.env"))
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--pressure-seconds", type=float, default=3.0)
    parser.add_argument("--target-cpu", type=int, default=12)
    parser.add_argument("--pressure-cpu", type=int, default=0)
    parser.add_argument(
        "--profile-binary", type=Path,
        default=root / "build-r39/orion-trt-native-smoke",
    )
    parser.add_argument(
        "--pressure-binary", type=Path,
        default=root / "build-r39/orion-profile-pressure",
    )
    parser.add_argument(
        "--capture-library", type=Path,
        default=root / "build-r39/liborion-trt-driver-capture.so",
    )
    args = parser.parse_args(argv)
    args.repo = args.repo.resolve()
    args.mig_env = args.mig_env.resolve()
    args.engine = args.engine.resolve()
    args.output = args.output.resolve()
    args.profile_binary = args.profile_binary.resolve()
    args.pressure_binary = args.pressure_binary.resolve()
    args.capture_library = args.capture_library.resolve()
    for path in (
        args.mig_env, args.engine, args.profile_binary,
        args.pressure_binary, args.capture_library,
    ):
        if not path.is_file():
            raise ValueError(f"missing Orion profiling input: {path}")
    result = run_campaign(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
