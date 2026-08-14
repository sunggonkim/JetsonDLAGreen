#!/usr/bin/env python3
"""Run the nonthermal real-ASR NVIDIA-MIG/QUIET crossover probe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import signal
import statistics
import struct
import subprocess
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Mode:
    name: str
    encoder_profile: str
    producer_device: str
    gate_background: bool


@dataclass(frozen=True)
class Background:
    model_name: str
    engine: pathlib.Path


MODES = (
    Mode("nvidia-mig", "2g", "big", False),
    Mode("nvidia-mps-static-split", "1g", "small", False),
    Mode("quiet", "1g", "small", True),
)


def parse_additional_background(value: str) -> tuple[str, pathlib.Path]:
    model_name, separator, engine = value.partition("=")
    if not separator or not model_name.strip() or not engine.strip():
        raise argparse.ArgumentTypeError(
            "--additional-background expects MODEL_NAME=ENGINE_PATH"
        )
    return model_name.strip(), pathlib.Path(engine)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_env(path: pathlib.Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key:
            raise ValueError("MIG environment line is malformed")
        result[key] = value
    required = {
        "JDG_MIG_SMALL_UUID",
        "JDG_MIG_BIG_UUID",
        "JDG_MPS_PIPE_DIRECTORY",
        "JDG_MPS_LOG_DIRECTORY",
    }
    if not required.issubset(result):
        raise ValueError("MIG environment is incomplete")
    return result


def input_count(path: pathlib.Path) -> int:
    with path.open("rb") as stream:
        if stream.read(8) != b"JDGINT1\x00":
            raise ValueError("input trace magic differs")
        schema, count, sample_bytes = struct.unpack("<IIQ", stream.read(16))
    if schema != 1 or count <= 0 or sample_bytes <= 0:
        raise ValueError("input trace header is invalid")
    return count


def process_state(pid: int) -> str:
    text = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    end = text.rfind(")")
    if end < 0 or end + 2 >= len(text):
        raise RuntimeError("background process state is malformed")
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


def stop_background(process: subprocess.Popen[str]) -> tuple[dict[str, Any], str]:
    if process.poll() is None:
        for number in (signal.SIGCONT, signal.SIGINT):
            try:
                os.kill(process.pid, number)
            except ProcessLookupError:
                break
    try:
        stdout, stderr = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            f"background failed ({process.returncode}): {stdout} {stderr}"
        )
    return json.loads(stdout), stderr


def stop_backgrounds(
    processes: list[subprocess.Popen[str]],
) -> list[tuple[dict[str, Any], str]]:
    results: list[tuple[dict[str, Any], str]] = []
    errors: list[Exception] = []
    for process in processes:
        try:
            results.append(stop_background(process))
        except Exception as error:  # Preserve cleanup for every remaining worker.
            errors.append(error)
    if errors:
        raise RuntimeError(
            "one or more background workers failed during cleanup: "
            + "; ".join(str(error) for error in errors)
        )
    return results


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def stage_metrics(path: pathlib.Path, warmup: int) -> dict[str, float]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))[warmup:]
    if not rows:
        raise ValueError("pipeline trace has no measured rows")
    producer = [
        (int(row["producer_done_ns"]) - int(row["producer_start_ns"])) / 1000.0
        for row in rows
    ]
    consumer = [
        (int(row["consumer_done_ns"]) - int(row["consumer_start_ns"])) / 1000.0
        for row in rows
    ]
    return {
        "producer_mean_us": statistics.fmean(producer),
        "producer_p99_us": percentile(producer, 0.99),
        "consumer_mean_us": statistics.fmean(consumer),
        "consumer_p99_us": percentile(consumer, 0.99),
    }


def run_one(
    args: argparse.Namespace,
    repo: pathlib.Path,
    mig: dict[str, str],
    mode: Mode,
    rate_rps: float,
    session: int,
) -> dict[str, Any]:
    label = "saturated" if rate_rps == 0.0 else f"{rate_rps:g}rps"
    run_dir = args.result_dir / f"session-{session:02d}" / label / mode.name
    run_dir.mkdir(parents=True)

    background_environment = os.environ.copy()
    background_environment.update(
        {
            "CUDA_VISIBLE_DEVICES": mig["JDG_MIG_SMALL_UUID"],
            "CUDA_MPS_PIPE_DIRECTORY": mig["JDG_MPS_PIPE_DIRECTORY"],
            "CUDA_MPS_LOG_DIRECTORY": mig["JDG_MPS_LOG_DIRECTORY"],
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "100",
        }
    )
    backgrounds: list[subprocess.Popen[str]] = []
    try:
        for index, specification in enumerate(args.backgrounds):
            background = subprocess.Popen(
                [
                    "taskset", "--cpu-list", str(index),
                    str(args.background_binary),
                    "--engine", str(specification.engine),
                    "--model-name", specification.model_name,
                    "--role", "pressure", "--duration-seconds", "3600",
                    "--burst-size", "1",
                    "--period-ms", str(args.background_period_ms),
                    "--warmup", str(args.warmup),
                    "--include-transfers", "true", "--priority", "default",
                    "--start-paused", "true",
                ],
                cwd=repo,
                env=background_environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            backgrounds.append(background)
            wait_paused(background)
        for background in backgrounds:
            os.kill(background.pid, signal.SIGCONT)
    except Exception:
        if backgrounds:
            stop_backgrounds(backgrounds)
        raise

    encoder = args.encoder_2g if mode.encoder_profile == "2g" else args.encoder_1g
    producer = (
        mig["JDG_MIG_BIG_UUID"]
        if mode.producer_device == "big"
        else mig["JDG_MIG_SMALL_UUID"]
    )
    period_us = 0.0 if rate_rps == 0.0 else 1.0e6 / rate_rps
    command = [
        "taskset", "--cpu-list", "13", str(args.binary),
        "--encoder-engine", str(encoder),
        "--decoder-initial-engine", str(args.decoder_initial),
        "--decoder-with-past-engine", str(args.decoder_with_past),
        "--input-trace", str(args.input_trace),
        "--output-trace", str(run_dir / "asr-output.bin"),
        "--trace-csv", str(run_dir / "pipeline.csv"),
        "--warmup", str(args.warmup), "--iterations", str(args.requests),
        "--max-tokens", str(args.max_tokens),
        "--pipeline-slots", str(args.pipeline_slots),
        "--arrival-period-us", f"{period_us:.9f}",
        "--deadline-us", str(args.deadline_us),
        "--producer", producer,
        "--consumer", mig["JDG_MIG_BIG_UUID"],
        "--mps-pipe", (
            mig["JDG_MPS_PIPE_DIRECTORY"] if mode.producer_device == "small" else ""
        ),
    ]
    if mode.gate_background:
        command.extend(
            ("--gate-pids", ",".join(str(worker.pid) for worker in backgrounds))
        )
    (run_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    try:
        completed = subprocess.run(
            command,
            cwd=repo,
            env={**os.environ, "LD_LIBRARY_PATH": "/usr/local/cuda-13.2/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")},
            capture_output=True,
            text=True,
            check=True,
            timeout=180,
        )
    except Exception:
        stop_backgrounds(backgrounds)
        raise
    background_results = stop_backgrounds(backgrounds)
    pipeline = json.loads(completed.stdout)
    (run_dir / "pipeline.json").write_text(
        json.dumps(pipeline, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "pipeline.stderr").write_text(completed.stderr, encoding="utf-8")
    workers = [
        {
            "model_name": specification.model_name,
            "engine": str(specification.engine),
            "result": worker_result,
        }
        for specification, (worker_result, _) in zip(
            args.backgrounds, background_results, strict=True
        )
    ]
    background_goodput = sum(
        float(worker["result"]["throughput_per_second"]) for worker in workers
    )
    background_completed = sum(
        int(worker["result"]["completed_requests"]) for worker in workers
    )
    (run_dir / "background.json").write_text(
        json.dumps(
            {
                "workers": workers,
                "completed_requests": background_completed,
                "throughput_per_second": background_goodput,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "background.stderr").write_text(
        "".join(
            f"worker {index} ({specification.model_name}):\n{stderr}"
            for index, (specification, (_, stderr)) in enumerate(
                zip(args.backgrounds, background_results, strict=True)
            )
        ),
        encoding="utf-8",
    )
    if (
        pipeline.get("status") != "ok"
        or background_completed <= 0
        or background_goodput <= 0.0
        or any(
            int(worker["result"].get("completed_requests", 0)) <= 0
            or float(worker["result"].get("throughput_per_second", 0.0)) <= 0.0
            for worker in workers
        )
    ):
        raise RuntimeError(f"{mode.name} run did not complete successfully")
    result = {
        "session": session,
        "rate_rps": rate_rps,
        "mode": mode.name,
        "scenario_id": args.scenario_id,
        "background_model_names": [
            specification.model_name for specification in args.backgrounds
        ],
        "background_workers": len(args.backgrounds),
        "deadline_misses": int(pipeline["deadline_misses"]),
        "requests": args.requests,
        "p50_us": float(pipeline["p50_us"]),
        "p99_us": float(pipeline["p99_us"]),
        "queue_p99_us": float(pipeline["queue_p99_us"]),
        "request_goodput_rps": float(pipeline["request_goodput_rps"]),
        "background_goodput_rps": background_goodput,
        "output_sha256": sha256(run_dir / "asr-output.bin"),
        "gated_processes": int(pipeline["gated_processes"]),
        **stage_metrics(run_dir / "pipeline.csv", args.warmup),
    }
    for field in ("gate_hold_p50_us", "gate_hold_p99_us", "gate_acquire_p99_us"):
        result[field] = float(pipeline[field]) if field in pipeline else None
    return result


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    keys = sorted({(row["rate_rps"], row["mode"]) for row in rows})
    for rate, mode in keys:
        group = [row for row in rows if row["rate_rps"] == rate and row["mode"] == mode]
        result.append(
            {
                "rate_rps": rate,
                "mode": mode,
                "sessions": len(group),
                "requests": sum(row["requests"] for row in group),
                "deadline_misses": sum(row["deadline_misses"] for row in group),
                **{
                    field + "_mean": statistics.fmean(row[field] for row in group)
                    for field in (
                        "p50_us", "p99_us", "queue_p99_us",
                        "request_goodput_rps", "background_goodput_rps",
                        "producer_mean_us", "consumer_mean_us",
                    )
                },
            }
        )
    return result


def scenario_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Return the deployment interpretation frozen into every raw summary."""
    return {
        "id": args.scenario_id,
        "label": args.scenario_label,
        "description": args.scenario_description,
        "foreground": "Whisper-Tiny encoder-decoder ASR",
        "background_models": [
            specification.model_name for specification in args.backgrounds
        ],
        "background_engines": [
            str(specification.engine) for specification in args.backgrounds
        ],
        "background_workers": len(args.backgrounds),
        "background_release": (
            "saturated-backlog"
            if args.background_period_ms == 0.0
            else f"periodic-{args.background_period_ms:g}ms"
        ),
        "deployment_scope": args.deployment_scope,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo = pathlib.Path(__file__).resolve().parents[1]
    parser.add_argument("--repo", type=pathlib.Path, default=repo)
    parser.add_argument("--result-dir", type=pathlib.Path, required=True)
    parser.add_argument("--mig-env", type=pathlib.Path, default=pathlib.Path("/tmp/jdg-mps-1g/mig.env"))
    parser.add_argument("--binary", type=pathlib.Path, default=repo / "build-r39/jdg-mig-whisper-asr")
    parser.add_argument("--background-binary", type=pathlib.Path, default=repo / "build-r39/jdg-trt-bench")
    parser.add_argument("--input-trace", type=pathlib.Path, required=True)
    parser.add_argument("--encoder-1g", type=pathlib.Path, default=repo / "models/engines/mig-1g-q100/whisper-tiny-encoder-fp32.engine")
    parser.add_argument("--encoder-2g", type=pathlib.Path, default=repo / "models/engines/mig-2g-q100/whisper-tiny-encoder-fp32.engine")
    parser.add_argument("--decoder-initial", type=pathlib.Path, default=repo / "models/engines/mig-2g-q100/whisper-tiny-decoder-initial-4-fp32.engine")
    parser.add_argument("--decoder-with-past", type=pathlib.Path, default=repo / "models/engines/mig-2g-q100/whisper-tiny-decoder-with-past-fp32.engine")
    parser.add_argument("--background-engine", type=pathlib.Path, default=repo / "models/engines/mig-1g-q100/distilbert-sst2.engine")
    parser.add_argument("--background-model-name", default="distilbert-sst2")
    parser.add_argument(
        "--additional-background",
        action="append",
        type=parse_additional_background,
        default=[],
        metavar="MODEL_NAME=ENGINE_PATH",
    )
    parser.add_argument("--scenario-id", default="speech-plus-nlp")
    parser.add_argument("--scenario-label", default="Speech + NLP")
    parser.add_argument(
        "--scenario-description",
        default="interactive ASR foreground with queued NLP classification",
    )
    parser.add_argument(
        "--deployment-scope",
        default="multi-channel-or-queued-edge-gateway-stress",
    )
    parser.add_argument(
        "--input-policy",
        default="cyclic-performance-replay-not-accuracy-expansion",
    )
    parser.add_argument("--rates", type=float, nargs="+", required=True)
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--pipeline-slots", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--deadline-us", type=float, default=250000.0)
    parser.add_argument("--background-period-ms", type=float, default=0.0)
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    args.result_dir = args.result_dir.resolve()
    for name in (
        "binary", "background_binary", "input_trace", "encoder_1g", "encoder_2g",
        "decoder_initial", "decoder_with_past", "background_engine", "mig_env",
    ):
        setattr(args, name, getattr(args, name).resolve())
        if not getattr(args, name).is_file():
            raise ValueError(f"--{name.replace('_', '-')} is not a file")
    additional_backgrounds: list[Background] = []
    for model_name, engine in args.additional_background:
        resolved = (args.repo / engine).resolve() if not engine.is_absolute() else engine.resolve()
        if not resolved.is_file():
            raise ValueError("--additional-background engine is not a file")
        additional_backgrounds.append(Background(model_name, resolved))
    args.backgrounds = [
        Background(args.background_model_name.strip(), args.background_engine),
        *additional_backgrounds,
    ]
    if args.result_dir.exists():
        raise ValueError("result directory already exists")
    if args.sessions <= 0 or args.requests <= 0 or args.warmup < 0:
        raise ValueError("session/request counts are invalid")
    if args.pipeline_slots != 3:
        raise ValueError("crossover probe requires exactly three pipeline slots")
    if any(rate < 0.0 for rate in args.rates) or len(set(args.rates)) != len(args.rates):
        raise ValueError("rates must be unique and non-negative")
    if input_count(args.input_trace) != args.warmup + args.requests:
        raise ValueError("input trace count differs from warmup plus requests")
    if (
        not args.scenario_id
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
            for character in args.scenario_id
        )
        or not args.scenario_label.strip()
        or not args.scenario_description.strip()
        or not args.background_model_name.strip()
        or len({item.model_name for item in args.backgrounds}) != len(args.backgrounds)
        or not args.deployment_scope.strip()
        or not args.input_policy.strip()
    ):
        raise ValueError("scenario metadata is invalid")
    args.result_dir.mkdir(parents=True)
    mig = load_env(args.mig_env)

    artifacts = (
        args.binary, args.background_binary, args.input_trace,
        args.encoder_1g, args.encoder_2g, args.decoder_initial,
        args.decoder_with_past,
        *(specification.engine for specification in args.backgrounds),
        args.repo / "benchmarks/mig_whisper_asr.cpp",
        pathlib.Path(__file__).resolve(),
    )
    provenance = {str(path): sha256(path) for path in artifacts}
    (args.result_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    for session in range(1, args.sessions + 1):
        offset = (session - 1) % len(MODES)
        order = MODES[offset:] + MODES[:offset]
        rates = args.rates if session % 2 == 1 else list(reversed(args.rates))
        for rate in rates:
            pair: list[dict[str, Any]] = []
            for mode in order:
                row = run_one(args, args.repo, mig, mode, rate, session)
                rows.append(row)
                pair.append(row)
            if len({row["output_sha256"] for row in pair}) != 1:
                raise RuntimeError("same-session comparator outputs differ")

    output = {
        "schema_version": 1,
        "kind": "p9-whisper-asr-mig-crossover",
        "evidence_class": "exploratory-nonthermal-directional",
        "thermal_campaign": False,
        "input_policy": args.input_policy,
        "scenario": scenario_metadata(args),
        "study_design": (
            "balanced-repeated" if args.sessions >= 3 and len(args.rates) == 1
            else "directional-sweep"
        ),
        "pipeline_slots": args.pipeline_slots,
        "deadline_us": args.deadline_us,
        "background_period_ms": args.background_period_ms,
        "background_workers": len(args.backgrounds),
        "comparator_output_contract": "byte-identical",
        "rows": rows,
        "aggregate": aggregate(rows),
    }
    (args.result_dir / "summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
