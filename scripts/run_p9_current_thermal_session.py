#!/usr/bin/env python3
"""Run one current ImageNette session while collecting timestamped tegrastats.

The launcher remains the authority for the workload and comparator contract.
This wrapper only adds a monotonic telemetry stream and explicit thermal phase
markers; it does not change scheduling policy or workload inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.tegrastats_telemetry import JsonlTelemetryWriter, TegrastatsMonitor


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--sequence-index", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--deadline-lock", type=Path, required=True)
    parser.add_argument("--quiet-plan", type=Path, required=True)
    parser.add_argument("--common-workload", type=Path, required=True)
    parser.add_argument("--input-trace", type=Path, required=True)
    parser.add_argument("--producer-engine", type=Path, required=True)
    parser.add_argument("--consumer-engine", type=Path, required=True)
    parser.add_argument("--accuracy-gate", type=Path, required=True)
    parser.add_argument("--reference-trace", type=Path, required=True)
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--reference-engine", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--deadline-us", required=True)
    parser.add_argument("--requests", type=int, default=1100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--background-period-ms", default="4.0")
    parser.add_argument("--prepare-seconds", type=float, default=5.0)
    parser.add_argument("--tegrastats-interval-ms", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.prepare_seconds < 0.0 or args.tegrastats_interval_ms <= 0:
        raise SystemExit("thermal preparation and interval values must be positive")
    output = args.result_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    telemetry_path = output / "thermal-telemetry.jsonl"
    command_log = output / "thermal-launcher.log"
    environment = os.environ.copy()
    environment.update(
        {
            "RESULT_ROOT": str(output),
            "DEADLINE_LOCK": str(args.deadline_lock.resolve()),
            "QUIET_PLAN": str(args.quiet_plan.resolve()),
            "WORKLOAD": "resnet50-classification",
            "COMMON_WORKLOAD_CONTRACT": str(args.common_workload.resolve()),
            "PRODUCER_INPUT_TRACE": str(args.input_trace.resolve()),
            "PRODUCER_ENGINE": str(args.producer_engine.resolve()),
            "CONSUMER_ENGINE": str(args.consumer_engine.resolve()),
            "APPLICATION_ACCURACY_GATE": str(args.accuracy_gate.resolve()),
            "APPLICATION_ACCURACY_REFERENCE_TRACE": str(args.reference_trace.resolve()),
            "APPLICATION_ACCURACY_REFERENCE_OUTPUT_TRACE": str(args.reference_output.resolve()),
            "APPLICATION_ACCURACY_REFERENCE_PIPELINE_CSV": str(args.reference_csv.resolve()),
            "APPLICATION_ACCURACY_REFERENCE_ENGINE": str(args.reference_engine.resolve()),
            "APPLICATION_ACCURACY_CLASS_MAP": str(args.class_map.resolve()),
            "APPLICATION_ACCURACY_DEADLINE_US": str(args.deadline_us),
            "REQUESTS": str(args.requests),
            "WARMUP": str(args.warmup),
            "BACKGROUND_PERIOD_MS": str(args.background_period_ms),
            "SEQUENCE_INDICES": str(args.sequence_index),
        }
    )
    with telemetry_path.open("w", encoding="utf-8") as stream:
        writer = JsonlTelemetryWriter(stream, flush_every=1)
        monitor = TegrastatsMonitor(writer)
        stats = subprocess.Popen(
            ["tegrastats", "--interval", str(args.tegrastats_interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        reader_error: list[BaseException] = []

        def read_stats() -> None:
            assert stats.stdout is not None
            try:
                for line in stats.stdout:
                    if line.strip():
                        monitor.record_line(line)
            except BaseException as error:  # preserve collector failures in parent
                reader_error.append(error)

        reader = threading.Thread(target=read_stats, name="tegrastats-reader")
        reader.start()
        child: subprocess.Popen[str] | None = None
        return_code = 1
        try:
            monitor.mark(
                "thermal_prepare",
                {
                    "protocol": "p9-current-quiet-thermal-v1",
                    "sequence_index": args.sequence_index,
                    "prepare_seconds": args.prepare_seconds,
                    "tegrastats_interval_ms": args.tegrastats_interval_ms,
                },
            )
            time.sleep(args.prepare_seconds)
            monitor.mark(
                "execution_start",
                {
                    "sequence_index": args.sequence_index,
                    "requests": args.requests,
                    "warmup": args.warmup,
                },
            )
            command = ["bash", str(ROOT / "scripts/run_p9_active_frontier_campaign.sh")]
            with command_log.open("w", encoding="utf-8") as log:
                child = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                return_code = child.wait()
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            if stats.poll() is None:
                stats.terminate()
            try:
                stats.wait(timeout=10)
            except subprocess.TimeoutExpired:
                stats.kill()
                stats.wait()
            if stats.stdout is not None:
                stats.stdout.close()
            reader.join(timeout=10)
            if reader.is_alive():
                raise RuntimeError("tegrastats reader did not terminate")
            monitor.mark(
                "execution_end",
                {"sequence_index": args.sequence_index, "return_code": return_code},
            )
            monitor.mark(
                "thermal_end",
                {"sequence_index": args.sequence_index, "return_code": return_code},
            )
            monitor.close()
    if reader_error:
        raise RuntimeError("tegrastats collection failed") from reader_error[0]
    result = {
        "kind": "p9-current-quiet-thermal-session",
        "protocol": "p9-current-quiet-thermal-v1",
        "sequence_index": args.sequence_index,
        "result_root": str(output),
        "telemetry": str(telemetry_path),
        "command_log": str(command_log),
        "return_code": return_code,
        "requests": args.requests,
        "warmup": args.warmup,
        "deadline_us": float(args.deadline_us),
        "deadline_lock": str(args.deadline_lock.resolve()),
        "quiet_plan": str(args.quiet_plan.resolve()),
        "common_workload": str(args.common_workload.resolve()),
    }
    (output / "thermal-session.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
