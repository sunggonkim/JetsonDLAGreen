#!/usr/bin/env python3
"""Replay XSched's dependent TensorRT numeric smoke from raw evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baselines/orion"))
from verify_dependent_smoke import load_deadline_lock, replay_pipeline  # noqa: E402
from verify_profiled_smoke import load_json, sha256  # noqa: E402


BE_COLUMNS = (
    "request", "release_to_completion_ms", "gpu_service_ms", "queue_delay_ms",
    "gate_overhead_ms", "drain_ms", "resume_ms",
)
CONNECTED = re.compile(r"client process (\d+) connected, cmdline: (.*)")
XQUEUE = re.compile(r"XQueue \(0x[0-9a-f]+\) from process (\d+) created")
TRANSITION = re.compile(
    r"schedule transition pid (\d+) operation \d+ running (\d+) suspended (\d+)"
)

WORKLOADS = {
    "whisper-projection": {
        "pipeline": "whisper-last-hidden-state-to-projection-mlp",
        "payload_bytes": 2_304_000,
    },
    "resnet-detection-head": {
        "pipeline": "resnet10-backbone-to-learned-detection-head",
        "payload_bytes": 1_884_160,
    },
    "resnet50-classification": {
        "pipeline": "resnet50-backbone-to-classification-head",
        "payload_bytes": 802_816,
    },
}


def _load_common_workload_contract(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    raw = resolved.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("common workload contract is not newline-complete")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("common workload contract is invalid JSON") from error
    required = (
        "workload_id", "topology", "placement", "input_tensor", "payload_bytes",
        "arrival_trace_path", "arrival_trace_sha256",
        "dataset_manifest_path", "dataset_manifest_sha256",
    )
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("common workload contract schema differs")
    if any(key not in value for key in required):
        raise ValueError("common workload contract lacks required fields")
    for path_key, digest_key in (
        ("arrival_trace_path", "arrival_trace_sha256"),
        ("dataset_manifest_path", "dataset_manifest_sha256"),
    ):
        evidence = Path(value[path_key]).resolve()
        digest = value[digest_key]
        if (
            not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or not evidence.is_file()
            or hashlib.sha256(evidence.read_bytes()).hexdigest() != digest
        ):
            raise ValueError(f"common workload evidence SHA mismatches: {path_key}")
        value[path_key] = str(evidence)
    producer_path_value = value.get("producer_input_trace_path")
    producer_digest = value.get("producer_input_trace_sha256")
    if producer_path_value is not None or producer_digest is not None:
        if (
            not isinstance(producer_path_value, str)
            or not producer_path_value
            or not isinstance(producer_digest, str)
            or len(producer_digest) != 64
            or any(c not in "0123456789abcdef" for c in producer_digest)
        ):
            raise ValueError("common workload producer input trace binding is invalid")
        producer = Path(producer_path_value).resolve()
        if not producer.is_file() or sha256(producer) != producer_digest:
            raise ValueError("common workload producer input trace SHA mismatches")
        value["producer_input_trace_path"] = str(producer)
    value["contract_path"] = str(resolved)
    value["contract_sha256"] = hashlib.sha256(raw).hexdigest()
    return value


def replay_be_window(path: Path, be: dict[str, Any], start_ns: int,
                     end_ns: int, background_period_ms: float = 4.0) -> dict[str, float | int]:
    config = be.get("config")
    if (
        not isinstance(config, dict)
        or not math.isclose(float(config.get("period_ms")), background_period_ms,
                            rel_tol=0.0, abs_tol=1e-9)
        or not math.isfinite(background_period_ms)
        or background_period_ms <= 0.0
    ):
        raise ValueError("XSched BE release period differs")
    measurement_start = be.get("measurement_start_monotonic_ns")
    measurement_end = be.get("measurement_end_monotonic_ns")
    completed = be.get("completed_requests")
    if (
        not isinstance(measurement_start, int) or isinstance(measurement_start, bool)
        or not isinstance(measurement_end, int) or isinstance(measurement_end, bool)
        or measurement_end <= measurement_start
        or not isinstance(completed, int) or isinstance(completed, bool)
        or completed <= 0 or end_ns <= start_ns
    ):
        raise ValueError("XSched BE measurement window differs")
    period_ns = round(background_period_ms * 1.0e6)
    offered = finished = rows = 0
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != BE_COLUMNS:
            raise ValueError("XSched BE trace schema differs")
        for index, row in enumerate(reader):
            if int(row["request"]) != index:
                raise ValueError("XSched BE request sequence differs")
            latency_ms = float(row["release_to_completion_ms"])
            if not math.isfinite(latency_ms) or latency_ms <= 0.0:
                raise ValueError("XSched BE latency is invalid")
            release_ns = measurement_start + (index + 1) * period_ns
            completion_ns = release_ns + round(latency_ms * 1.0e6)
            offered += int(start_ns <= release_ns <= end_ns)
            finished += int(start_ns <= completion_ns <= end_ns)
            rows += 1
    if rows != completed:
        raise ValueError("XSched BE completion count differs")
    seconds = (end_ns - start_ns) / 1.0e9
    return {
        "window_seconds": seconds,
        "offered_requests": offered,
        "completed_requests": finished,
        "offered_rps": offered / seconds,
        "completion_goodput_rps": finished / seconds,
    }


def replay_scheduler(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    if any(token in text for token in ("[XSCHED ERRO", "Assertion failed", "cuda error")):
        raise ValueError("XSched server recorded a runtime failure")
    clients: dict[int, str] = {
        int(match.group(1)): match.group(2) for match in CONNECTED.finditer(text)
    }
    be_pids = [pid for pid, command in clients.items() if "jdg-trt-bench" in command]
    critical_pids = [
        pid for pid, command in clients.items() if "jdg-mig-trt-pipeline" in command
    ]
    xqueue_pids = {int(match.group(1)) for match in XQUEUE.finditer(text)}
    if len(be_pids) != 1 or len(critical_pids) != 2:
        raise ValueError("XSched did not connect the BE and two DAG stages")
    expected = set(be_pids + critical_pids)
    if not expected.issubset(xqueue_pids):
        raise ValueError("XSched did not create every measured XQueue")
    be_pid = be_pids[0]
    states = [
        (int(match.group(2)), int(match.group(3)))
        for match in TRANSITION.finditer(text)
        if int(match.group(1)) == be_pid
    ]
    suspended = sum(state == (0, 1) for state in states)
    resumed = sum(state == (1, 0) for state in states[1:])
    if suspended <= 0 or resumed <= 0:
        raise ValueError("XSched HPF did not suspend and resume BE")
    return {
        "connected_clients": len(expected),
        "xqueue_clients": len(expected & xqueue_pids),
        "be_suspend_transitions": suspended,
        "be_resume_transitions": resumed,
    }


def verify(result_path: Path, pipeline_path: Path, be_result_path: Path,
           be_trace_path: Path, server_log_path: Path, client_logs: list[Path],
           deadline_lock_path: Path, pipeline_binary: Path, xsched_server: Path,
           xsched_shim: Path, xsched_patch: Path, repo: Path,
           workload: str = "whisper-projection",
           producer_engine: Path | None = None,
           consumer_engine: Path | None = None,
           consumer_input_tensor: str | None = None,
           application_output_trace_path: Path | None = None,
           common_workload_contract: Path | None = None,
           require_common_workload: bool = False) -> dict[str, Any]:
    result = load_json(result_path)
    be = load_json(be_result_path)
    lock = load_deadline_lock(deadline_lock_path, repo)
    deadline_us = float(lock["deadline_us"])
    start_ns = result.get("measurement_start_monotonic_ns")
    end_ns = result.get("measurement_end_monotonic_ns")
    spec = WORKLOADS.get(workload)
    if spec is None:
        raise ValueError(f"unsupported XSched workload: {workload}")
    if require_common_workload and common_workload_contract is None:
        raise ValueError("XSched verifier requires a common workload contract")
    common_workload = (
        _load_common_workload_contract(common_workload_contract)
        if common_workload_contract is not None else None
    )
    if common_workload is not None:
        expected_tensor = (
            "Layer6_relu_Y" if workload == "resnet-detection-head"
            else "gpu_0/res4_5_branch2c_bn_2" if workload == "resnet50-classification"
            else "last_hidden_state"
        )
        expected = {
            "workload_id": workload,
            "topology": "fixed-2g+1g",
            "placement": "fixed-1g-producer-2g-consumer",
            "input_tensor": expected_tensor,
            "payload_bytes": spec["payload_bytes"],
        }
        if any(common_workload.get(k) != v for k, v in expected.items()):
            raise ValueError("XSched common workload contract differs")
        if workload in {"resnet-detection-head", "resnet50-classification"}:
            producer_contract = common_workload.get("producer_input_trace_path")
            producer_digest = common_workload.get("producer_input_trace_sha256")
            result_trace = result.get("producer_input_trace")
            if (
                not isinstance(producer_contract, str)
                or not isinstance(producer_digest, str)
                or not isinstance(result_trace, str)
                or not Path(result_trace).is_file()
                or sha256(Path(result_trace)) != producer_digest
            ):
                raise ValueError("XSched producer input trace differs from common workload")
        operational_path_value = common_workload.get("operational_arrival_trace_path")
        operational_digest = common_workload.get("operational_arrival_trace_sha256")
        if operational_path_value is not None or operational_digest is not None:
            if (
                not isinstance(operational_path_value, str)
                or not isinstance(operational_digest, str)
                or not Path(operational_path_value).is_file()
                or sha256(Path(operational_path_value)) != operational_digest
                or result.get("arrival_schedule_mode") != "operational-trace"
            ):
                raise ValueError("XSched operational arrival trace differs from common workload")
    if (
        result.get("status") != "ok"
        or result.get("pipeline") != spec["pipeline"]
        or result.get("transport") != "registered-shared-sysmem-direct-binding"
        or result.get("payload_bytes") != spec["payload_bytes"]
        or result.get("producer_sms") != 8 or result.get("consumer_sms") != 12
        or result.get("checksum_failures") != 0
        or result.get("unique_payload_checksums", 0) < 2
        or result.get("unique_policy_output_checksums", 0) < 2
        # The shell runner measures arrival-to-completion wall time.  The old
        # verifier accepted a legacy interval that excluded correctness
        # validation, which silently made this comparator incomparable with
        # QUIET and MPS.
        or result.get("deadline_mode") != "wall"
        or not math.isclose(float(result.get("deadline_us")), deadline_us,
                            rel_tol=0.0, abs_tol=1e-9)
        or not isinstance(start_ns, int) or isinstance(start_ns, bool)
        or not isinstance(end_ns, int) or isinstance(end_ns, bool)
        or end_ns <= start_ns
    ):
        raise ValueError("XSched dependent result contract differs")
    if workload == "resnet-detection-head":
        if (
            result.get("payload_shape") != [1, 512, 23, 40]
            or result.get("producer_output_tensor") != "Layer6_relu_Y"
            or result.get("consumer_input_tensor") != (consumer_input_tensor or "Layer6_relu_Y")
            or result.get("consumer_output_tensor") != "external-output"
        ):
            raise ValueError("XSched learned-head tensor contract differs")
        if consumer_engine is None or not consumer_engine.is_file():
            raise ValueError("XSched learned-head run lacks consumer engine provenance")
    if workload == "resnet50-classification":
        if (
            result.get("payload_shape") != [1, 1024, 14, 14]
            or result.get("producer_output_tensor") != "gpu_0/res4_5_branch2c_bn_2"
            or result.get("consumer_input_tensor") != (consumer_input_tensor or "gpu_0/res4_5_branch2c_bn_2")
            or result.get("consumer_output_tensor") != "external-output"
        ):
            raise ValueError("XSched ResNet50 tensor contract differs")
        if consumer_engine is None or not consumer_engine.is_file():
            raise ValueError("XSched ResNet50 run lacks consumer engine provenance")
    if producer_engine is not None and not producer_engine.is_file():
        raise ValueError("XSched producer engine provenance is missing")
    environment = be.get("execution_environment")
    config = be.get("config")
    gpu = be.get("gpu")
    if (
        be.get("model") != "distilbert-sst2"
        or be.get("role") != "benchmark"
        or not isinstance(environment, dict)
        or environment.get("mps_active_thread_percentage") != 100
        or not isinstance(config, dict)
        or config.get("warmup") != 20 or config.get("burst_size") != 1
        or config.get("period_ms") != 4 or config.get("priority") != "low"
        or config.get("include_transfers") is not True
        or not isinstance(gpu, dict) or gpu.get("multiprocessors") != 8
    ):
        raise ValueError("XSched BE workload contract differs")
    for path in client_logs:
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in ("[XSCHED ERRO", "Assertion failed", "cuda error")):
            raise ValueError("XSched client recorded a runtime failure")
    pipeline = replay_pipeline(pipeline_path, result, deadline_us)
    background = replay_be_window(be_trace_path, be, start_ns, end_ns)
    scheduler = replay_scheduler(server_log_path)
    application_output_trace = None
    if application_output_trace_path is not None:
        if not application_output_trace_path.is_file():
            raise ValueError("XSched application output trace is missing")
        application_output_trace = {
            "path": str(application_output_trace_path.resolve()),
            "sha256": sha256(application_output_trace_path),
            "capture_boundary": "post-completion",
        }
    return {
        "schema_version": 1,
        "kind": f"xsched-dependent-{workload}-numeric-smoke-verification",
        "system": "XSched (Thor port)",
        "functional_gate_passed": True,
        "numeric_smoke_valid": True,
        "formal_claim_allowed": False,
        "workload": workload,
        "background": "DistilBERT SST-2",
        "deadline_us": deadline_us,
        "requests": pipeline["requests"],
        "misses": pipeline["misses"],
        "dmr": pipeline["misses"] / pipeline["requests"],
        "p99_us": pipeline["p99_us"],
        "wall_p99_us": pipeline["p99_us"],
        "deadline_mode": "wall",
        "latency_contract": "production-wall-arrival-to-completion",
        "production_wall_definition": (
            "arrival-to-consumer-completion-excludes-correctness-validation"
        ),
        "correctness_validation_placement": "post-completion",
        "background_window": background,
        "scheduler": scheduler,
        "application_output_trace": application_output_trace,
        "common_workload": common_workload,
        "inputs": {
            "result_sha256": sha256(result_path),
            "pipeline_sha256": sha256(pipeline_path),
            "be_result_sha256": sha256(be_result_path),
            "be_trace_sha256": sha256(be_trace_path),
            "server_log_sha256": sha256(server_log_path),
            "deadline_lock_sha256": sha256(deadline_lock_path),
            "pipeline_binary_sha256": sha256(pipeline_binary),
            "producer_engine_sha256": sha256(producer_engine) if producer_engine else None,
            "consumer_engine_sha256": sha256(consumer_engine) if consumer_engine else None,
            "xsched_server_sha256": sha256(xsched_server),
            "xsched_shim_sha256": sha256(xsched_shim),
            "xsched_patch_sha256": sha256(xsched_patch),
            "client_log_sha256": {
                path.name: sha256(path) for path in client_logs
            },
        },
        "formal_next_gate": "counterbalanced repetitions under the frozen contract",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--be-result", type=Path, required=True)
    parser.add_argument("--be-trace", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--client-log", type=Path, action="append", default=[])
    parser.add_argument("--deadline-lock", type=Path, required=True)
    parser.add_argument("--pipeline-binary", type=Path, required=True)
    parser.add_argument("--xsched-server", type=Path, required=True)
    parser.add_argument("--xsched-shim", type=Path, required=True)
    parser.add_argument("--xsched-patch", type=Path, required=True)
    parser.add_argument("--application-output-trace", type=Path)
    parser.add_argument("--workload", choices=tuple(WORKLOADS), default="whisper-projection")
    parser.add_argument("--producer-engine", type=Path)
    parser.add_argument("--consumer-engine", type=Path)
    parser.add_argument("--consumer-input-tensor")
    parser.add_argument("--common-workload-contract", type=Path)
    parser.add_argument("--require-common-workload", action="store_true")
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = verify(
        args.result, args.pipeline, args.be_result, args.be_trace,
        args.server_log, args.client_log, args.deadline_lock,
        args.pipeline_binary, args.xsched_server, args.xsched_shim,
        args.xsched_patch, args.repo.resolve(),
        args.workload,
        args.producer_engine.resolve() if args.producer_engine else None,
        args.consumer_engine.resolve() if args.consumer_engine else None,
        args.consumer_input_tensor,
        args.application_output_trace.resolve() if args.application_output_trace else None,
        args.common_workload_contract.resolve() if args.common_workload_contract else None,
        args.require_common_workload,
    )
    args.output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
