#!/usr/bin/env python3
"""Verify XSched on the real ResNet10 Layer7_cov dependent pipeline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
XSCHED_UPSTREAM_COMMIT = "bd494cb7a72958cd11900243a0798df00d856c6e"
sys.path.insert(0, str(ROOT / "baselines" / "orion"))
from verify_profiled_smoke import load_json, sha256  # noqa: E402


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load verifier module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ORION = _load_module(
    "orion_resnet_control_verifier",
    ROOT / "baselines" / "orion" / "verify_resnet_control_smoke.py",
)
_XSCHED = _load_module(
    "xsched_dependent_verifier",
    ROOT / "baselines" / "xsched" / "verify_dependent_smoke.py",
)
replay_checksums = _ORION.replay_checksums
replay_wall_pipeline = _ORION.replay_wall_pipeline
replay_be_window = _XSCHED.replay_be_window
replay_scheduler = _XSCHED.replay_scheduler


def _load_common_workload_contract(path: Path) -> dict[str, Any]:
    """Load and rehash the immutable input contract used by every arm."""
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
    value["contract_path"] = str(resolved)
    value["contract_sha256"] = hashlib.sha256(raw).hexdigest()
    return value


def verify(
    result_path: Path, pipeline_path: Path, checksum_path: Path,
    be_result_path: Path, be_trace_path: Path, server_log_path: Path,
    client_logs: list[Path], pipeline_binary: Path, producer_engine: Path,
    background_engine: Path, xsched_server: Path, xsched_shim: Path,
    xsched_patch: Path, xsched_commit_path: Path, deadline_us: float,
    expected_requests: int | None = None,
    deadline_lock_path: Path | None = None,
    background_period_ms: float = 4.0,
    application_output_trace_path: Path | None = None,
    common_workload_contract: Path | None = None,
    require_common_workload: bool = False,
) -> dict[str, Any]:
    if not math.isfinite(deadline_us) or deadline_us <= 0.0:
        raise ValueError("XSched deadline must be positive and finite")
    if require_common_workload and common_workload_contract is None:
        raise ValueError("XSched verifier requires a common workload contract")
    common_workload = (
        _load_common_workload_contract(common_workload_contract)
        if common_workload_contract is not None else None
    )
    if common_workload is not None:
        expected = {
            "workload_id": "resnet-detection-head",
            "topology": "fixed-2g+1g",
            "placement": "fixed-1g-producer-2g-consumer",
            "input_tensor": "Layer6_relu_Y",
            "payload_bytes": 1_884_160,
        }
        if any(common_workload.get(k) != v for k, v in expected.items()):
            raise ValueError("XSched common workload contract differs")
    result = load_json(result_path)
    be = load_json(be_result_path)
    deadline_lock = None
    if deadline_lock_path is not None:
        lock_bytes = deadline_lock_path.read_bytes()
        lock = json.loads(lock_bytes)
        contract = lock.get("contract", {})
        lock_kind = lock.get("kind")
        common_placements = set(contract.get("allowed_placements", []))
        if (
            lock_kind not in {
                "p9-dependent-pipeline-deadline-lock",
                "p9-common-placement-deadline-lock",
            }
            or contract.get("workload") != "resnet-control"
            or contract.get("deadline_mode") != "wall"
            or not math.isclose(float(lock.get("deadline_us")), deadline_us,
                                rel_tol=0.0, abs_tol=1e-9)
            or (
                lock_kind == "p9-common-placement-deadline-lock"
                and "fixed-1g-producer-2g-consumer" not in common_placements
            )
        ):
            raise ValueError("XSched deadline lock contract differs")
        deadline_lock = {
            "path": str(deadline_lock_path.resolve()),
            "sha256": sha256(deadline_lock_path),
        }
    upstream_commit = xsched_commit_path.read_text(encoding="utf-8").strip()
    if upstream_commit != XSCHED_UPSTREAM_COMMIT:
        raise ValueError("XSched numeric smoke does not use the pinned upstream")
    start_ns = result.get("measurement_start_monotonic_ns")
    end_ns = result.get("measurement_end_monotonic_ns")
    iterations = result.get("iterations")
    if (
        result.get("schema_version") != 1 or result.get("status") != "ok"
        or result.get("pipeline") != "resnet10-layer7-cov-to-control-mlp"
        or result.get("transport") != "registered-shared-sysmem-direct-binding"
        or result.get("payload_bytes") != 14_720
        or result.get("payload_shape") != [1, 4, 23, 40]
        or result.get("producer_output_tensor") != "Layer7_cov"
        or result.get("consumer_input_tensor") != "features"
        or result.get("consumer_output_tensor") != "policy_output"
        or result.get("warmup") != 10
        or isinstance(iterations, bool) or not isinstance(iterations, int)
        or iterations <= 0
        or (expected_requests is not None and iterations != expected_requests)
        or result.get("producer_sms") != 8 or result.get("consumer_sms") != 12
        or result.get("producer_quota") != 100 or result.get("consumer_quota") != 100
        or result.get("checksum_failures") != 0
        or result.get("unique_payload_checksums", 0) < 2
        or result.get("unique_policy_output_checksums", 0) < 2
        or result.get("deadline_mode") != "wall"
        or not math.isclose(float(result.get("deadline_us")), deadline_us,
                            rel_tol=0.0, abs_tol=1e-9)
        or not isinstance(start_ns, int) or isinstance(start_ns, bool)
        or not isinstance(end_ns, int) or isinstance(end_ns, bool)
        or end_ns <= start_ns
    ):
        raise ValueError("XSched ResNet control result contract differs")
    if result.get("producer_uuid") == result.get("consumer_uuid"):
        raise ValueError("XSched ResNet stages did not use distinct MIG instances")

    environment = be.get("execution_environment")
    config = be.get("config")
    gpu = be.get("gpu")
    if (
        be.get("model") != "distilbert-sst2" or be.get("role") != "benchmark"
        or not isinstance(environment, dict)
        or environment.get("mps_active_thread_percentage") != 100
        or not isinstance(config, dict) or config.get("warmup") != 20
        or config.get("burst_size") != 1
        or not math.isclose(float(config.get("period_ms")), background_period_ms,
                            rel_tol=0.0, abs_tol=1e-9)
        or config.get("priority") != "low" or config.get("include_transfers") is not True
        or not isinstance(gpu, dict) or gpu.get("multiprocessors") != 8
    ):
        raise ValueError("XSched ResNet background contract differs")
    for path in client_logs:
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in ("[XSCHED ERRO", "Assertion failed", "cuda error")):
            raise ValueError("XSched client recorded a runtime failure")

    pipeline = replay_wall_pipeline(pipeline_path, result, deadline_us)
    checksums = replay_checksums(checksum_path, result)
    background = replay_be_window(
        be_trace_path, be, start_ns, end_ns, background_period_ms
    )
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
        "kind": "xsched-thor-resnet-control-numeric-smoke-verification",
        "system": "XSched (Thor port)",
        "upstream_commit": upstream_commit,
        "status": "passed-smoke",
        "scope": "same-workload-smoke-not-formal-statistics",
        "workload": "resnet10-layer7-cov-to-control-mlp",
        "payload_bytes": 14_720,
        "requests": pipeline["requests"],
        "misses": pipeline["misses"],
        "dmr": pipeline["misses"] / pipeline["requests"],
        "p99_us": pipeline["p99_us"],
        "deadline_us": deadline_us,
        "deadline_mode": "wall",
        "latency_contract": "production-wall-arrival-to-completion",
        "production_wall_definition": (
            "arrival-to-consumer-completion-excludes-correctness-validation"
        ),
        "correctness_validation_placement": "post-completion",
        "deadline_lock": deadline_lock,
        "background_goodput_rps": background["completion_goodput_rps"],
        "background_period_ms": background_period_ms,
        "background_window": background,
        "checksum_failures": 0,
        "unique_payload_checksums": checksums["payloads"],
        "unique_policy_output_checksums": checksums["outputs"],
        "scheduler": scheduler,
        "application_output_trace": application_output_trace,
        "common_workload": common_workload,
        "inputs_sha256": {
            "result": sha256(result_path), "pipeline": sha256(pipeline_path),
            "checksums": sha256(checksum_path), "be_result": sha256(be_result_path),
            "be_trace": sha256(be_trace_path), "server_log": sha256(server_log_path),
            "pipeline_binary": sha256(pipeline_binary),
            "producer_engine": sha256(producer_engine),
            "background_engine": sha256(background_engine),
            "xsched_server": sha256(xsched_server), "xsched_shim": sha256(xsched_shim),
            "xsched_patch": sha256(xsched_patch),
            "xsched_commit": sha256(xsched_commit_path),
            "client_logs": {path.name: sha256(path) for path in client_logs},
        },
        "token_only": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "result", "pipeline", "checksums", "be-result", "be-trace", "server-log",
        "pipeline-binary", "producer-engine", "background-engine", "xsched-server",
        "xsched-shim", "xsched-patch",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--client-log", type=Path, action="append", default=[])
    parser.add_argument("--xsched-commit", type=Path, required=True)
    parser.add_argument("--deadline-us", type=float, required=True)
    parser.add_argument("--deadline-lock", type=Path)
    parser.add_argument("--background-period-ms", type=float, default=4.0)
    parser.add_argument("--application-output-trace", type=Path)
    parser.add_argument("--common-workload-contract", type=Path)
    parser.add_argument("--require-common-workload", action="store_true")
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = verify(
        args.result.resolve(), args.pipeline.resolve(), args.checksums.resolve(),
        args.be_result.resolve(), args.be_trace.resolve(), args.server_log.resolve(),
        [path.resolve() for path in args.client_log], args.pipeline_binary.resolve(),
        args.producer_engine.resolve(), args.background_engine.resolve(),
        args.xsched_server.resolve(), args.xsched_shim.resolve(),
        args.xsched_patch.resolve(), args.xsched_commit.resolve(), args.deadline_us,
        args.expected_requests, args.deadline_lock.resolve() if args.deadline_lock else None,
        args.background_period_ms,
        args.application_output_trace.resolve() if args.application_output_trace else None,
        args.common_workload_contract.resolve() if args.common_workload_contract else None,
        args.require_common_workload,
    )
    args.output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
