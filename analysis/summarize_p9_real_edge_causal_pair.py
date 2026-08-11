#!/usr/bin/env python3
"""Bind an independent/dependent real-edge pipeline pair.

The independent arm runs the same producer and consumer TensorRT stages
concurrently while replaying the byte-identical producer activation captured
outside the measured interval.  The dependent arm consumes the producer's
live output tensor through the declared transport.  This is a characterization
artifact until repeated sessions are added.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _read(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.resolve().read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{path} is not newline-complete")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value, hashlib.sha256(raw).hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _sha(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def _trace_evidence(path: Path, expected_rows: int, label: str,
                    expected_header: str) -> dict[str, Any]:
    raw = path.resolve().read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{label} trace is not newline-complete")
    lines = raw.decode("utf-8").splitlines()
    if len(lines) != expected_rows + 1 or lines[0] != expected_header:
        raise ValueError(f"{label} trace row count differs")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
            "rows": expected_rows}


def summarize(
    independent_path: Path,
    dependent_path: Path,
    *,
    pipeline_binary: Path | None = None,
    producer_engine: Path | None = None,
    deadline_lock: Path | None = None,
    independent_trace: Path | None = None,
    dependent_trace: Path | None = None,
    independent_checksums: Path | None = None,
    dependent_checksums: Path | None = None,
) -> dict[str, Any]:
    independent, independent_sha = _read(independent_path)
    dependent, dependent_sha = _read(dependent_path)
    values = (independent, dependent)
    modes = ("independent", "dependent")
    for value, mode in zip(values, modes, strict=True):
        if value.get("status") != "ok" or value.get("dependency_mode") != mode:
            raise ValueError(f"{mode} pipeline result has the wrong status or mode")
        if value.get("pipeline") != "resnet10-layer7-cov-to-control-mlp":
            raise ValueError("causal pair must use the ResNet control DAG")
        if value.get("transport") != "registered-shared-sysmem-direct-binding":
            raise ValueError("causal pair transport differs")
        expected_scope = (
            "producer-output-consumer-input-equality"
            if mode == "dependent" else "producer-activation-replay-output-oracle"
        )
        if (
            value.get("checksum_mode") != "inline"
            or value.get("correctness_validated") is not True
            or value.get("correctness_scope") != expected_scope
        ):
            raise ValueError("causal pair requires inline correctness evidence")
        if value.get("checksum_failures") != 0 or value.get("unique_payload_checksums", 0) < 2:
            raise ValueError("causal pair payload correctness is incomplete")
        edge = value.get("dependency_edge")
        if not isinstance(edge, dict) or edge.get("payload_bytes") != 14_720:
            raise ValueError("causal pair lacks the real Layer7_cov payload contract")
        if edge.get("present") is not (mode == "dependent"):
            raise ValueError(f"{mode} edge presence differs from its declared mode")
        for key in ("producer_uuid", "consumer_uuid", "producer_sms", "consumer_sms",
                    "payload_bytes", "iterations", "warmup", "deadline_us"):
            if key not in value:
                raise ValueError(f"causal pair result lacks {key}")
        if value["producer_uuid"] == value["consumer_uuid"]:
            raise ValueError("causal pair requires distinct MIG instances")
    shared_keys = (
        "pipeline", "transport", "producer_uuid", "consumer_uuid", "producer_sms",
        "consumer_sms", "producer_quota", "consumer_quota", "payload_bytes",
        "producer_output_tensor", "consumer_input_tensor", "consumer_output_tensor",
        "payload_shape", "warmup", "iterations", "checksum_mode", "deadline_mode",
        "deadline_us",
    )
    for key in shared_keys:
        if independent.get(key) != dependent.get(key):
            raise ValueError(f"causal pair contract differs at {key}")
    if not math.isclose(
        _finite(independent["deadline_us"], "deadline"),
        _finite(dependent["deadline_us"], "deadline"),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("causal pair deadlines differ")
    if deadline_lock is not None:
        lock, lock_sha = _read(deadline_lock)
        if lock.get("kind") != "p9-dependent-pipeline-deadline-lock":
            raise ValueError("deadline lock kind differs")
        if not math.isclose(
            _finite(lock.get("deadline_us"), "locked deadline"),
            _finite(independent["deadline_us"], "deadline"),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("causal pair does not use the supplied deadline lock")
    else:
        lock_sha = None

    def row(value: dict[str, Any], raw_sha: str) -> dict[str, Any]:
        stage = value.get("stage_latency_us")
        end_to_end = value.get("end_to_end_us")
        if not isinstance(stage, dict) or not isinstance(end_to_end, dict):
            raise ValueError("pipeline timing summaries are missing")
        return {
            "dependency_mode": value["dependency_mode"],
            "iterations": value["iterations"],
            "deadline_misses": value["deadline_misses"],
            "deadline_miss_rate": value["deadline_misses"] / value["iterations"],
            "wall_p99_us": _finite(end_to_end.get("p99"), "wall p99"),
            "wall_max_us": _finite(end_to_end.get("max"), "wall max"),
            "producer_compute_p99_us": _finite(stage.get("producer_compute_p99"), "producer p99"),
            "consumer_compute_p99_us": _finite(stage.get("consumer_compute_p99"), "consumer p99"),
            "edge_transport_p99_us": _finite(stage.get("edge_transport_p99"), "edge p99"),
            "handoff_p99_us": _finite(value.get("handoff_us", {}).get("p99"), "handoff p99"),
            "checksum_failures": value["checksum_failures"],
            "unique_payload_checksums": value["unique_payload_checksums"],
            "unique_policy_output_checksums": value["unique_policy_output_checksums"],
            "input": {"path": str(Path(value_path).resolve()), "sha256": raw_sha},
        }

    # Keep the path associated with each raw object without trusting a JSON path.
    value_path = independent_path
    independent_row = row(independent, independent_sha)
    value_path = dependent_path
    dependent_row = row(dependent, dependent_sha)
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "p9-real-edge-causal-pair",
        "proposed_system": "QUIET",
        "scope": "same-resnet-control-stages-independent-local-input-vs-dependent-layer7-edge",
        "formal": False,
        "workload": {
            "producer_tensor": "Layer7_cov",
            "consumer_tensor": "features",
            "payload_bytes": 14_720,
            "payload_shape": [1, 4, 23, 40],
            "transport": "registered-shared-sysmem-direct-binding",
        },
        "shared_contract": {
            key: independent[key] for key in shared_keys
        },
        "independent": independent_row,
        "dependent": dependent_row,
        "delta_dependent_minus_independent": {
            "wall_p99_us": dependent_row["wall_p99_us"] - independent_row["wall_p99_us"],
            "wall_max_us": dependent_row["wall_max_us"] - independent_row["wall_max_us"],
            "edge_transport_p99_us": dependent_row["edge_transport_p99_us"] - independent_row["edge_transport_p99_us"],
        },
        "inputs": {
            "independent": {"path": str(independent_path.resolve()), "sha256": independent_sha},
            "dependent": {"path": str(dependent_path.resolve()), "sha256": dependent_sha},
        },
        "interpretation": (
            "This identifies the edge effect for the fixed ResNet control stages, "
            "MIG placement, transport, deadline, and request contract; it is not "
            "a general application or thermal claim."
        ),
    }
    if lock_sha is not None:
        result["deadline_lock"] = {
            "path": str(deadline_lock.resolve()),
            "sha256": lock_sha,
        }
    if pipeline_binary is not None or producer_engine is not None:
        if pipeline_binary is None or producer_engine is None:
            raise ValueError("pipeline binary and producer engine must be supplied together")
        result["artifacts"] = {
            "pipeline_binary": {"path": str(pipeline_binary.resolve()), "sha256": _sha(pipeline_binary)},
            "producer_engine": {"path": str(producer_engine.resolve()), "sha256": _sha(producer_engine)},
        }
    trace_args = (independent_trace, dependent_trace, independent_checksums, dependent_checksums)
    if any(path is not None for path in trace_args):
        if any(path is None for path in trace_args):
            raise ValueError("all four causal trace paths must be supplied together")
        result["raw_evidence"] = {
            "independent_trace": _trace_evidence(
                independent_trace, independent["iterations"], "independent",
                "request,producer_compute_us,producer_copy_us,producer_validation_us,notification_us,consumer_validation_us,consumer_copy_us,edge_transport_us,consumer_compute_us,output_verification_us,validation_excluded_end_to_end_us,wall_end_to_end_us,deadline_miss",
            ),
            "dependent_trace": _trace_evidence(
                dependent_trace, dependent["iterations"], "dependent",
                "request,producer_compute_us,producer_copy_us,producer_validation_us,notification_us,consumer_validation_us,consumer_copy_us,edge_transport_us,consumer_compute_us,output_verification_us,validation_excluded_end_to_end_us,wall_end_to_end_us,deadline_miss",
            ),
            "independent_checksums": _trace_evidence(
                independent_checksums, independent["iterations"], "independent checksum",
                "request,payload_checksum,output_checksum",
            ),
            "dependent_checksums": _trace_evidence(
                dependent_checksums, dependent["iterations"], "dependent checksum",
                "request,payload_checksum,output_checksum",
            ),
        }
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--dependent", type=Path, required=True)
    parser.add_argument("--pipeline-binary", type=Path)
    parser.add_argument("--producer-engine", type=Path)
    parser.add_argument("--deadline-lock", type=Path)
    parser.add_argument("--independent-trace", type=Path)
    parser.add_argument("--dependent-trace", type=Path)
    parser.add_argument("--independent-checksums", type=Path)
    parser.add_argument("--dependent-checksums", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(
        args.independent, args.dependent, pipeline_binary=args.pipeline_binary,
        producer_engine=args.producer_engine, deadline_lock=args.deadline_lock,
        independent_trace=args.independent_trace, dependent_trace=args.dependent_trace,
        independent_checksums=args.independent_checksums,
        dependent_checksums=args.dependent_checksums,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
