#!/usr/bin/env python3
"""Summarize actual-payload registered, pinned, and pageable transports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


EXPECTED = {
    "registered": "registered-shared-sysmem-direct-binding",
    "pinned": "pinned-shared-sysmem-d2h-h2d",
    "pageable": "pageable-shared-sysmem-d2h-h2d",
}
RUNNER_TRANSPORT = {
    "registered": "registered-direct",
    "pinned": "pinned-bounce",
    "pageable": "pageable-bounce",
}


def _application_output_trace(raw: dict[str, Any], result_path: Path) -> dict[str, Any] | None:
    record = raw.get("application_output_trace")
    if record is None:
        return None
    if isinstance(record, str):
        trace_path = Path(record)
        if not trace_path.is_absolute():
            trace_path = result_path.parent / trace_path
        expected_sha = None
    elif isinstance(record, dict):
        path_value = record.get("path")
        expected_sha = record.get("sha256")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError("application output trace path is invalid")
        trace_path = Path(path_value)
        if not trace_path.is_absolute():
            trace_path = result_path.parent / trace_path
    else:
        raise ValueError("application output trace metadata is invalid")
    trace_path = trace_path.resolve()
    if not trace_path.is_file() or trace_path.stat().st_size <= 8:
        raise ValueError("application output trace is missing or empty")
    digest = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    if expected_sha is not None and expected_sha != digest:
        raise ValueError("application output trace SHA differs")
    return {
        "path": str(trace_path),
        "sha256": digest,
        "capture_boundary": "post-completion",
    }


def row(
    label: str,
    path: Path,
    *,
    require_application_output_trace: bool = False,
    expected_pipeline: str = "whisper-last-hidden-state-to-projection-mlp",
    expected_payload_bytes: int = 2_304_000,
) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if (
        raw.get("status") != "ok"
        or raw.get("pipeline") != expected_pipeline
        or raw.get("transport") != EXPECTED[label]
        or raw.get("payload_bytes") != expected_payload_bytes
        or raw.get("checksum_failures") != 0
        or raw.get("unique_payload_checksums", 0) < 2
        or raw.get("unique_policy_output_checksums", 0) < 2
    ):
        raise ValueError(f"invalid {label} large-edge evidence")
    application_output_trace = _application_output_trace(raw, path)
    if require_application_output_trace and application_output_trace is None:
        raise ValueError(f"{label} lacks a post-completion application output trace")
    stage = raw["stage_latency_us"]
    return {
        "transport": label,
        "pipeline": raw.get("pipeline"),
        "workload": raw.get("workload"),
        "payload_bytes": raw.get("payload_bytes"),
        "requests": raw.get("iterations") if isinstance(raw.get("iterations"), int) else None,
        "validation_excluded_p99_us": stage[
            "validation_excluded_end_to_end_p99"
        ],
        "wall_p99_us": raw["end_to_end_us"]["p99"],
        "producer_p99_us": stage["producer_compute_p99"],
        "consumer_p99_us": stage["consumer_compute_p99"],
        "producer_validation_p99_us": stage[
            "producer_payload_verification_p99"
        ],
        "consumer_validation_p99_us": stage[
            "consumer_payload_verification_p99"
        ],
        "notification_p99_us": stage["transport_notification_p99"],
        "producer_copy_p99_us": stage["producer_handoff_copy_p99"],
        "consumer_copy_p99_us": stage["consumer_handoff_copy_p99"],
        "edge_p99_sum_us": (
            stage["producer_handoff_copy_p99"]
            + stage["transport_notification_p99"]
            + stage["consumer_handoff_copy_p99"]
        ),
        "edge_p99_us": stage["edge_transport_p99"],
        "application_output_trace": application_output_trace,
        "input": {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    }


def _runner_row(
    label: str,
    path: Path,
    *,
    expected_workload: str,
    expected_payload_bytes: int,
    require_application_output_trace: bool = False,
) -> dict[str, Any]:
    """Validate a ``run_p9_dependent_stress_smoke.py`` summary row."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    results = raw.get("results")
    if (
        raw.get("kind") != "p9-dependent-small-stress-smoke"
        or raw.get("workload") != expected_workload
        or not isinstance(results, list)
        or len(results) != 1
        or (
            raw.get("transport") is not None
            and raw.get("transport")
            not in {RUNNER_TRANSPORT[label], label, EXPECTED[label]}
        )
    ):
        raise ValueError(f"invalid {label} runner evidence")
    result = results[0]
    expected_transport = EXPECTED[label]
    if (
        (
            result.get("transport") != expected_transport
            and not (label == "registered" and result.get("transport") is None)
        )
        or result.get("payload_bytes") != expected_payload_bytes
        or result.get("correctness_validated") is not True
        or result.get("deadline_misses") != 0
    ):
        raise ValueError(f"invalid {label} runner result")
    trace = _application_output_trace(result, path)
    if require_application_output_trace and trace is None:
        raise ValueError(f"{label} lacks a post-completion application output trace")
    stage = result["stage_latency_us"]
    return {
        "transport": label,
        "pipeline": None,
        "workload": expected_workload,
        "payload_bytes": expected_payload_bytes,
        "requests": result.get("pipeline_requests"),
        "validation_excluded_p99_us": stage["validation_excluded_end_to_end_p99"],
        "wall_p99_us": result["wall_pipeline_p99_us"],
        "producer_p99_us": stage["producer_compute_p99"],
        "consumer_p99_us": stage["consumer_compute_p99"],
        "notification_p99_us": stage["transport_notification_p99"],
        "producer_copy_p99_us": stage["producer_handoff_copy_p99"],
        "consumer_copy_p99_us": stage["consumer_handoff_copy_p99"],
        "edge_p99_sum_us": (
            stage["producer_handoff_copy_p99"]
            + stage["transport_notification_p99"]
            + stage["consumer_handoff_copy_p99"]
        ),
        "edge_p99_us": stage["edge_transport_p99"],
        "application_output_trace": trace,
        "input": {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    }


def summarize(
    paths: dict[str, Path], *, require_application_output_trace: bool = False,
) -> dict[str, Any]:
    rows = [
        row(label, paths[label], require_application_output_trace=require_application_output_trace)
        for label in EXPECTED
    ]
    same_raw = json.loads(paths["same_instance"].read_text(encoding="utf-8"))
    if (
        same_raw.get("status") != "ok"
        or same_raw.get("transport") != EXPECTED["registered"]
        or same_raw.get("producer_uuid") != same_raw.get("consumer_uuid")
        or same_raw.get("payload_bytes") != 2_304_000
        or same_raw.get("checksum_failures") != 0
    ):
        raise ValueError("invalid same-instance MPS evidence")
    same_application_output_trace = _application_output_trace(same_raw, paths["same_instance"])
    if require_application_output_trace and same_application_output_trace is None:
        raise ValueError("same-instance evidence lacks a post-completion application output trace")
    same_stage = same_raw["stage_latency_us"]
    direct = rows[0]["validation_excluded_p99_us"]
    return {
        "schema_version": 1,
        "kind": "p9-whisper-dependent-transport-ablation-smoke",
        "proposed_system": "QUIET",
        "payload_bytes": 2_304_000,
        "scope": (
            f"{rows[0]['requests']}-request-functional-smoke-not-formal-statistics"
            if isinstance(rows[0]["requests"], int)
            else "request-count-unavailable-functional-smoke-not-formal-statistics"
        ),
        "requests": rows[0]["requests"],
        "transports": rows,
        "registered_savings_us": {
            "vs_pinned": rows[1]["validation_excluded_p99_us"] - direct,
            "vs_pageable": rows[2]["validation_excluded_p99_us"] - direct,
        },
        "placement_control": {
            "cross_mig_registered_edge_p99_us": rows[0]["edge_p99_us"],
            "same_instance_mps_edge_p99_us": same_stage["edge_transport_p99"],
            "cross_mig_validation_excluded_p99_us": rows[0][
                "validation_excluded_p99_us"
            ],
            "same_instance_validation_excluded_p99_us": same_stage[
                "validation_excluded_end_to_end_p99"
            ],
            "same_instance_input": {
                "path": str(paths["same_instance"].resolve()),
                "sha256": hashlib.sha256(paths["same_instance"].read_bytes()).hexdigest(),
            },
            "same_instance_application_output_trace": same_application_output_trace,
        },
        "application_output_trace_required": require_application_output_trace,
        "application_trace_bound": all(
            item["application_output_trace"] is not None for item in rows
        ) and same_application_output_trace is not None,
    }


def summarize_transport_only(
    paths: dict[str, Path],
    *,
    workload: str,
    pipeline: str,
    payload_bytes: int,
    require_application_output_trace: bool = False,
) -> dict[str, Any]:
    """Summarize cross-MIG transport arms without inventing a same-instance arm."""

    rows = [
        _runner_row(
            label,
            paths[label],
            expected_workload=workload,
            expected_payload_bytes=payload_bytes,
            require_application_output_trace=require_application_output_trace,
        )
        for label in EXPECTED
    ]
    request_counts = {item["requests"] for item in rows}
    if len(request_counts) != 1:
        raise ValueError("transport rows do not use the same request count")
    direct = rows[0]["validation_excluded_p99_us"]
    return {
        "schema_version": 1,
        "kind": f"p9-{workload}-dependent-transport-ablation-smoke",
        "proposed_system": "QUIET",
        "workload": workload,
        "pipeline": pipeline,
        "payload_bytes": payload_bytes,
        "formal": False,
        "ranking_allowed": False,
        "claim_status": "same-workload-cross-MIG-transport-motivation-smoke",
        "scope": (
            f"{rows[0]['requests']}-request-cross-mig-transport-smoke-"
            "not-formal-statistics"
        ),
        "requests": rows[0]["requests"],
        "transports": rows,
        "registered_delta_us": {
            "pinned_minus_registered": (
                rows[1]["validation_excluded_p99_us"] - direct
            ),
            "pageable_minus_registered": (
                rows[2]["validation_excluded_p99_us"] - direct
            ),
        },
        "application_output_trace_required": require_application_output_trace,
        "application_trace_bound": all(
            item["application_output_trace"] is not None for item in rows
        ),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in EXPECTED:
        parser.add_argument(f"--{label}", type=Path, required=True)
    parser.add_argument("--same-instance", type=Path)
    parser.add_argument(
        "--transport-only",
        action="store_true",
        help="summarize only the three cross-MIG transport arms",
    )
    parser.add_argument("--workload", default="whisper-projection")
    parser.add_argument(
        "--pipeline", default="whisper-last-hidden-state-to-projection-mlp"
    )
    parser.add_argument("--payload-bytes", type=int, default=2_304_000)
    parser.add_argument("--require-application-output-trace", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {label: getattr(args, label).resolve() for label in EXPECTED}
    if args.transport_only:
        result = summarize_transport_only(
            {label: paths[label] for label in EXPECTED},
            workload=args.workload,
            pipeline=args.pipeline,
            payload_bytes=args.payload_bytes,
            require_application_output_trace=args.require_application_output_trace,
        )
    else:
        if args.same_instance is None:
            parser.error("--same-instance is required unless --transport-only is set")
        paths["same_instance"] = args.same_instance.resolve()
        result = summarize(
            paths,
            require_application_output_trace=args.require_application_output_trace,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
