#!/usr/bin/env python3
"""Replay Orion's labelled ResNet-50/ImageNette production-wall smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baselines/orion"))
from verify_dependent_smoke import load_deadline_lock, replay_pipeline  # noqa: E402
from verify_profiled_smoke import load_json, load_profile, sha256  # noqa: E402


UPSTREAM_COMMIT = "20f9469764fb96d94ce23a8e70615196e9ce4ba1"
TRACE_KEYS = {
    "schema_version", "decision_sequence", "arrival_sequence", "client_id",
    "priority", "api", "reordered", "profile_position", "resource_class",
    "sm_used", "profile_duration_us", "admission_reason",
    "active_sm_at_admission", "active_be_duration_us_at_admission",
    "high_priority_active_at_admission", "initial_gate_clients",
    "start_monotonic_ns", "end_monotonic_ns", "result",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{path} is not newline complete")
    rows = [json.loads(line) for line in raw.splitlines()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} is empty or malformed")
    return rows


def replay_events(
    path: Path, profiles: list[dict[str, Any]], scheduler: dict[str, Any]
) -> dict[str, int]:
    rows = load_jsonl(path)
    if len(rows) != scheduler.get("trace_records"):
        raise ValueError("Orion event trace count differs")
    previous_decision = -1
    arrivals: set[int] = set()
    complementary = reordered = 0
    for index, row in enumerate(rows):
        if set(row) != TRACE_KEYS or row.get("schema_version") != 1:
            raise ValueError(f"Orion event {index} schema differs")
        decision = row["decision_sequence"]
        arrival = row["arrival_sequence"]
        if (
            not isinstance(decision, int) or isinstance(decision, bool)
            or decision <= previous_decision
            or decision >= scheduler["decisions"]
            or not isinstance(arrival, int) or isinstance(arrival, bool)
            or arrival in arrivals or arrival >= scheduler["arrivals"]
        ):
            raise ValueError(f"Orion event {index} identity differs")
        previous_decision = decision
        arrivals.add(arrival)
        client = row["client_id"]
        if client not in (0, 1) or row["result"] != 0:
            raise ValueError(f"Orion event {index} execution failed")
        operations = profiles[client]["operations"]
        position = row["profile_position"]
        if not isinstance(position, int) or position < 0 or position >= len(operations):
            raise ValueError(f"Orion event {index} profile position differs")
        operation = operations[position]
        if (
            row["api"] != operation["api"]
            or row["resource_class"] != operation["profile"]
            or row["sm_used"] != operation["sm_used"]
            or not math.isclose(
                float(row["profile_duration_us"]),
                float(operation["duration_us"]),
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
        ):
            raise ValueError(f"Orion event {index} profile differs")
        is_complementary = (
            row["admission_reason"] == "complementary-with-high-priority"
        )
        if not row["reordered"] and not is_complementary:
            raise ValueError("Orion event trace contains an ordinary decision")
        if is_complementary:
            if client != 0 or row["high_priority_active_at_admission"] is not True:
                raise ValueError("Orion complementary event lacks active HP")
            complementary += 1
        reordered += int(row["reordered"])
    if (
        complementary != scheduler.get("complementary_admissions")
        or reordered != scheduler.get("reordered_decisions")
    ):
        raise ValueError("Orion event summary differs from raw replay")
    return {
        "records": len(rows),
        "complementary_admissions": complementary,
        "reordered_decisions": reordered,
    }


def verify(
    result_path: Path,
    pipeline_path: Path,
    events_path: Path,
    best_effort_profile_path: Path,
    high_priority_profile_path: Path,
    deadline_lock_path: Path,
    common_workload_path: Path,
    repo: Path,
    binary_path: Path | None = None,
    producer_engine_path: Path | None = None,
    consumer_engine_path: Path | None = None,
    source_path: Path | None = None,
) -> dict[str, Any]:
    result = load_json(result_path)
    binary_path = (binary_path or repo / "build-r39/jdg-orion-mig-trt-pipeline").resolve()
    producer_engine_path = (
        producer_engine_path
        or repo / "results/p9-resnet50-imagenette-model-20260811/resnet50-imagenette-backbone.engine"
    ).resolve()
    consumer_engine_path = (
        consumer_engine_path
        or repo / "results/p9-resnet50-imagenette-model-20260811/resnet50-imagenette-head.engine"
    ).resolve()
    source_path = (
        source_path or repo / "baselines/orion/driver_capture/scheduler.cpp"
    ).resolve()
    for path in (binary_path, producer_engine_path, consumer_engine_path, source_path):
        if not path.is_file():
            raise ValueError(f"Orion provenance input is missing: {path}")
    lock = load_deadline_lock(deadline_lock_path, repo)
    deadline_us = float(lock["deadline_us"])
    common = load_json(common_workload_path)
    if (
        common.get("workload_id") != "resnet50-classification"
        or common.get("topology") != "fixed-2g+1g"
        or common.get("placement") != "fixed-1g-producer-2g-consumer"
        or common.get("input_tensor") != "gpu_0/res4_5_branch2c_bn_2"
        or common.get("payload_bytes") != 802816
    ):
        raise ValueError("Orion common workload contract differs")
    producer_input = Path(common["producer_input_trace_path"]).resolve()
    arrival_trace = Path(common["operational_arrival_trace_path"]).resolve()
    if (
        not producer_input.is_file()
        or sha256(producer_input) != common["producer_input_trace_sha256"]
        or not arrival_trace.is_file()
        or sha256(arrival_trace) != common["operational_arrival_trace_sha256"]
    ):
        raise ValueError("Orion common input or arrival trace is stale")
    orion = result.get("orion")
    scheduler = orion.get("scheduler") if isinstance(orion, dict) else None
    if (
        result.get("schema_version") != 1
        or result.get("status") != "ok"
        or result.get("pipeline") != "resnet50-backbone-to-classification-head"
        or result.get("transport") != "registered-shared-sysmem-direct-binding"
        or result.get("transport_description")
        != "full-coherent registered system-memory activation edge"
        or result.get("producer_sms") != 8
        or result.get("consumer_sms") != 12
        or result.get("payload_bytes") != 802816
        or result.get("payload_shape") != [1, 1024, 14, 14]
        or result.get("producer_output_tensor") != "gpu_0/res4_5_branch2c_bn_2"
        or result.get("consumer_input_tensor") != "gpu_0/res4_5_branch2c_bn_2"
        or result.get("consumer_output_tensor") != "external-output"
        or result.get("consumer_engine_mode") != "external-trained-engine"
        or result.get("dependency_mode") != "dependent"
        or result.get("deadline_mode") != "wall"
        or result.get("arrival_schedule_mode") != "operational-trace"
        or not math.isclose(float(result.get("deadline_us")), deadline_us, abs_tol=1e-9)
        or result.get("checksum_failures") != 0
        or result.get("correctness_validated") is not True
        or result.get("unique_payload_checksums", 0) < 2
        or result.get("unique_policy_output_checksums", 0) < 2
        or not isinstance(orion, dict)
        or orion.get("enabled") is not True
        or orion.get("status") != 0
        or not isinstance(scheduler, dict)
        or scheduler.get("decisions") != scheduler.get("arrivals")
        or scheduler.get("complementary_admissions", 0) <= 0
    ):
        raise ValueError("Orion ImageNette result contract differs")
    if (
        sha256(producer_input) != sha256(Path(result["producer_input_trace"]).resolve())
        or sha256(arrival_trace) != sha256(Path(result["arrival_trace"]).resolve())
    ):
        raise ValueError("Orion result trace binding differs")
    output_trace = Path(result["application_output_trace"]).resolve()
    if not output_trace.is_file():
        raise ValueError("Orion application output trace is missing")
    profiles = [
        load_profile(best_effort_profile_path),
        load_profile(high_priority_profile_path),
    ]
    if profiles[0].get("model") != "distilbert-sst2":
        raise ValueError("Orion BE profile identity differs")
    if profiles[1].get("model") != "resnet50-imagenette-backbone":
        raise ValueError("Orion HP profile identity differs")
    pipeline = replay_pipeline(pipeline_path, result, deadline_us)
    events = replay_events(events_path, profiles, scheduler)
    return {
        "schema_version": 1,
        "kind": "orion-dependent-resnet50-imagenette-numeric-smoke-verification",
        "system": "Orion (Thor port)",
        "workload": "resnet50-classification",
        "functional_gate_passed": True,
        "numeric_smoke_valid": True,
        "formal_claim_allowed": False,
        "formal_blockers": [
            "upstream-differential-fidelity",
            "counterbalanced-session-repetition",
            "thermal-normalization",
        ],
        "deadline_us": deadline_us,
        "requests": pipeline["requests"],
        "misses": pipeline["misses"],
        "dmr": pipeline["misses"] / pipeline["requests"],
        "p99_us": pipeline["p99_us"],
        "wall_p99_us": pipeline["p99_us"],
        "latency_contract": "production-wall-arrival-to-completion",
        "correctness_validation_placement": "post-completion",
        "background_goodput_rps": orion["measured_background_goodput_rps"],
        "scheduler_events": events,
        "application_output_trace": {
            "path": str(output_trace),
            "sha256": sha256(output_trace),
            "capture_boundary": "post-completion",
        },
        "common_workload": {
            "path": str(common_workload_path.resolve()),
            "sha256": sha256(common_workload_path),
            "input_trace_sha256": common["producer_input_trace_sha256"],
            "arrival_trace_sha256": common["operational_arrival_trace_sha256"],
            "dataset_manifest_sha256": common["dataset_manifest_sha256"],
        },
        "inputs": {
            "result_sha256": sha256(result_path),
            "pipeline_sha256": sha256(pipeline_path),
            "events_sha256": sha256(events_path),
            "deadline_lock_sha256": sha256(deadline_lock_path),
            "best_effort_profile_sha256": sha256(best_effort_profile_path),
            "high_priority_profile_sha256": sha256(high_priority_profile_path),
            "pipeline_binary_sha256": sha256(binary_path),
            "producer_engine_sha256": sha256(producer_engine_path),
            "consumer_engine_sha256": sha256(consumer_engine_path),
            "orion_source_sha256": sha256(source_path),
        },
        "upstream_commit": UPSTREAM_COMMIT,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--best-effort-profile", type=Path, required=True)
    parser.add_argument("--high-priority-profile", type=Path, required=True)
    parser.add_argument("--deadline-lock", type=Path, required=True)
    parser.add_argument("--common-workload", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--producer-engine", type=Path)
    parser.add_argument("--consumer-engine", type=Path)
    parser.add_argument("--orion-source", type=Path)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = verify(
        args.result.resolve(), args.pipeline.resolve(), args.events.resolve(),
        args.best_effort_profile.resolve(), args.high_priority_profile.resolve(),
        args.deadline_lock.resolve(), args.common_workload.resolve(),
        args.repo.resolve(),
        args.binary.resolve() if args.binary else None,
        args.producer_engine.resolve() if args.producer_engine else None,
        args.consumer_engine.resolve() if args.consumer_engine else None,
        args.orion_source.resolve() if args.orion_source else None,
    )
    args.output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
