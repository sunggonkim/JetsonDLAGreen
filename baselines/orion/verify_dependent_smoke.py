#!/usr/bin/env python3
"""Replay Orion's common dependent TensorRT numeric smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

from verify_profiled_smoke import TRACE_KEYS, load_json, load_jsonl, sha256


TRACE_COLUMNS = (
    "request", "producer_compute_us", "producer_copy_us",
    "input_sha256",
    "producer_validation_us", "notification_us", "consumer_validation_us",
    "consumer_copy_us", "edge_transport_us", "consumer_compute_us",
    "output_verification_us", "validation_excluded_end_to_end_us",
    "wall_end_to_end_us", "deadline_miss",
)


def load_profile_bundle(profile_path: Path, scheduler_path: Path) -> dict[str, Any]:
    profile = load_json(profile_path)
    scheduler = profile.get("scheduler_profile")
    operations = profile.get("operations")
    if (
        profile.get("kind") != "orion-thor-operation-profile"
        or not isinstance(scheduler, dict)
        or scheduler.get("schema") != "orion-thor-profile-v1"
        or sha256(scheduler_path) != scheduler.get("sha256")
        or not isinstance(operations, list)
        or not operations
    ):
        raise ValueError("Orion dependent profile bundle differs")
    lines = scheduler_path.read_text(encoding="ascii").splitlines()
    header = (
        "position\tapi\tgrid_x\tgrid_y\tgrid_z\tblock_x\tblock_y\tblock_z\t"
        "shared_mem_bytes\tprofile\tsm_used\tduration_us"
    )
    if lines[:2] != ["orion-thor-profile-v1", header] or len(lines) != len(operations) + 2:
        raise ValueError("Orion dependent scheduler profile shape differs")
    for index, (line, operation) in enumerate(zip(lines[2:], operations, strict=True)):
        expected = "\t".join(
            str(item)
            for item in (
                index, operation["api"], *operation["grid"], *operation["block"],
                operation["shared_mem_bytes"], operation["profile"],
                operation["sm_used"], f"{operation['duration_us']:.9g}",
            )
        )
        if line != expected:
            raise ValueError(f"Orion dependent scheduler profile row {index} differs")
    return profile


def verify_run_contract(
    path: Path,
    result: dict[str, Any],
    deadline_lock: dict[str, Any],
    deadline_lock_path: Path,
    be_profile_path: Path,
    hp_profile_path: Path,
    be_scheduler_path: Path,
    hp_scheduler_path: Path,
) -> dict[str, Any]:
    contract = load_json(path)
    orion = result.get("orion")
    if (
        contract.get("schema_version") != 1
        or contract.get("kind") != "orion-thor-dependent-run-contract"
        or contract.get("workload") != "whisper-projection"
        or contract.get("requests") != result.get("iterations")
        or contract.get("warmup") != result.get("warmup")
        or not isinstance(orion, dict)
        or contract.get("background_period_us") != orion.get("background_period_us")
        or contract.get("max_be_duration_source")
        != "frozen-isolated-pipeline-p99"
        or not math.isclose(
            float(contract.get("max_be_duration_us")),
            float(deadline_lock.get("pooled_p99_us")),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or contract.get("deadline_lock_sha256") != sha256(deadline_lock_path)
        or contract.get("best_effort_profile_sha256") != sha256(be_profile_path)
        or contract.get("high_priority_profile_sha256") != sha256(hp_profile_path)
        or contract.get("best_effort_scheduler_profile_sha256")
        != sha256(be_scheduler_path)
        or contract.get("high_priority_scheduler_profile_sha256")
        != sha256(hp_scheduler_path)
    ):
        raise ValueError("Orion dependent run contract differs")
    return contract


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def replay_pipeline(path: Path, result: dict[str, Any], deadline_us: float) -> dict[str, Any]:
    latencies: list[float] = []
    misses = 0
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != TRACE_COLUMNS:
            raise ValueError("Orion dependent pipeline trace schema differs")
        for index, row in enumerate(reader):
            if int(row["request"]) != int(result["warmup"]) + index:
                raise ValueError("Orion dependent request sequence differs")
            latency = float(row["validation_excluded_end_to_end_us"])
            wall = float(row["wall_end_to_end_us"])
            if not math.isfinite(latency) or latency <= 0.0 or wall < latency:
                raise ValueError("Orion dependent latency is invalid")
            miss = latency > deadline_us
            if int(row["deadline_miss"]) != int(miss):
                raise ValueError("Orion dependent deadline classification differs")
            misses += int(miss)
            latencies.append(latency)
    if len(latencies) != result.get("iterations") or misses != result.get("deadline_misses"):
        raise ValueError("Orion dependent request totals differ")
    p99 = percentile(latencies, 0.99)
    reported = result.get("stage_latency_us", {}).get(
        "validation_excluded_end_to_end_p99"
    )
    if not math.isclose(p99, float(reported), rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError("Orion dependent p99 differs from raw trace")
    return {"requests": len(latencies), "misses": misses, "p99_us": p99}


def replay_events(path: Path, profiles: list[dict[str, Any]],
                  scheduler: dict[str, Any]) -> dict[str, int]:
    rows = load_jsonl(path)
    if len(rows) != scheduler.get("trace_records"):
        raise ValueError("Orion event trace count differs")
    previous_decision = -1
    arrivals: set[int] = set()
    complementary = reordered = 0
    for row in rows:
        if set(row) != TRACE_KEYS or row.get("schema_version") != 1:
            raise ValueError("Orion dependent event schema differs")
        decision = row["decision_sequence"]
        arrival = row["arrival_sequence"]
        if (
            not isinstance(decision, int) or isinstance(decision, bool)
            or decision <= previous_decision
            or decision >= scheduler["decisions"]
            or not isinstance(arrival, int) or isinstance(arrival, bool)
            or arrival in arrivals or arrival >= scheduler["arrivals"]
        ):
            raise ValueError("Orion dependent event identity differs")
        previous_decision = decision
        arrivals.add(arrival)
        client = row["client_id"]
        if client not in (0, 1) or row["result"] != 0:
            raise ValueError("Orion dependent event execution failed")
        operations = profiles[client]["operations"]
        position = row["profile_position"]
        if not isinstance(position, int) or position < 0 or position >= len(operations):
            raise ValueError("Orion dependent profile position differs")
        operation = operations[position]
        if (
            row["api"] != operation["api"]
            or row["resource_class"] != operation["profile"]
            or row["sm_used"] != operation["sm_used"]
            or not math.isclose(float(row["profile_duration_us"]),
                                float(operation["duration_us"]),
                                rel_tol=1e-9, abs_tol=1e-6)
        ):
            raise ValueError("Orion dependent event profile differs")
        is_complementary = (
            row["admission_reason"] == "complementary-with-high-priority"
        )
        if not row["reordered"] and not is_complementary:
            raise ValueError("Orion event-only trace contains an ordinary decision")
        if is_complementary:
            if client != 0 or row["high_priority_active_at_admission"] is not True:
                raise ValueError("Orion complementary event lacks active HP")
            complementary += 1
        reordered += int(row["reordered"])
    if (
        complementary != scheduler.get("complementary_admissions")
        or reordered != scheduler.get("reordered_decisions")
    ):
        raise ValueError("Orion event summary differs from raw trace")
    return {"records": len(rows), "complementary": complementary,
            "reordered": reordered}


def load_deadline_lock(path: Path, repo: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(
        "freeze_p9_pipeline_deadline",
        repo / "analysis/freeze_p9_pipeline_deadline.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pipeline deadline verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lock = load_json(path)
    rebuilt = module.build_lock(Path(lock["source_summary"]))
    if lock != rebuilt:
        raise ValueError("pipeline deadline lock differs from source evidence")
    for artifact in lock["artifacts"].values():
        artifact_path = Path(artifact["path"])
        if sha256(artifact_path) != artifact["sha256"]:
            raise ValueError("deadline artifact changed after calibration")
    return lock


def verify(result_path: Path, pipeline_path: Path, events_path: Path,
           be_profile_path: Path, hp_profile_path: Path,
           be_scheduler_path: Path, hp_scheduler_path: Path,
           run_contract_path: Path, deadline_lock_path: Path,
           orion_binary: Path, repo: Path) -> dict[str, Any]:
    result = load_json(result_path)
    deadline_lock = load_deadline_lock(deadline_lock_path, repo)
    deadline_us = float(deadline_lock["deadline_us"])
    orion = result.get("orion")
    scheduler = orion.get("scheduler") if isinstance(orion, dict) else None
    if (
        result.get("status") != "ok"
        or result.get("pipeline") != "whisper-last-hidden-state-to-projection-mlp"
        or result.get("transport") != "registered-shared-sysmem-direct-binding"
        or result.get("payload_bytes") != 2_304_000
        or result.get("producer_sms") != 8 or result.get("consumer_sms") != 12
        or result.get("checksum_failures") != 0
        or result.get("unique_payload_checksums", 0) < 2
        or result.get("unique_policy_output_checksums", 0) < 2
        or result.get("deadline_mode") != "validation-excluded"
        or not math.isclose(float(result.get("deadline_us")), deadline_us,
                            rel_tol=0.0, abs_tol=1e-9)
        or not isinstance(orion, dict) or orion.get("enabled") is not True
        or orion.get("status") != 0
        or not isinstance(scheduler, dict)
        or scheduler.get("decisions") != scheduler.get("arrivals")
        or scheduler.get("complementary_admissions", 0) <= 0
    ):
        raise ValueError("Orion dependent result contract differs")
    start = orion["measurement_start_monotonic_ns"]
    end = orion["measurement_end_monotonic_ns"]
    measured = orion["measured_background_completed"]
    goodput = measured / ((end - start) / 1.0e9)
    if (
        end <= start or measured <= 0
        or not math.isclose(goodput, orion["measured_background_goodput_rps"],
                            rel_tol=1e-12, abs_tol=1e-9)
    ):
        raise ValueError("Orion background measurement window differs")
    profiles = [
        load_profile_bundle(be_profile_path, be_scheduler_path),
        load_profile_bundle(hp_profile_path, hp_scheduler_path),
    ]
    contract = verify_run_contract(
        run_contract_path, result, deadline_lock, deadline_lock_path,
        be_profile_path, hp_profile_path, be_scheduler_path, hp_scheduler_path,
    )
    pipeline = replay_pipeline(pipeline_path, result, deadline_us)
    events = replay_events(events_path, profiles, scheduler)
    return {
        "schema_version": 1,
        "kind": "orion-dependent-whisper-numeric-smoke-verification",
        "system": "Orion",
        "functional_gate_passed": True,
        "numeric_smoke_valid": True,
        "formal_claim_allowed": False,
        "workload": "Whisper -> 2.304MB coherent edge -> projection",
        "background": "DistilBERT SST-2",
        "deadline_us": deadline_us,
        "requests": pipeline["requests"],
        "misses": pipeline["misses"],
        "dmr": pipeline["misses"] / pipeline["requests"],
        "p99_us": pipeline["p99_us"],
        "background_goodput_rps": goodput,
        "scheduler_events": events,
        "inputs": {
            "result_sha256": sha256(result_path),
            "pipeline_sha256": sha256(pipeline_path),
            "events_sha256": sha256(events_path),
            "deadline_lock_sha256": sha256(deadline_lock_path),
            "orion_binary_sha256": sha256(orion_binary),
            "best_effort_profile_sha256": sha256(be_profile_path),
            "high_priority_profile_sha256": sha256(hp_profile_path),
            "best_effort_scheduler_profile_sha256": sha256(be_scheduler_path),
            "high_priority_scheduler_profile_sha256": sha256(hp_scheduler_path),
            "run_contract_sha256": sha256(run_contract_path),
        },
        "run_contract": contract,
        "formal_next_gate": "counterbalanced repetitions under the frozen contract",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--best-effort-profile", type=Path, required=True)
    parser.add_argument("--high-priority-profile", type=Path, required=True)
    parser.add_argument("--best-effort-scheduler-profile", type=Path, required=True)
    parser.add_argument("--high-priority-scheduler-profile", type=Path, required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--deadline-lock", type=Path, required=True)
    parser.add_argument("--orion-binary", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = verify(args.result, args.pipeline, args.events,
                   args.best_effort_profile, args.high_priority_profile,
                   args.best_effort_scheduler_profile,
                   args.high_priority_scheduler_profile, args.run_contract,
                   args.deadline_lock, args.orion_binary, args.repo.resolve())
    args.output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
