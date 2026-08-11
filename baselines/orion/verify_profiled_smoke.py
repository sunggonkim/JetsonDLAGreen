#!/usr/bin/env python3
"""Replay a Thor profile-aware Orion positive control from raw evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


UPSTREAM_COMMIT = "20f9469764fb96d94ce23a8e70615196e9ce4ba1"
TRACE_KEYS = {
    "schema_version", "decision_sequence", "arrival_sequence", "client_id",
    "priority", "api", "reordered", "profile_position", "resource_class",
    "sm_used", "profile_duration_us", "admission_reason",
    "active_sm_at_admission", "active_be_duration_us_at_admission",
    "high_priority_active_at_admission", "initial_gate_clients",
    "start_monotonic_ns", "end_monotonic_ns", "result",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        raise ValueError("Orion decision trace is empty")
    return rows


def load_profile(path: Path) -> dict[str, Any]:
    value = load_json(path)
    scheduler = value.get("scheduler_profile")
    operations = value.get("operations")
    if (
        value.get("kind") != "orion-thor-operation-profile"
        or value.get("upstream_commit") != UPSTREAM_COMMIT
        or value.get("numeric_comparison_allowed") is not False
        or not isinstance(scheduler, dict)
        or scheduler.get("schema") != "orion-thor-profile-v1"
        or not isinstance(operations, list)
        or not operations
    ):
        raise ValueError("invalid Orion profile bundle")
    profile_path = Path(scheduler.get("path", "")).resolve()
    if not profile_path.is_file() or sha256(profile_path) != scheduler.get("sha256"):
        raise ValueError("Orion scheduler profile hash differs")
    lines = profile_path.read_text(encoding="ascii").splitlines()
    header = (
        "position\tapi\tgrid_x\tgrid_y\tgrid_z\tblock_x\tblock_y\tblock_z\t"
        "shared_mem_bytes\tprofile\tsm_used\tduration_us"
    )
    if lines[:2] != ["orion-thor-profile-v1", header] or len(lines) != len(operations) + 2:
        raise ValueError("Orion scheduler profile shape differs")
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
            raise ValueError(f"Orion scheduler profile row {index} differs")
    return value


def verify(result_path: Path, decisions_path: Path,
           best_effort_profile: Path, high_priority_profile: Path) -> dict[str, Any]:
    result = load_json(result_path)
    profiles = [load_profile(best_effort_profile), load_profile(high_priority_profile)]
    scheduler = result.get("scheduler")
    if (
        result.get("kind") != "orion-thor-profile-aware-positive-control"
        or result.get("upstream_commit") != UPSTREAM_COMMIT
        or result.get("port_stage") != "profile-aware-admission"
        or result.get("numeric_comparison_allowed") is not False
        or not isinstance(scheduler, dict)
        or scheduler.get("algorithm") != "orion-profile-aware"
        or scheduler.get("initial_gate_clients") != 0
        or scheduler.get("complementary_admissions", 0) <= 0
        or scheduler.get("profile_blocked_polls", 0) <= 0
    ):
        raise ValueError("profile-aware Orion result contract differs")
    for name in ("best_effort", "high_priority"):
        client = result.get(name)
        if not isinstance(client, dict) or client.get("completed_requests", 0) <= 0:
            raise ValueError(f"Orion {name} client did not complete")

    rows = load_jsonl(decisions_path)
    if scheduler.get("arrivals") != len(rows) or scheduler.get("decisions") != len(rows):
        raise ValueError("Orion decision count differs from raw trace")
    arrivals: set[int] = set()
    reordered = high = complementary = profiled_be = 0
    for index, row in enumerate(rows):
        if set(row) != TRACE_KEYS or row["schema_version"] != 1:
            raise ValueError(f"Orion decision {index} schema differs")
        if row["decision_sequence"] != index or row["initial_gate_clients"] != 0:
            raise ValueError(f"Orion decision {index} sequence differs")
        arrival = row["arrival_sequence"]
        if not isinstance(arrival, int) or isinstance(arrival, bool) or arrival in arrivals:
            raise ValueError(f"Orion decision {index} arrival identity differs")
        arrivals.add(arrival)
        client_id = row["client_id"]
        if client_id not in (0, 1) or row["result"] != 0:
            raise ValueError(f"Orion decision {index} execution failed")
        operation = profiles[client_id]["operations"][row["profile_position"]]
        if (
            row["api"] != operation["api"]
            or row["resource_class"] != operation["profile"]
            or row["sm_used"] != operation["sm_used"]
            or not math.isclose(row["profile_duration_us"], operation["duration_us"],
                                rel_tol=1e-9, abs_tol=1e-6)
        ):
            raise ValueError(f"Orion decision {index} profile differs")
        reason = row["admission_reason"]
        if client_id == 1:
            if row["priority"] != "high" or reason != "high-priority":
                raise ValueError(f"Orion HP decision {index} differs")
            high += 1
        else:
            if row["priority"] != "best-effort" or reason not in {
                "no-active-high-priority", "complementary-with-high-priority"
            }:
                raise ValueError(f"Orion BE decision {index} differs")
            profiled_be += 1
            if reason == "complementary-with-high-priority":
                if row["high_priority_active_at_admission"] is not True:
                    raise ValueError(f"Orion complementary decision {index} lacks active HP")
                complementary += 1
        reordered += int(row["reordered"])
    if arrivals != set(range(len(rows))):
        raise ValueError("Orion arrivals are not contiguous")
    expected = {
        "reordered_decisions": reordered,
        "high_priority_decisions": high,
        "profiled_best_effort_admissions": profiled_be,
        "complementary_admissions": complementary,
    }
    if any(scheduler.get(key) != value for key, value in expected.items()):
        raise ValueError("Orion scheduler summary differs from raw replay")
    return {
        "schema_version": 1,
        "kind": "orion-thor-profile-aware-verification",
        "upstream_commit": UPSTREAM_COMMIT,
        "functional_gate_passed": True,
        "numeric_comparison_allowed": False,
        "decisions": len(rows),
        "complementary_admissions": complementary,
        "result_sha256": sha256(result_path),
        "decision_trace_sha256": sha256(decisions_path),
        "profile_sha256": {
            "best_effort": sha256(best_effort_profile),
            "high_priority": sha256(high_priority_profile),
        },
        "next_gate": "same dependent TensorRT payload, arrival trace, and SLO",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--best-effort-profile", type=Path, required=True)
    parser.add_argument("--high-priority-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = verify(args.result, args.decisions, args.best_effort_profile,
                    args.high_priority_profile)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
