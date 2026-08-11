#!/usr/bin/env python3
"""Strict verifier for the Orion TensorRT software-queue positive control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

UPSTREAM_COMMIT = "20f9469764fb96d94ce23a8e70615196e9ce4ba1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    require(raw.endswith(b"\n"), f"{path} is not newline-complete")
    value = json.loads(raw)
    require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    require(raw.endswith(b"\n"), f"{path} is not newline-complete")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        require(bool(line), f"{path}:{line_number} is empty")
        row = json.loads(line)
        require(isinstance(row, dict), f"{path}:{line_number} is not an object")
        rows.append(row)
    require(bool(rows), f"{path} contains no records")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_client(
    client: Any, *, priority: str, requests: int, mig_uuid: str
) -> None:
    require(isinstance(client, dict), f"{priority} client is not an object")
    require(client.get("schema_version") == 1, f"{priority} schema differs")
    require(client.get("model") == "resnet10-detection", f"{priority} model differs")
    require(client.get("completed_requests") == requests, f"{priority} request count differs")
    require(client.get("deadline_misses") == 0, f"{priority} has deadline misses")
    config = client.get("config")
    require(isinstance(config, dict), f"{priority} config is missing")
    require(config.get("priority") == priority, f"{priority} stream priority differs")
    environment = client.get("execution_environment")
    require(isinstance(environment, dict), f"{priority} environment is missing")
    require(environment.get("cuda_visible_devices") == mig_uuid, f"{priority} MIG differs")
    gpu = client.get("gpu")
    require(isinstance(gpu, dict), f"{priority} GPU is missing")
    require(gpu.get("multiprocessors") == 12, f"{priority} did not use the 2g instance")


def verify(
    result_path: Path,
    decision_path: Path,
    launch_path: Path,
    requests: int,
    mig_uuid: str,
) -> dict[str, Any]:
    result = load_json(result_path)
    require(result.get("schema_version") == 1, "result schema differs")
    require(result.get("kind") == "orion-thor-native-positive-control", "result kind differs")
    require(result.get("upstream_commit") == UPSTREAM_COMMIT, "Orion commit differs")
    require(result.get("port_stage") == "driver-operation-software-queue", "port stage differs")
    require(result.get("numeric_comparison_allowed") is False, "positive control cannot be numeric")
    verify_client(result.get("best_effort"), priority="low", requests=requests, mig_uuid=mig_uuid)
    verify_client(result.get("high_priority"), priority="high", requests=requests, mig_uuid=mig_uuid)

    decisions = load_jsonl(decision_path)
    arrivals: list[int] = []
    reordered_rows: list[dict[str, Any]] = []
    clients: set[int] = set()
    for index, row in enumerate(decisions):
        require(set(row) == {
            "schema_version", "decision_sequence", "arrival_sequence",
            "client_id", "priority", "api", "reordered",
            "profile_position", "resource_class", "sm_used",
            "profile_duration_us", "admission_reason",
            "active_sm_at_admission", "active_be_duration_us_at_admission",
            "high_priority_active_at_admission",
            "initial_gate_clients", "start_monotonic_ns",
            "end_monotonic_ns", "result",
        }, f"decision {index} schema differs")
        require(row["schema_version"] == 1, f"decision {index} schema differs")
        require(row["decision_sequence"] == index, f"decision {index} sequence differs")
        require(type(row["arrival_sequence"]) is int, f"decision {index} arrival is invalid")
        require(row["api"] in {"cuLaunchKernelEx", "cuLaunchKernelEx_ptsz"}, "non-driver launch")
        require(row["result"] == 0, f"decision {index} failed")
        require(row["initial_gate_clients"] == 2, "positive-control gate differs")
        require(
            row["profile_position"] == 0
            and row["resource_class"] == -1
            and row["sm_used"] == 0
            and row["profile_duration_us"] == 0
            and row["admission_reason"] == "unprofiled"
            and row["active_sm_at_admission"] == 0
            and row["active_be_duration_us_at_admission"] == 0
            and row["high_priority_active_at_admission"] is False,
            "positive-control profile fields differ",
        )
        require(row["end_monotonic_ns"] >= row["start_monotonic_ns"], "decision clocks regress")
        arrivals.append(row["arrival_sequence"])
        clients.add(row["client_id"])
        if row["reordered"]:
            reordered_rows.append(row)
    require(sorted(arrivals) == list(range(len(decisions))), "arrival sequence is not complete")
    require(clients == {0, 1}, "both Orion clients did not reach the queue")
    require(bool(reordered_rows), "Orion never changed FIFO order")
    for row in reordered_rows:
        decision = row["decision_sequence"]
        arrival = row["arrival_sequence"]
        require(any(later < arrival for later in arrivals[decision + 1 :]),
                "reported reordering has no later earlier arrival")

    launches = load_jsonl(launch_path)
    require(len(launches) == len(decisions), "launch and decision counts differ")
    for index, row in enumerate(launches):
        require(row.get("sequence") == index, f"launch {index} sequence differs")
        require(row.get("result") == 0, f"launch {index} failed")
        require(row.get("api") in {"cuLaunchKernelEx", "cuLaunchKernelEx_ptsz"}, "launch API differs")

    scheduler = result.get("scheduler")
    require(isinstance(scheduler, dict), "scheduler summary is missing")
    require(scheduler.get("arrivals") == len(decisions), "scheduler arrivals differ")
    require(scheduler.get("decisions") == len(decisions), "scheduler decisions differ")
    require(scheduler.get("reordered_decisions") == len(reordered_rows), "reorder count differs")
    return {
        "schema_version": 1,
        "kind": "orion-thor-native-positive-control-verification",
        "status": "passed",
        "numeric_comparison_allowed": False,
        "upstream_commit": UPSTREAM_COMMIT,
        "requests_per_client": requests,
        "driver_launches": len(launches),
        "reordered_decisions": len(reordered_rows),
        "result_sha256": sha256(result_path),
        "decision_trace_sha256": sha256(decision_path),
        "driver_trace_sha256": sha256(launch_path),
        "next_gate": "profile-aware Orion pairing with initial_gate_clients=0",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--launches", type=Path, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--mig-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.requests > 0, "--requests must be positive")
    verified = verify(args.result, args.decisions, args.launches, args.requests, args.mig_uuid)
    args.output.write_text(json.dumps(verified, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
