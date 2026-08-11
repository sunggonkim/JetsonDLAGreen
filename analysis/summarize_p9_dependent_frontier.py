#!/usr/bin/env python3
"""Summarize payload-valid dependent pipeline frontier smoke results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


LABELS = {
    "nvidia-mig-isolation": "NVIDIA MIG",
    "nvidia-mps-spatial-sharing": "NVIDIA MPS",
    "process-stop-ablation": "Process-stop ablation",
    "quiet": "QUIET",
}
PUBLIC_LABELS = set(LABELS.values())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def summarize(paths: list[Path], lookahead_us: float) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one input is required")
    lookahead_us = number(lookahead_us, "lookahead_us")
    rows: list[dict[str, Any]] = []
    deadline: float | None = None
    payload_bytes: int | None = None
    workload: str | None = None
    deadline_mode: str | None = None
    deadline_source: str | None = None
    deadline_lock: dict[str, str] | None = None
    seen_loads: set[float | None] = set()
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("kind") != "p9-dependent-small-stress-smoke":
            raise ValueError(f"wrong input kind: {path}")
        current_deadline = number(raw.get("deadline_us"), "deadline_us")
        current_workload = raw.get("workload", "resnet-control")
        if current_workload not in {"resnet-control", "whisper-projection"}:
            raise ValueError("unsupported dependent workload")
        if workload is None:
            workload = current_workload
        elif workload != current_workload:
            raise ValueError("frontier workloads differ")
        if deadline is None:
            deadline = current_deadline
        elif current_deadline != deadline:
            raise ValueError("frontier deadlines differ")
        current_source = raw.get("deadline_source")
        current_lock = raw.get("deadline_lock")
        if current_source is not None:
            if current_source != "frozen-independent-pipeline-p99-factor":
                raise ValueError("frontier deadline source is not frozen")
            if not isinstance(current_lock, dict) or set(current_lock) != {"path", "sha256"}:
                raise ValueError("frontier lacks deadline lock provenance")
            if deadline_source is None:
                deadline_source, deadline_lock = current_source, current_lock
            elif current_source != deadline_source or current_lock != deadline_lock:
                raise ValueError("frontier deadline locks differ")
        offered = raw.get("background_offered_rps")
        if offered is not None:
            offered = number(offered, "background_offered_rps")
        if offered in seen_loads:
            raise ValueError("duplicate offered load")
        seen_loads.add(offered)
        results = raw.get("results")
        result_names = {item.get("system") for item in results} if isinstance(results, list) else set()
        if result_names != set(LABELS) and result_names != PUBLIC_LABELS:
            raise ValueError("input does not contain the exact frontier systems")
        for item in results:
            current_payload = item.get("payload_bytes")
            if current_payload not in {14720, 2_304_000}:
                raise ValueError("frontier workload payload is unsupported")
            if payload_bytes is None:
                payload_bytes = current_payload
            elif payload_bytes != current_payload:
                raise ValueError("frontier workload payload differs")
            current_mode = item.get("deadline_mode", "wall")
            if current_mode not in {"wall", "validation-excluded"}:
                raise ValueError("frontier deadline mode is unsupported")
            if deadline_mode is None:
                deadline_mode = current_mode
            elif deadline_mode != current_mode:
                raise ValueError("frontier deadline modes differ")
            if item.get("unique_payload_checksums", 0) < 2 or item.get(
                "unique_policy_output_checksums", 0
            ) < 2:
                raise ValueError("frontier row lacks payload-dependent execution")
            requests = item.get("pipeline_requests")
            misses = item.get("deadline_misses")
            if (
                isinstance(requests, bool)
                or not isinstance(requests, int)
                or requests <= 0
                or isinstance(misses, bool)
                or not isinstance(misses, int)
                or not 0 <= misses <= requests
            ):
                raise ValueError("invalid request/miss counts")
            p99 = number(item.get("pipeline_p99_us"), "pipeline_p99_us")
            gate = item.get("gate_p99_us")
            gate = 0.0 if gate is None else number(gate, "gate_p99_us")
            uncovered = max(0.0, gate - lookahead_us)
            arrival_p99_bound = p99 + uncovered
            system_id = item["system"]
            public_system = LABELS.get(system_id, system_id)
            rows.append(
                {
                    "offered_rps": offered,
                    "system": public_system,
                    "role": "proposed" if public_system == "QUIET" else "baseline-or-ablation",
                    "requests": requests,
                    "misses": misses,
                    "dmr": misses / requests,
                    "post_release_p99_us": p99,
                    "gate_p99_us": gate if gate else None,
                    "lookahead_us": lookahead_us,
                    "uncovered_guard_us": uncovered,
                    "arrival_p99_bound_us": arrival_p99_bound,
                    "post_release_zero_miss": misses == 0 and p99 <= current_deadline,
                    "arrival_bound_feasible": misses == 0
                    and arrival_p99_bound <= current_deadline,
                    "background_goodput_rps": number(
                        item.get("background_goodput_rps"), "background goodput"
                    ),
                    "producer_compute_p99_us": number(
                        item.get("stage_latency_us", {}).get("producer_compute_p99"),
                        "producer compute p99",
                    ),
                }
            )
    rows.sort(
        key=lambda item: (
            math.inf if item["offered_rps"] is None else item["offered_rps"],
            item["system"],
        )
    )
    return {
        "schema_version": 1,
        "kind": "p9-dependent-payload-frontier-smoke",
        "proposed_system": "QUIET",
        "workload": workload,
        "payload_bytes": payload_bytes,
        "deadline_mode": deadline_mode,
        "deadline_us": deadline,
        "deadline_source": deadline_source,
        "deadline_lock": deadline_lock,
        "critical_lookahead_us": lookahead_us,
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in paths
        ],
        "rows": rows,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--lookahead-us", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize([path.resolve() for path in args.input], args.lookahead_us)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
