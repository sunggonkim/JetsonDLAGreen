#!/usr/bin/env python3
"""Compare QUIET placement characterization runs without hiding contract drift."""

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


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def compare(paths: Iterable[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    contract_values: list[tuple[Any, ...]] = []
    for path in paths:
        value, sha = _read(path)
        if value.get("kind") != "p9-dependent-small-stress-smoke":
            raise ValueError(f"{path} is not a QUIET placement smoke")
        if value.get("workload") != "resnet-control":
            raise ValueError(f"{path} is not the frozen placement workload")
        if value.get("latency_contract") != "production-wall-arrival-to-completion":
            raise ValueError(f"{path} is not production-wall evidence")
        if value.get("deadline_mode") != "wall" or value.get("checksum_mode") != "inline":
            raise ValueError(f"{path} is missing wall/inline correctness contract")
        result_rows = value.get("results")
        if not isinstance(result_rows, list) or len(result_rows) != 1:
            raise ValueError(f"{path} must contain exactly one result")
        row = result_rows[0]
        if not isinstance(row, dict) or row.get("system") != "QUIET":
            raise ValueError(f"{path} is not a QUIET result")
        requests = row.get("pipeline_requests")
        misses = row.get("deadline_misses")
        if not isinstance(requests, int) or isinstance(requests, bool) or requests <= 0:
            raise ValueError(f"{path} has invalid request count")
        if not isinstance(misses, int) or isinstance(misses, bool) or not 0 <= misses <= requests:
            raise ValueError(f"{path} has invalid miss count")
        placement = row.get("placement_variant", value.get("placement_variant"))
        if not isinstance(placement, str) or not placement:
            raise ValueError(f"{path} lacks placement variant")
        lock = value.get("deadline_lock")
        lock_sha = lock.get("sha256") if isinstance(lock, dict) else None
        deadline = _number(value.get("deadline_us"), "deadline_us")
        goodput = _number(row.get("background_goodput_rps"), "background_goodput_rps")
        p99 = _number(row.get("wall_pipeline_p99_us", row.get("pipeline_p99_us")), "wall p99")
        checksums = (row.get("unique_payload_checksums"), row.get("unique_policy_output_checksums"))
        if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in checksums):
            raise ValueError(f"{path} lacks checksum diversity")
        contract_values.append((value.get("iterations"), value.get("background_period_ms"),
                               value.get("background_offered_rps"), deadline,
                               lock_sha, value.get("workload"), value.get("checksum_mode")))
        rows.append({
            "placement_variant": placement,
            "deadline_us": deadline,
            "requests": requests,
            "deadline_misses": misses,
            "dmr": misses / requests,
            "wall_p99_us": p99,
            "background_goodput_rps": goodput,
            "slo_qualified": misses == 0 and p99 <= deadline,
            "correctness_validated": row.get("correctness_validated") is True,
            "unique_payload_checksums": checksums[0],
            "unique_policy_output_checksums": checksums[1],
            "input": {"path": str(path.resolve()), "sha256": sha},
        })
    if len(rows) < 2:
        raise ValueError("at least two placement runs are required")
    if len({row["placement_variant"] for row in rows}) != len(rows):
        raise ValueError("placement variants must be unique")
    same_contract = len(set(contract_values)) == 1
    deadlines = {contract[3] for contract in contract_values}
    lock_shas = {contract[4] for contract in contract_values}
    deadline_equal = len(deadlines) == 1
    lock_equal = len(lock_shas) == 1 and None not in lock_shas
    correctness_equal = all(row["correctness_validated"] for row in rows)
    all_slo_qualified = all(row["slo_qualified"] for row in rows)
    if same_contract and lock_equal and correctness_equal and not all_slo_qualified:
        comparison_status = "comparable-infeasible-candidates"
    elif same_contract and correctness_equal and lock_equal:
        comparison_status = "comparable"
    elif same_contract and lock_equal and not correctness_equal:
        comparison_status = "comparable-but-correctness-unvalidated"
    elif deadline_equal and not lock_equal:
        comparison_status = "same-deadline-unbound-lock"
    else:
        comparison_status = "not-comparable-contract-drift"
    by_p99 = sorted(rows, key=lambda row: row["wall_p99_us"])
    return {
        "schema_version": 1,
        "kind": "p9-quiet-placement-characterization",
        "proposed_system": "QUIET",
        "workload": "resnet-control",
        "latency_contract": "production-wall-arrival-to-completion",
        "checksum_mode": "inline",
        "formal": False,
        "scope": "exploratory-placement-direction; no-thermal-normalization",
        "contract_equal": same_contract,
        "deadline_equal": deadline_equal,
        "lock_provenance_equal": lock_equal,
        "correctness_contract_equal": correctness_equal,
        "all_candidates_slo_qualified": all_slo_qualified,
        "numeric_comparison_allowed": bool(
            same_contract and lock_equal and correctness_equal and all_slo_qualified
        ),
        "slo_comparison_status": comparison_status,
        "rows": rows,
        "best_wall_p99_placement": by_p99[0]["placement_variant"],
        "notes": [
            "A different deadline or lock is contract drift, not a SLO win.",
            "Both rows require inline checksum correctness before latency is reported.",
        ],
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
