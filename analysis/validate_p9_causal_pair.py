#!/usr/bin/env python3
"""Validate an independent/dependent causal pair without changing workload identity.

The pair is intentionally stricter than a side-by-side table: the two summaries
must share engines, hardware/MIG layout, deadline, trace, CPU placement, and
arrival contract. Only the dependency scenario and its explicitly declared
semantics may differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


ALLOWED_CONFIG_DIFFERENCES = {
    "scenario",
    "fixed_deadline_ms",
    "deadline_source",
    "experiment_label",
    "dependency_semantics",
}
REQUIRED_SCENARIOS = {"independent", "dependent"}
DEFAULT_POLICY = "mig-governor"


def dependency_edges(summary: dict[str, Any], scenario: str, policy_name: str) -> list[dict[str, Any]]:
    """Return the replayed dependency edges and reject control-only claims.

    A completion pipe is useful for controller testing, but it is not evidence
    that the dependent workload consumed a producer tensor.  Causal motivation
    must therefore bind a positive payload size and an explicit transport.
    """
    item = policy(summary, policy_name)
    epochs = item.get("epochs")
    if not isinstance(epochs, list) or not epochs:
        raise ValueError("causal pair lacks epoch dependency evidence")
    edges: list[dict[str, Any]] = []
    for epoch in epochs:
        if not isinstance(epoch, dict):
            raise ValueError("causal pair epoch evidence is malformed")
        current = epoch.get("dependency_edges", [])
        if not isinstance(current, list):
            raise ValueError("causal pair dependency edges are malformed")
        for edge in current:
            if not isinstance(edge, dict):
                raise ValueError("causal pair dependency edge is malformed")
            edges.append(edge)
    if scenario == "independent":
        if edges:
            raise ValueError("independent causal arm contains dependency edges")
        return []
    if not edges:
        raise ValueError("dependent causal arm lacks a dependency edge")
    for edge in edges:
        payload_bytes = edge.get("payload_bytes")
        transport = edge.get("transport")
        if (
            isinstance(payload_bytes, bool)
            or not isinstance(payload_bytes, int)
            or payload_bytes <= 0
            or not isinstance(transport, str)
            or not transport
        ):
            raise ValueError(
                "dependent causal arm must prove a positive payload and transport"
            )
    return edges


def load(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{path} is not newline-complete")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    value["_sha256"] = hashlib.sha256(raw).hexdigest()
    return value


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def policy(summary: dict[str, Any], name: str) -> dict[str, Any]:
    policies = summary.get("policies")
    if not isinstance(policies, list):
        raise ValueError("summary lacks policies")
    matches = [item for item in policies if isinstance(item, dict) and item.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"summary must contain exactly one {name} policy")
    return matches[0]


def comparable_config(summary: dict[str, Any]) -> dict[str, Any]:
    config = summary.get("config")
    if not isinstance(config, dict):
        raise ValueError("summary lacks config")
    return {
        key: value for key, value in config.items()
        if key not in ALLOWED_CONFIG_DIFFERENCES
    }


def metric(item: dict[str, Any], key: str) -> float:
    return finite(item.get(key), f"policy.{key}")


def validate_pair(
    independent_path: Path,
    dependent_path: Path,
    policy_name: str = DEFAULT_POLICY,
) -> dict[str, Any]:
    independent = load(independent_path.resolve())
    dependent = load(dependent_path.resolve())
    summaries = (independent, dependent)
    configs = []
    edge_evidence: dict[str, list[dict[str, Any]]] = {}
    for summary, expected in zip(summaries, ("independent", "dependent"), strict=True):
        if summary.get("schema_version") not in {3, 4}:
            raise ValueError("causal pair requires schema version 3 or 4")
        config = summary.get("config")
        if not isinstance(config, dict) or config.get("scenario") != expected:
            raise ValueError(f"summary scenario is not {expected}")
        configs.append(config)
        # The production path, not legacy validation-excluded timings, is the
        # only admissible causal evidence.
        if config.get("includes_transfers") is not True:
            raise ValueError("causal pair must include transfers")
        if config.get("worker_max_inflight") != 1:
            raise ValueError("causal pair requires the same one-in-flight contract")
        finite(summary.get("deadline_ms"), "deadline_ms")
        policy(summary, policy_name)
        edge_evidence[expected] = dependency_edges(summary, expected, policy_name)

    if comparable_config(independent) != comparable_config(dependent):
        differences = [
            key for key in sorted(set(configs[0]) | set(configs[1]))
            if key not in ALLOWED_CONFIG_DIFFERENCES
            and configs[0].get(key) != configs[1].get(key)
        ]
        raise ValueError(f"causal pair workload contract differs: {differences}")
    for key in ("artifacts", "hardware", "mig"):
        if independent.get(key) != dependent.get(key):
            raise ValueError(f"causal pair {key} provenance differs")
    if not math.isclose(
        finite(independent["deadline_ms"], "independent deadline"),
        finite(dependent["deadline_ms"], "dependent deadline"),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("causal pair deadlines differ")

    rows = []
    for label, summary in (("independent", independent), ("dependent", dependent)):
        item = policy(summary, policy_name)
        requests = item.get("critical_requests")
        misses = item.get("deadline_misses")
        if (
            isinstance(requests, bool) or not isinstance(requests, int) or requests <= 0
            or isinstance(misses, bool) or not isinstance(misses, int)
            or not 0 <= misses <= requests
        ):
            raise ValueError(f"{label} policy request counts are invalid")
        miss_rate = metric(item, "deadline_miss_rate")
        if not math.isclose(miss_rate, misses / requests, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{label} deadline miss rate differs from counts")
        rows.append({
            "scenario": label,
            "critical_requests": requests,
            "deadline_misses": misses,
            "deadline_miss_rate": miss_rate,
            "critical_p99_ms": metric(item, "critical_p99_ms_max"),
            "pressure_goodput_per_second": metric(item, "pressure_goodput_per_second"),
            "telemetry_unhealthy_epochs": item.get("telemetry_unhealthy_epochs", 0),
            "rejected_tenants": item.get("rejected_tenants", 0),
        })
    if rows[0]["critical_requests"] != rows[1]["critical_requests"]:
        raise ValueError("causal pair request counts differ")

    independent_row, dependent_row = rows
    result = {
        "schema_version": 1,
        "kind": "p9-causal-independent-dependent-pair",
        "proposed_system": "QUIET",
        "policy_id": policy_name,
        "scope": "same-workload-edge-toggle",
        "independent": independent_row,
        "dependent": dependent_row,
        "edge_evidence": {
            "independent": edge_evidence["independent"],
            "dependent": edge_evidence["dependent"],
        },
        "delta_dependent_minus_independent": {
            "deadline_miss_rate": dependent_row["deadline_miss_rate"] - independent_row["deadline_miss_rate"],
            "critical_p99_ms": dependent_row["critical_p99_ms"] - independent_row["critical_p99_ms"],
            "pressure_goodput_per_second": dependent_row["pressure_goodput_per_second"] - independent_row["pressure_goodput_per_second"],
        },
        "shared_contract": {
            "deadline_ms": independent["deadline_ms"],
            "config": comparable_config(independent),
            "artifacts": independent["artifacts"],
            "hardware": independent["hardware"],
            "mig": independent["mig"],
        },
        "inputs": {
            "independent": {"path": str(independent_path.resolve()), "sha256": independent["_sha256"]},
            "dependent": {"path": str(dependent_path.resolve()), "sha256": dependent["_sha256"]},
        },
        "interpretation": (
            "dependency-edge effect is identified only for this fixed workload, "
            "placement, deadline, and arrival contract; it is not a general application claim"
        ),
    }
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--dependent", type=Path, required=True)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_pair(args.independent, args.dependent, args.policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
