#!/usr/bin/env python3
"""Aggregate an exploratory same-contract frontier for the learned DAG.

Zero misses at a 100-request point are useful for fast iteration, but they are
not a 0.05% SLO certification.  The output therefore keeps descriptive and
exact-CP95 qualification separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import beta


PRODUCTION_WALL = "arrival-to-consumer-completion-excludes-correctness-validation"


def _read(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.resolve().read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{path} is not newline-complete")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return value


def _cp95(misses: int, requests: int) -> float:
    if misses == requests:
        return 1.0
    return float(beta.ppf(0.95, misses + 1, requests - misses))


def summarize(paths: Iterable[Path], *, dmr_target: float = 0.0005) -> dict[str, Any]:
    paths = [path.resolve() for path in paths]
    if len(paths) != 6:
        raise ValueError("learned frontier requires exactly six summaries")
    target = _finite(dmr_target, "dmr_target")
    if target >= 1.0:
        raise ValueError("dmr_target must be below one")
    contract: tuple[Any, ...] | None = None
    engine_sha: str | None = None
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for path in paths:
        value, digest = _read(path)
        if (
            value.get("kind") != "p9-dependent-small-stress-smoke"
            or value.get("workload") != "resnet-detection-head"
            or value.get("dependency_mode") != "dependent"
            or value.get("latency_contract") != "production-wall-arrival-to-completion"
            or value.get("production_wall_definition") != PRODUCTION_WALL
            or value.get("correctness_validation_placement") != "post-completion"
            or value.get("checksum_mode") != "inline"
            or value.get("consumer_engine_mode") != "external-trained-engine"
            or value.get("consumer_input_tensor") != "Layer6_relu_Y"
            or value.get("placement_variant") != "fixed-1g-producer-2g-consumer"
        ):
            raise ValueError(f"{path} is outside the learned production contract")
        rows_value = value.get("results")
        if not isinstance(rows_value, list) or len(rows_value) != 1:
            raise ValueError(f"{path} must contain one result row")
        row = rows_value[0]
        system = row.get("system")
        if system not in {"QUIET", "NVIDIA MPS"}:
            raise ValueError(f"{path} has unsupported system {system!r}")
        offered = _finite(value.get("background_offered_rps"), "offered rps")
        if offered <= 0.0 or (system, offered) in seen:
            raise ValueError(f"{path} duplicates or lacks a positive offered load")
        seen.add((system, offered))
        requests = row.get("pipeline_requests")
        misses = row.get("deadline_misses")
        if not isinstance(requests, int) or isinstance(requests, bool) or requests <= 0:
            raise ValueError(f"{path} has invalid request count")
        if not isinstance(misses, int) or isinstance(misses, bool) or not 0 <= misses <= requests:
            raise ValueError(f"{path} has invalid miss count")
        if row.get("correctness_validated") is not True or row.get("checksum_failures", 0) != 0:
            raise ValueError(f"{path} lacks inline correctness evidence")
        if row.get("payload_bytes") != 1_884_160:
            raise ValueError(f"{path} payload differs")
        current_contract = (
            value.get("deadline_us"), value.get("deadline_mode"), value.get("checksum_mode"),
            value.get("consumer_engine_mode"), value.get("consumer_input_tensor"),
            value.get("placement_variant"), row.get("producer_uuid"), row.get("consumer_uuid"),
            row.get("producer_sms"), row.get("consumer_sms"), requests,
        )
        if contract is None:
            contract = current_contract
        elif current_contract != contract:
            raise ValueError(f"{path} differs from the learned frontier contract")
        engine = value.get("consumer_engine")
        if not isinstance(engine, dict) or not isinstance(engine.get("sha256"), str):
            raise ValueError(f"{path} lacks consumer engine provenance")
        if engine_sha is None:
            engine_sha = engine["sha256"]
        elif engine["sha256"] != engine_sha:
            raise ValueError(f"{path} uses a different consumer engine")
        rows.append({
            "system": system, "offered_rps": offered, "requests": requests,
            "deadline_misses": misses, "observed_dmr": misses / requests,
            "cp95_upper_dmr": _cp95(misses, requests),
            "descriptive_zero_miss": misses == 0,
            "cp95_slo_qualified": _cp95(misses, requests) <= target,
            "p99_us": _finite(row.get("wall_pipeline_p99_us"), "wall p99"),
            "goodput_rps": _finite(row.get("background_goodput_rps"), "goodput"),
            "input": {"path": str(path), "sha256": digest},
        })
    if {row["system"] for row in rows} != {"QUIET", "NVIDIA MPS"}:
        raise ValueError("learned frontier must contain QUIET and NVIDIA MPS")
    loads = {row["offered_rps"] for row in rows if row["system"] == "QUIET"}
    if len(loads) != 3 or loads != {row["offered_rps"] for row in rows if row["system"] == "NVIDIA MPS"}:
        raise ValueError("learned frontier load points are not paired")
    systems: dict[str, Any] = {}
    for system in ("NVIDIA MPS", "QUIET"):
        selected = sorted((row for row in rows if row["system"] == system), key=lambda row: row["offered_rps"])
        descriptive = [row for row in selected if row["descriptive_zero_miss"]]
        formal = [row for row in selected if row["cp95_slo_qualified"]]
        systems[system] = {
            "points": selected,
            "descriptive_max_offered_rps": max((row["offered_rps"] for row in descriptive), default=None),
            "descriptive_max_goodput_rps": max((row["goodput_rps"] for row in descriptive), default=None),
            "formal_cp95_max_offered_rps": max((row["offered_rps"] for row in formal), default=None),
            "formal_cp95_max_goodput_rps": max((row["goodput_rps"] for row in formal), default=None),
        }
    return {
        "schema_version": 1,
        "kind": "p9-real-learned-dependent-frontier",
        "proposed_system": "QUIET",
        "workload": "resnet-detection-head",
        "systems": systems,
        "deadline_us": contract[0],
        "deadline_mode": contract[1],
        "dmr_target": target,
        "consumer_engine_sha256": engine_sha,
        "formal": False,
        "scope": "exploratory-same-contract-no-thermal-normalization-no-accuracy-gate",
        "claim_guard": "descriptive zero-miss points are not CP95 SLO certification at 100 requests per point",
        "statistical_unit": "request-point; session repetition required for formal promotion",
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--dmr-target", type=float, default=0.0005)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(args.input, dmr_target=args.dmr_target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
