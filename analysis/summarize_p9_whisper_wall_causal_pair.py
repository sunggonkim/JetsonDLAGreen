#!/usr/bin/env python3
"""Validate an independent/dependent production-wall Whisper pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from .p9_frontier_evidence import validate_correctness
except ImportError:
    from p9_frontier_evidence import validate_correctness


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.resolve().read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{path} is not newline-complete")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not an object")
    return value, hashlib.sha256(raw).hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _raw_pipeline(summary_path: Path, row: dict[str, Any]) -> tuple[dict[str, Any], str]:
    trace = row.get("request_trace")
    if not isinstance(trace, dict) or not isinstance(trace.get("path"), str):
        raise ValueError(f"{summary_path} lacks pipeline trace provenance")
    trace_path = Path(trace["path"])
    candidates = []
    if trace_path.is_absolute():
        candidates.append(trace_path.parent / "pipeline.json")
    else:
        candidates.extend((summary_path.parent / trace_path.parent / "pipeline.json", Path.cwd() / trace_path.parent / "pipeline.json"))
    pipeline_path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if pipeline_path is None:
        raise ValueError(f"{summary_path} pipeline trace is missing")
    raw = pipeline_path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{pipeline_path} is not an object")
    return value, hashlib.sha256(raw).hexdigest()


def summarize(independent_path: Path, dependent_path: Path) -> dict[str, Any]:
    arms: dict[str, tuple[dict[str, Any], str, dict[str, Any], dict[str, Any], str]] = {}
    for expected, path in (("independent", independent_path), ("dependent", dependent_path)):
        value, digest = _load(path)
        if (
            value.get("kind") != "p9-dependent-small-stress-smoke"
            or value.get("workload") != "whisper-projection"
            or value.get("dependency_mode") != expected
            or value.get("latency_contract") != "production-wall-arrival-to-completion"
            or value.get("deadline_mode") != "wall"
            or value.get("checksum_mode") != "inline"
        ):
            raise ValueError(f"{path} is not the expected {expected} arm")
        rows = value.get("results")
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise ValueError(f"{path} must contain one result")
        row = rows[0]
        if row.get("system") != "QUIET" or row.get("payload_bytes") != 2_304_000:
            raise ValueError(f"{path} is not a QUIET Whisper arm")
        validate_correctness(value, row, path)
        raw, raw_digest = _raw_pipeline(path, row)
        if raw.get("dependency_mode") != expected:
            raise ValueError(f"{path} raw dependency mode differs")
        edge = raw.get("dependency_edge")
        if not isinstance(edge, dict) or edge.get("present") is not (expected == "dependent"):
            raise ValueError(f"{path} dependency-edge evidence differs")
        lock = value.get("deadline_lock")
        if not isinstance(lock, dict) or not isinstance(lock.get("sha256"), str):
            raise ValueError(f"{path} lacks deadline lock provenance")
        arms[expected] = (value, digest, row, raw, raw_digest)
    independent, independent_digest, irow, iraw, iraw_digest = arms["independent"]
    dependent, dependent_digest, drow, draw, draw_digest = arms["dependent"]
    for key in ("deadline_us", "iterations", "background_period_ms", "placement_variant"):
        if independent.get(key) != dependent.get(key):
            raise ValueError(f"causal arms differ in {key}")
    if independent["deadline_lock"]["sha256"] != dependent["deadline_lock"]["sha256"]:
        raise ValueError("causal arms use different deadline locks")
    # The dependency toggle is the only permitted raw-contract change. Timing,
    # checksums, and handoff fields are intentionally excluded because they are
    # the measured consequences of that toggle.
    derived_raw_fields = {
        "dependency_mode", "dependency_edge", "correctness_scope",
        "unique_policy_output_checksums", "handoff_us", "end_to_end_us",
        "stage_latency_us", "gate_us", "measurement_start_monotonic_ns",
        "measurement_end_monotonic_ns", "elapsed_seconds", "pipeline_rps",
    }
    common_raw_keys = (set(iraw) | set(draw)) - derived_raw_fields
    for key in common_raw_keys:
        if iraw.get(key) != draw.get(key):
            raise ValueError(f"causal raw contract differs in {key}")
    ip99 = _finite(irow.get("wall_pipeline_p99_us"), "independent p99")
    dp99 = _finite(drow.get("wall_pipeline_p99_us"), "dependent p99")
    istages = irow.get("stage_latency_us")
    dstages = drow.get("stage_latency_us")
    if not isinstance(istages, dict) or not isinstance(dstages, dict):
        raise ValueError("causal arms lack stage latency decomposition")
    stage_names = (
        "producer_compute_p99", "transport_ready_p99", "consumer_compute_p99",
        "edge_transport_p99", "output_verification_p99",
    )
    stage_p99 = {
        "independent": {name: _finite(istages.get(name), f"independent {name}") for name in stage_names},
        "dependent": {name: _finite(dstages.get(name), f"dependent {name}") for name in stage_names},
    }
    stage_delta = {
        name: stage_p99["dependent"][name] - stage_p99["independent"][name]
        for name in stage_names
    }
    wall_delta = dp99 - ip99
    edge_delta = stage_delta["edge_transport_p99"]
    return {
        "schema_version": 1,
        "kind": "p9-whisper-production-wall-causal-pair",
        "proposed_system": "QUIET",
        "workload": "whisper-projection",
        "payload_bytes": 2_304_000,
        "deadline_us": independent["deadline_us"],
        "deadline_lock_sha256": independent["deadline_lock"]["sha256"],
        "formal": False,
        "scope": "exploratory-same-workload-independent-vs-dependent; no-thermal-normalization",
        "independent": {
            "path": str(independent_path.resolve()), "sha256": independent_digest,
            "raw_pipeline_sha256": iraw_digest, "p99_us": ip99,
            "deadline_misses": irow["deadline_misses"],
            "edge_transport_p99_us": irow["stage_latency_us"]["edge_transport_p99"],
        },
        "dependent": {
            "path": str(dependent_path.resolve()), "sha256": dependent_digest,
            "raw_pipeline_sha256": draw_digest, "p99_us": dp99,
            "deadline_misses": drow["deadline_misses"],
            "edge_transport_p99_us": drow["stage_latency_us"]["edge_transport_p99"],
        },
        "delta_dependent_minus_independent": {
            "wall_p99_us": wall_delta,
            "edge_transport_p99_us": edge_delta,
            "transport_ready_p99_us": stage_delta["transport_ready_p99"],
        },
        "stage_p99_us": stage_p99,
        "causal_interpretation": {
            "dominant_observed_wait_stage": "transport_ready",
            "edge_transport_fraction_of_wall_delta": edge_delta / wall_delta if wall_delta > 0 else None,
            "warning": "p99 stage values are marginal percentiles and are not additive",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--dependent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = summarize(args.independent, args.dependent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
