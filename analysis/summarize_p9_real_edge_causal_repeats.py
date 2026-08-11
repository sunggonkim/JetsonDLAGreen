#!/usr/bin/env python3
"""Aggregate repeated real-edge causal pairs without fabricating formal CI."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import t as student_t

try:
    from .summarize_p9_real_edge_causal_pair import summarize
except ImportError:  # direct CLI execution from the analysis directory
    from summarize_p9_real_edge_causal_pair import summarize


def _read(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.resolve().read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{path} is not newline-complete")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value, hashlib.sha256(raw).hexdigest()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _learned_head_pair(independent_path: Path, dependent_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate a paired learned ResNet backbone/head smoke.

    The independent arm still executes the same externally-built head engine;
    only the producer-output-to-head-input dependency edge is disabled.  This
    keeps the causal contrast on dependency semantics rather than model,
    payload, MIG placement, or correctness-path changes.
    """
    independent, independent_sha = _read(independent_path)
    dependent, dependent_sha = _read(dependent_path)
    expected = {
        "kind": "p9-dependent-small-stress-smoke",
        "workload": "resnet-detection-head",
        "consumer_engine_mode": "external-trained-engine",
        "consumer_input_tensor": "Layer6_relu_Y",
        "production_wall_definition": "arrival-to-consumer-completion-excludes-correctness-validation",
        "correctness_validation_placement": "post-completion",
        "checksum_mode": "inline",
        "placement_variant": "fixed-1g-producer-2g-consumer",
    }
    for value, mode in ((independent, "independent"), (dependent, "dependent")):
        if any(value.get(key) != expected_value for key, expected_value in expected.items()):
            raise ValueError("learned-head causal arms do not share the production contract")
        if value.get("dependency_mode") != mode or value.get("schema_version") != 1:
            raise ValueError("learned-head causal arm has the wrong dependency/schema")
        rows = value.get("results")
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise ValueError("learned-head summary must contain exactly one result row")
        row = rows[0]
        if row.get("system") not in {"QUIET", "NVIDIA MPS"}:
            raise ValueError("learned-head pair is not a supported executable arm")
        if row.get("pipeline_requests") != value.get("iterations"):
            raise ValueError("learned-head request count differs from summary iterations")
        if row.get("deadline_misses") != 0:
            raise ValueError("learned-head smoke has deadline misses")
        if row.get("checksum_mode") != "inline" or row.get("correctness_validation_placement") != "post-completion":
            raise ValueError("learned-head correctness evidence is not inline/post-completion")
        if row.get("payload_bytes") != 1_884_160:
            raise ValueError("learned-head result payload differs")
        if row.get("consumer_engine_mode") != expected["consumer_engine_mode"]:
            raise ValueError("learned-head result is not backed by an external engine")
        for key in ("producer_uuid", "consumer_uuid", "producer_sms", "consumer_sms", "request_trace"):
            if key not in row:
                raise ValueError(f"learned-head result lacks {key}")
    shared_keys = (
        "workload", "consumer_engine_mode", "consumer_input_tensor", "placement_variant",
        "iterations", "deadline_us", "deadline_mode",
        "production_wall_definition", "correctness_validation_placement", "checksum_mode",
        # Treatment-independent controls must match for a causal edge pair.
        "transport", "quiet_gate_scope", "background_period_ms",
        "producer_quota_percent", "background_quota_percent", "consumer_engine",
    )
    for key in shared_keys:
        if independent.get(key) != dependent.get(key):
            raise ValueError(f"learned-head pair differs in {key}")
    ir, dr = independent["results"][0], dependent["results"][0]
    row_shared = (
        "system", "producer_uuid", "consumer_uuid", "producer_sms", "consumer_sms",
        "payload_bytes", "consumer_engine_mode", "consumer_input_tensor",
        "transport", "gate_scope", "producer_quota_percent", "background_quota_percent",
    )
    for key in row_shared:
        if ir.get(key) != dr.get(key):
            raise ValueError(f"learned-head result differs in {key}")
    if not isinstance(ir.get("stage_latency_us"), dict) or not isinstance(dr.get("stage_latency_us"), dict):
        raise ValueError("learned-head stage latency evidence is missing")

    def row(value: dict[str, Any], raw_sha: str) -> dict[str, Any]:
        result = value["results"][0]
        p99 = _finite(result.get("wall_pipeline_p99_us"), "learned-head wall p99")
        p50 = _finite(result["stage_latency_us"].get("validation_excluded_end_to_end_p50"), "learned-head wall p50")
        edge = _finite(result["stage_latency_us"].get("edge_transport_p99"), "learned-head edge p99")
        return {
            "system": result["system"],
            "dependency_mode": value["dependency_mode"],
            "iterations": value["iterations"],
            "deadline_misses": result["deadline_misses"],
            "deadline_miss_rate": result["deadline_misses"] / value["iterations"],
            "wall_p50_us": p50,
            "wall_p99_us": p99,
            "wall_max_us": None,
            "edge_transport_p99_us": edge,
            "background_goodput_rps": _finite(result.get("background_goodput_rps", 0.0), "learned-head goodput"),
            "request_trace": result["request_trace"],
            "input": {"path": str((independent_path if value is independent else dependent_path).resolve()), "sha256": raw_sha},
        }
    independent_row = row(independent, independent_sha)
    dependent_row = row(dependent, dependent_sha)
    contract = {key: independent.get(key) for key in shared_keys}
    contract.update({key: ir.get(key) for key in row_shared})
    return contract, independent_row, dependent_row


def summarize_repeats(independent: list[Path], dependent: list[Path]) -> dict[str, Any]:
    if not independent or len(independent) != len(dependent):
        raise ValueError("independent/dependent repeat counts must match")
    rows: list[dict[str, Any]] = []
    shared: dict[str, Any] | None = None
    learned_head = False
    for index, (ip, dp) in enumerate(zip(independent, dependent, strict=True), start=1):
        probe, _ = _read(ip)
        pair_inputs: dict[str, Any]
        if probe.get("workload") == "resnet-detection-head":
            learned_head = True
            contract, i, d = _learned_head_pair(ip, dp)
            pair_inputs = {
                "independent": {"path": str(ip.resolve()), "sha256": i["input"]["sha256"]},
                "dependent": {"path": str(dp.resolve()), "sha256": d["input"]["sha256"]},
            }
        else:
            pair = summarize(ip, dp)
            contract = pair["shared_contract"]
            i = pair["independent"]
            d = pair["dependent"]
            pair_inputs = pair["inputs"]
        if shared is None:
            shared = contract
        elif contract != shared:
            raise ValueError("causal repeat contract differs")
        rows.append({
            "repeat": index,
            "independent_p99_us": i["wall_p99_us"],
            "dependent_p99_us": d["wall_p99_us"],
            "delta_p99_us": d["wall_p99_us"] - i["wall_p99_us"],
            "independent_max_us": i.get("wall_max_us"),
            "dependent_max_us": d.get("wall_max_us"),
            "delta_max_us": (
                None if i.get("wall_max_us") is None or d.get("wall_max_us") is None
                else d["wall_max_us"] - i["wall_max_us"]
            ),
            "delta_edge_transport_p99_us": (
                d["edge_transport_p99_us"] - i["edge_transport_p99_us"]
            ),
            "independent_wall_p50_us": i.get("wall_p50_us"),
            "dependent_wall_p50_us": d.get("wall_p50_us"),
            "delta_wall_p50_us": (
                None if i.get("wall_p50_us") is None or d.get("wall_p50_us") is None
                else d["wall_p50_us"] - i["wall_p50_us"]
            ),
            "independent_deadline_miss_rate": i.get("deadline_miss_rate"),
            "dependent_deadline_miss_rate": d.get("deadline_miss_rate"),
            "inputs": pair_inputs,
        })
    deltas = [row["delta_p99_us"] for row in rows]
    mean_delta = _mean(deltas)
    if len(deltas) >= 2:
        sample_sd = statistics.stdev(deltas)
        standard_error = sample_sd / math.sqrt(len(deltas))
        t_critical = float(student_t.ppf(0.975, len(deltas) - 1))
        ci_half_width = t_critical * standard_error
        paired_ci = {
            "method": "paired-session-t-interval",
            "confidence": 0.95,
            "unit": "session-pair",
            "n": len(deltas),
            "mean_us": mean_delta,
            "sample_sd_us": sample_sd,
            "standard_error_us": standard_error,
            "t_critical": t_critical,
            "lower_us": mean_delta - ci_half_width,
            "upper_us": mean_delta + ci_half_width,
        }
    else:
        paired_ci = None
    result = {
        "schema_version": 1,
        "kind": "p9-real-edge-causal-repeats",
        "proposed_system": "QUIET",
        "formal": False,
        "scope": "exploratory-no-thermal-normalization",
        "repeat_count": len(rows),
        "shared_contract": shared,
        "rows": rows,
        "delta_p99_us": {
            "mean": mean_delta,
            "min": min(deltas),
            "max": max(deltas),
        },
        "paired_session_ci95_us": paired_ci,
        "statistical_unit": "paired-session",
        "claim_guard": (
            "Session interval is descriptive with the observed repeats only; no "
            "thermal normalization, formal SLO certification, or general application claim."
        ),
    }
    if learned_head:
        result["workload"] = "resnet-detection-head"
        result["causal_contract"] = (
            "same externally-trained ResNet10 backbone/head, fixed 1g+2g MIG, "
            "same production wall and inline correctness; only dependency edge toggles"
        )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent", action="append", type=Path, required=True)
    parser.add_argument("--dependent", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize_repeats(args.independent, args.dependent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
