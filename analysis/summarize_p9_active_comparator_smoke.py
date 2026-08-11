#!/usr/bin/env python3
"""Validate a fast, same-contract QUIET/MPS/XSched comparison.

This is an exploratory smoke aggregator.  It deliberately refuses to rank a
row unless all three inputs share the workload, topology, wall deadline lock,
request count, and inline correctness contract.  Formal SLO/session claims
remain the responsibility of the Williams/session analyzers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


SYSTEMS = ("QUIET", "NVIDIA MPS", "XSched")
PRESENTATION_LABELS = {
    "QUIET": "QUIET",
    "NVIDIA MPS": "NVIDIA MPS baseline",
    "XSched": "XSched (Thor port)",
}


def _load_summary(path: Path) -> dict[str, Any]:
    summary_path = path / "summary.json"
    try:
        value = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load summary: {summary_path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"summary must be an object: {summary_path}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _bound_file(value: Any, label: str, *, capture_boundary: str | None = None) -> dict[str, Any] | None:
    """Validate an optional path/SHA evidence object without trusting metadata."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{label}: evidence must be an object")
    path = value.get("path")
    digest = value.get("sha256")
    if (
        not isinstance(path, str) or not path
        or not isinstance(digest, str) or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{label}: invalid path or SHA-256")
    if capture_boundary is not None and value.get("capture_boundary") != capture_boundary:
        raise ValueError(f"{label}: capture boundary differs")
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"{label}: referenced file is missing")
    actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if actual != digest:
        raise ValueError(f"{label}: SHA-256 mismatches bytes")
    result = {"path": str(resolved), "sha256": digest}
    if capture_boundary is not None:
        result["capture_boundary"] = capture_boundary
    return result


def _arm(summary: dict[str, Any], expected_system: str, source: Path) -> dict[str, Any]:
    if summary.get("kind") != "p9-dependent-small-stress-smoke":
        raise ValueError(f"{expected_system}: unexpected summary kind")
    if summary.get("system") is not None:
        raise ValueError(f"{expected_system}: system must be nested in results")
    if summary.get("claim_status", "").startswith("diagnostic-only"):
        raise ValueError(f"{expected_system}: diagnostic-only evidence cannot be aggregated")
    results = summary.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{expected_system}: summary results are missing")
    selected = [
        item for item in results
        if isinstance(item, dict) and item.get("system") == expected_system
    ]
    if len(selected) != 1:
        raise ValueError(f"{expected_system}: summary must contain one selected result")
    row = selected[0]
    common_workload = summary.get("common_workload")
    if common_workload is not None:
        required_workload_keys = (
            "schema_version", "workload_id", "topology", "placement",
            "input_tensor", "payload_bytes", "arrival_trace_path",
            "arrival_trace_sha256", "dataset_manifest_path",
            "dataset_manifest_sha256", "contract_path", "contract_sha256",
        )
        if (
            not isinstance(common_workload, dict)
            or any(key not in common_workload for key in required_workload_keys)
            or common_workload.get("schema_version") != 1
        ):
            raise ValueError(f"{expected_system}: common workload contract is malformed")
    if row.get("correctness_validated") is not True:
        raise ValueError(f"{expected_system}: inline correctness was not validated")
    if row.get("checksum_mode") != "inline":
        raise ValueError(f"{expected_system}: checksum mode is not inline")
    if row.get("latency_contract") != "production-wall-arrival-to-completion":
        raise ValueError(f"{expected_system}: latency contract is not production-wall")
    if row.get("deadline_mode") != "wall":
        raise ValueError(f"{expected_system}: deadline mode is not wall")
    if row.get("production_wall_definition") != "arrival-to-consumer-completion-excludes-correctness-validation":
        raise ValueError(f"{expected_system}: production-wall definition differs")
    if row.get("correctness_validation_placement") != "post-completion":
        raise ValueError(f"{expected_system}: correctness validation is inside the wall interval")
    requests = row.get("pipeline_requests")
    misses = row.get("deadline_misses")
    if (
        isinstance(requests, bool) or not isinstance(requests, int) or requests <= 0
        or isinstance(misses, bool) or not isinstance(misses, int)
        or misses < 0 or misses > requests
    ):
        raise ValueError(f"{expected_system}: invalid request/miss counts")
    lock = summary.get("deadline_lock")
    if not isinstance(lock, dict) or not isinstance(lock.get("sha256"), str):
        raise ValueError(f"{expected_system}: deadline lock provenance is missing")
    if len(lock["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in lock["sha256"]):
        raise ValueError(f"{expected_system}: deadline lock SHA is invalid")
    lock_path = Path(lock.get("path", ""))
    if not lock_path.is_file():
        raise ValueError(f"{expected_system}: referenced deadline lock is missing")
    actual_lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    if actual_lock_sha != lock["sha256"]:
        raise ValueError(f"{expected_system}: referenced deadline lock SHA mismatches bytes")
    sota = row.get("sota_verification")
    if expected_system == "XSched":
        if not isinstance(sota, dict) or sota.get("path") is None or sota.get("sha256") is None:
            raise ValueError("XSched: native verification provenance is missing")
        verification_path = Path(sota["path"])
        if not verification_path.is_file():
            raise ValueError("XSched: native verification file is missing")
        if hashlib.sha256(verification_path.read_bytes()).hexdigest() != sota["sha256"]:
            raise ValueError("XSched: native verification SHA mismatches bytes")
    request_trace = _bound_file(row.get("request_trace"), f"{expected_system}: request trace")
    output_trace = _bound_file(
        row.get("application_output_trace"),
        f"{expected_system}: application output trace",
        capture_boundary="post-completion",
    )
    return {
        "system": expected_system,
        "source": str(source.resolve()),
        "workload": summary.get("workload"),
        "placement_variant": summary.get("placement_variant"),
        "deadline_us": _finite(summary.get("deadline_us"), f"{expected_system}.deadline_us"),
        "deadline_lock": lock,
        "requests": requests,
        "misses": misses,
        "observed_dmr": misses / requests,
        "p99_us": _finite(row.get("wall_pipeline_p99_us"), f"{expected_system}.p99"),
        "background_goodput_rps": _finite(
            row.get("background_goodput_rps"), f"{expected_system}.goodput"
        ),
        "unique_payload_checksums": row.get("unique_payload_checksums"),
        "unique_policy_output_checksums": row.get("unique_policy_output_checksums"),
        "sota_verification": row.get("sota_verification"),
        "input_contract": {
            "payload_bytes": row.get("payload_bytes"),
            "consumer_input_tensor": row.get("consumer_input_tensor"),
            "consumer_engine_mode": row.get("consumer_engine_mode"),
            "consumer_engine": row.get("consumer_engine"),
            "producer_sms": row.get("producer_sms"),
            "consumer_sms": row.get("consumer_sms"),
        },
        "request_trace": request_trace,
        "application_output_trace": output_trace,
        "common_workload": common_workload,
    }


def summarize(
    quiet_dir: Path, mps_dir: Path, xsched_dir: Path,
) -> dict[str, Any]:
    arms = {
        "QUIET": _arm(_load_summary(quiet_dir), "QUIET", quiet_dir),
        "NVIDIA MPS": _arm(_load_summary(mps_dir), "NVIDIA MPS", mps_dir),
        "XSched": _arm(_load_summary(xsched_dir), "XSched", xsched_dir),
    }
    first = arms["QUIET"]
    for name in SYSTEMS[1:]:
        current = arms[name]
        for key in ("workload", "placement_variant", "requests"):
            if current[key] != first[key]:
                raise ValueError(f"{name}: common contract differs at {key}")
        if not math.isclose(current["deadline_us"], first["deadline_us"], rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"{name}: deadline differs")
        if current["deadline_lock"].get("sha256") != first["deadline_lock"].get("sha256"):
            raise ValueError(f"{name}: deadline lock SHA differs")
        if current["input_contract"] != first["input_contract"]:
            raise ValueError(f"{name}: workload engine or MIG capacity differs")
        if (current["common_workload"] is None) != (first["common_workload"] is None):
            raise ValueError(f"{name}: common workload evidence is incomplete")
        if (
            current["common_workload"] is not None
            and current["common_workload"] != first["common_workload"]
        ):
            raise ValueError(f"{name}: common workload contract differs")
        if (current["request_trace"] is None) != (first["request_trace"] is None):
            raise ValueError(f"{name}: request trace evidence is incomplete")
        if (current["application_output_trace"] is None) != (first["application_output_trace"] is None):
            raise ValueError(f"{name}: application output trace evidence is incomplete")
    return {
        "schema_version": 1,
        "kind": "p9-active-comparator-smoke-aggregate",
        "proposed_system": "QUIET",
        "systems": arms,
        "system_order": list(SYSTEMS),
        # Keep raw identifiers for replay, but expose one stable public
        # proposed-system name and descriptive comparator labels for tables.
        "presentation": {
            "proposed_system": "QUIET",
            "labels": {name: PRESENTATION_LABELS[name] for name in SYSTEMS},
            "headline_order": [PRESENTATION_LABELS[name] for name in SYSTEMS],
            "numeric_comparison_allowed": False,
        },
        "contract": {
            "workload": first["workload"],
            "placement_variant": first["placement_variant"],
            "deadline_us": first["deadline_us"],
            "deadline_lock_sha256": first["deadline_lock"]["sha256"],
            "requests": first["requests"],
            "latency_contract": "production-wall-arrival-to-completion",
            "checksum_mode": "inline",
            "deadline_mode": "wall",
            "input_contract": first["input_contract"],
            "common_workload": first["common_workload"],
            "request_trace_bound": first["request_trace"] is not None,
            "post_completion_output_trace_bound": first["application_output_trace"] is not None,
        },
        "formal": False,
        "claim_status": "exploratory-same-contract-no-session-or-thermal-certification",
        "ranking_allowed": False,
        "ranking_reason": (
            "common workload, request/output trace evidence, session-level SLO "
            "confidence, thermal normalization, and application accuracy remain pending"
            if (
                first["common_workload"] is None
                or first["request_trace"] is None
                or first["application_output_trace"] is None
            )
            else "session-level SLO confidence, thermal normalization, and application accuracy remain pending"
        ),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet-dir", type=Path, required=True)
    parser.add_argument("--mps-dir", type=Path, required=True)
    parser.add_argument("--xsched-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(args.quiet_dir, args.mps_dir, args.xsched_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
