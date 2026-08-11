#!/usr/bin/env python3
"""Build a QUIET stage-DAG candidate spec from measured payload smokes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


PLACEMENT_VARIANTS = {
    "fixed-1g-producer-2g-consumer": ("1g", "2g"),
    "fixed-2g-producer-1g-consumer": ("2g", "1g"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def candidate(
    summary_path: Path,
    output_root: Path,
    margin_us: float,
    fallback_deadline_us: float,
    expected_workload: str | None = None,
    expected_payload_bytes: int | None = None,
    expected_deadline_lock_sha256: str | None = None,
    allowed_placements: set[str] | None = None,
    formal: bool = False,
) -> dict[str, Any]:
    summary_path = summary_path.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("kind") != "p9-dependent-small-stress-smoke":
        raise ValueError("unexpected candidate summary kind")
    producer_quota = summary.get("producer_quota_percent")
    background_quota = summary.get("background_quota_percent")
    if producer_quota not in {10, 25, 50, 75, 90, 100}:
        raise ValueError("candidate lacks a supported producer quota")
    if background_quota not in {10, 25, 50, 75, 90, 100}:
        raise ValueError("candidate lacks a supported background quota")
    rows = [row for row in summary.get("results", []) if row.get("system") == "QUIET"]
    if len(rows) != 1:
        raise ValueError("candidate summary must contain exactly one QUIET row")
    row = rows[0]
    if row.get("producer_quota_percent") != producer_quota:
        raise ValueError("producer quota differs between summary and row")
    if row.get("background_quota_percent") != background_quota:
        raise ValueError("background quota differs between summary and row")
    pipeline_path = summary_path.parent / "quiet" / "pipeline.json"
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    if formal:
        if summary.get("checksum_mode") != "inline":
            raise ValueError("formal candidate requires inline checksums")
        if summary.get("deadline_mode") != "wall":
            raise ValueError("formal candidate requires wall deadline mode")
        if row.get("correctness_validated") is not True:
            raise ValueError("formal candidate lacks correctness validation")
        if row.get("checksum_failures", 0) != 0 or pipeline.get("checksum_failures") != 0:
            raise ValueError("formal candidate has checksum failures")
        for name in ("unique_payload_checksums", "unique_policy_output_checksums"):
            if not isinstance(row.get(name), int) or row[name] < 2:
                raise ValueError(f"formal candidate lacks {name}")
    if expected_workload is not None and summary.get("workload") != expected_workload:
        raise ValueError("candidate workload differs from deadline lock")
    if (
        expected_payload_bytes is not None
        and pipeline.get("payload_bytes") != expected_payload_bytes
    ):
        raise ValueError("candidate payload differs from deadline lock")
    if pipeline.get("iterations") != row.get("pipeline_requests"):
        raise ValueError("pipeline request count differs from summary")
    placement_variant = row.get(
        "placement_variant",
        summary.get("placement_variant", "fixed-1g-producer-2g-consumer"),
    )
    if not isinstance(placement_variant, str) or not placement_variant:
        raise ValueError("candidate placement variant is missing")
    if allowed_placements is not None and placement_variant not in allowed_placements:
        raise ValueError("candidate placement is outside the common deadline lock")
    if expected_deadline_lock_sha256 is not None:
        bound_lock = summary.get("deadline_lock")
        if (
            not isinstance(bound_lock, dict)
            or bound_lock.get("sha256") != expected_deadline_lock_sha256
        ):
            raise ValueError("candidate deadline lock differs from common lock")
    bound_lock = summary.get("deadline_lock")
    candidate_lock_sha = bound_lock.get("sha256") if isinstance(bound_lock, dict) else None
    try:
        producer_slice, consumer_slice = PLACEMENT_VARIANTS[placement_variant]
    except KeyError as error:
        raise ValueError("candidate placement variant is unsupported") from error
    deadline_mode = row.get("deadline_mode", "wall")
    pipeline_mode = pipeline.get("deadline_mode", deadline_mode)
    if pipeline_mode != deadline_mode:
        raise ValueError("pipeline deadline mode differs from summary")
    if deadline_mode == "validation-excluded":
        pipeline_p99 = pipeline.get("stage_latency_us", {}).get(
            "validation_excluded_end_to_end_p99"
        )
    elif deadline_mode == "wall":
        pipeline_p99 = pipeline.get("end_to_end_us", {}).get("p99")
    else:
        raise ValueError("unsupported candidate deadline mode")
    if finite(pipeline_p99, "pipeline p99") != finite(
        row.get("pipeline_p99_us"), "summary p99"
    ):
        raise ValueError("pipeline p99 differs from summary")
    requests = row.get("pipeline_requests")
    misses = row.get("deadline_misses")
    if formal:
        if not isinstance(requests, int) or isinstance(requests, bool) or requests <= 0:
            raise ValueError("formal candidate has invalid request count")
        if not isinstance(misses, int) or isinstance(misses, bool) or not 0 <= misses <= requests:
            raise ValueError("formal candidate has invalid deadline misses")
    candidate_deadline = finite(summary.get("deadline_us", fallback_deadline_us), "candidate deadline")
    slo_qualified = misses == 0 and finite(pipeline_p99, "pipeline p99") <= candidate_deadline
    return {
        "candidate_id": (
            f"q{producer_quota}-q{background_quota}@{placement_variant}"
            if "placement_variant" in row or "placement_variant" in summary
            else f"q{producer_quota}-q{background_quota}"
        ),
        "summary": {
            "path": os.path.relpath(summary_path, output_root),
            "sha256": sha256(summary_path),
        },
        "profile_path": os.path.relpath(pipeline_path, output_root),
        "profile_sha256": sha256(pipeline_path),
        "placement": {
            "producer": f"{producer_slice}-q{producer_quota}",
            "consumer": f"{consumer_slice}-q100",
        },
        # The consumer MIG slice is fixed by placement; the best-effort
        # client quota is an independent search dimension and must not be
        # inferred from the consumer slice.
        "background_quota_percent": background_quota,
        "placement_variant": placement_variant,
        "background_goodput_rps": finite(
            row.get("background_goodput_rps"), "background goodput"
        ),
        "pipeline_requests": requests,
        "deadline_misses": misses,
        "observed_p99_us": finite(pipeline_p99, "pipeline p99"),
        "slo_qualified": slo_qualified,
        "workload": summary.get("workload"),
        "background_period_ms": summary.get("background_period_ms"),
        "background_offered_rps": summary.get("background_offered_rps"),
        "deadline_mode": deadline_mode,
        "deadline_us": candidate_deadline,
        "deadline_lock_sha256": candidate_lock_sha,
        "checksum_mode": summary.get("checksum_mode"),
        "correctness_validated": row.get("correctness_validated") is True,
        "unique_payload_checksums": row.get("unique_payload_checksums"),
        "unique_policy_output_checksums": row.get("unique_policy_output_checksums"),
        "pre_release_guard_p99_us": finite(row.get("gate_p99_us"), "gate p99"),
        "reservation_margin_us": margin_us,
    }


def build_spec(
    summaries: list[Path], output_path: Path, deadline_us: float,
    lookahead_us: float, margin_us: float, deadline_lock_path: Path | None = None,
    formal: bool = False,
) -> dict[str, Any]:
    output_root = output_path.resolve().parent
    deadline_lock = None
    expected_workload = None
    expected_payload_bytes = None
    allowed_placements: set[str] | None = None
    expected_deadline_lock_sha256: str | None = None
    if deadline_lock_path is not None:
        deadline_lock_path = deadline_lock_path.resolve()
        lock_bytes = deadline_lock_path.read_bytes()
        lock = json.loads(lock_bytes)
        contract = lock.get("contract", {})
        expected = {
            "whisper-projection": (2_304_000, "validation-excluded"),
            "resnet-control": (14_720, "wall"),
            "resnet50-classification": (802_816, "wall"),
        }.get(contract.get("workload"))
        kind = lock.get("kind")
        common = kind == "p9-common-placement-deadline-lock"
        if (
            kind not in {"p9-dependent-pipeline-deadline-lock", "p9-common-placement-deadline-lock"}
            or expected is None
            or contract.get("payload_bytes") != expected[0]
            or contract.get("deadline_mode") != expected[1]
        ):
            raise ValueError("invalid dependent-pipeline deadline lock")
        if common:
            raw_allowed = lock.get("allowed_placements", contract.get("allowed_placements"))
            if not isinstance(raw_allowed, list) or not raw_allowed or not all(
                isinstance(item, str) and item for item in raw_allowed
            ):
                raise ValueError("common deadline lock lacks allowed placements")
            allowed_placements = set(raw_allowed)
        expected_workload = contract["workload"]
        expected_payload_bytes = contract["payload_bytes"]
        deadline_us = finite(lock.get("deadline_us"), "locked deadline")
        expected_deadline_lock_sha256 = (
            hashlib.sha256(lock_bytes).hexdigest() if common else None
        )
        deadline_lock = {
            "path": str(deadline_lock_path),
            "sha256": hashlib.sha256(lock_bytes).hexdigest(),
        }
    result = {
        "schema_version": 1,
        "system": "QUIET",
        "deadline_us": finite(deadline_us, "deadline"),
        "critical_lookahead_us": finite(lookahead_us, "lookahead"),
        "candidates": [
            candidate(
                path,
                output_root,
                margin_us,
                deadline_us,
                expected_workload,
                expected_payload_bytes,
                expected_deadline_lock_sha256,
                allowed_placements,
                formal,
            )
            for path in summaries
        ],
    }
    if deadline_lock is not None:
        result["deadline_lock"] = deadline_lock
    ids = [item["candidate_id"] for item in result["candidates"]]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate quota pairs must be unique")
    # A single measured profile is useful for characterization, but it is not
    # evidence that the planner searched placement/slack alternatives. Keep
    # this distinction explicit so a smoke cannot be presented as an
    # optimization result by accident.
    candidate_count = len(result["candidates"])
    placement_variants = {
        item["placement_variant"] for item in result["candidates"]
    }
    placement_search = len(placement_variants) >= 2
    quota_pairs_by_placement: dict[str, set[tuple[int, int]]] = {}
    for item in result["candidates"]:
        producer_quota = int(item["placement"]["producer"].split("-q", 1)[1])
        consumer_quota = item.get("background_quota_percent")
        if consumer_quota not in {10, 25, 50, 75, 90, 100}:
            raise ValueError("candidate lacks an explicit background quota")
        quota_pairs_by_placement.setdefault(item["placement_variant"], set()).add(
            (producer_quota, consumer_quota)
        )
    quota_search = bool(quota_pairs_by_placement) and all(
        len(pairs) >= 2 for pairs in quota_pairs_by_placement.values()
    )
    if formal:
        if not placement_search or not quota_search:
            raise ValueError(
                "formal candidate search requires >=2 placements and >=2 quota pairs per placement"
            )
        contracts = {
            (
                item.get("workload"), item.get("background_period_ms"),
                item.get("background_offered_rps"), item.get("pipeline_requests"),
                item.get("deadline_mode"), item.get("deadline_us"),
                item.get("deadline_lock_sha256"), item.get("checksum_mode"),
                item.get("correctness_validated"), item.get("unique_payload_checksums"),
                item.get("unique_policy_output_checksums"),
            ) for item in result["candidates"]
        }
        if len(contracts) != 1:
            raise ValueError("formal candidates do not share request/correctness contract")
        if next(iter(contracts))[6] is None:
            raise ValueError("formal candidates require a shared deadline lock")
    result["candidate_search"] = {
        "candidate_count": candidate_count,
        "multi_candidate_evaluated": candidate_count >= 2,
        "placement_variant_count": len(placement_variants),
        "placement_search_evaluated": placement_search,
        "quota_search_evaluated": quota_search,
        "formal_contract_requested": formal,
        "formal_eligible": bool(formal and placement_search and quota_search),
        "claim_status": (
            "multi-candidate-placement-and-quota-search"
            if candidate_count >= 2 and placement_search and quota_search
            else "multi-candidate-placement-characterization"
            if candidate_count >= 2 and placement_search
            else "multi-candidate-quota-search-only"
            if candidate_count >= 2
            else "single-candidate-characterization"
        ),
    }
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-us", type=float, default=760.0)
    parser.add_argument("--deadline-lock", type=Path)
    parser.add_argument("--critical-lookahead-us", type=float, default=1000.0)
    parser.add_argument("--reservation-margin-us", type=float, default=0.0)
    parser.add_argument(
        "--formal", action="store_true",
        help="require the frozen placement-by-quota formal search contract",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    spec = build_spec(
        args.summary,
        args.output,
        args.deadline_us,
        args.critical_lookahead_us,
        args.reservation_margin_us,
        args.deadline_lock,
        args.formal,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(spec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
