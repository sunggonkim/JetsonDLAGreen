#!/usr/bin/env python3
"""Aggregate six current production-wall ImageNette formal sessions.

This analyzer is intentionally separate from the exploratory three-sequence
frontier summarizer.  It requires six independent launcher sessions, each
with 1,100 measured requests and a current common-workload contract, then
computes request-level DMR confidence and tail statistics from the raw CSVs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import beta, t as student_t

try:
    from analysis.summarize_p9_active_williams_repeats import (
        _read,
        _source_row,
        active_williams_orders,
        sha256,
    )
except ModuleNotFoundError:  # direct execution from analysis/
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.summarize_p9_active_williams_repeats import (  # type: ignore
        _read,
        _source_row,
        active_williams_orders,
        sha256,
    )


SYSTEMS = ("NVIDIA MPS", "XSched", "QUIET")
REQUESTS_PER_SESSION = 1100
SESSION_COUNT = 6
PRODUCTION_WALL_DEFINITION = (
    "arrival-to-consumer-completion-excludes-correctness-validation"
)


def _cp95(misses: int, requests: int) -> float:
    if misses == requests:
        return 1.0
    return float(beta.ppf(0.95, misses + 1, requests - misses))


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _nearest_rank(values: list[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("invalid percentile input")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _tail(values: list[float]) -> dict[str, float]:
    return {
        "p50_us": _nearest_rank(values, 0.50),
        "p99_us": _nearest_rank(values, 0.99),
        "p99_9_us": _nearest_rank(values, 0.999),
        "mean_us": statistics.fmean(values),
    }


def _pipeline_path(system: str, evidence: Path) -> Path:
    value = json.loads(evidence.read_bytes())
    if system in {"NVIDIA MPS", "QUIET"}:
        rows = value.get("results")
        if not isinstance(rows, list) or len(rows) != 1:
            raise ValueError(f"{system} raw result row is missing")
        record = rows[0].get("request_trace")
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError(f"{system} pipeline trace provenance is missing")
        return Path(record["path"]).resolve()
    if system == "XSched":
        path = evidence.resolve().parent / "pipeline.csv"
        if not path.is_file():
            raise ValueError("XSched pipeline trace is missing")
        return path
    raise ValueError(f"unsupported formal system: {system}")


def _raw_latencies(system: str, evidence: Path, expected_requests: int) -> tuple[list[float], str]:
    path = _pipeline_path(system, evidence)
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"request", "input_sha256", "wall_end_to_end_us", "deadline_miss"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{system} raw production-wall CSV schema differs")
        expected_request = 10
        for row in reader:
            request = int(row["request"])
            latency = _finite(float(row["wall_end_to_end_us"]), f"{system} latency")
            if request != expected_request or latency <= 0.0:
                raise ValueError(f"{system} raw production-wall requests are not dense")
            expected_request += 1
            observed_miss = row["deadline_miss"] in {"1", "true", "True"}
            values.append(latency)
        if len(values) != expected_requests:
            raise ValueError(f"{system} raw production-wall request count differs")
    return values, sha256(path)


def _paired_t(values: list[float]) -> dict[str, Any]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"n": len(values), "mean": mean, "lower": None, "upper": None}
    sd = statistics.stdev(values)
    half = float(student_t.ppf(0.975, len(values) - 1)) * sd / math.sqrt(len(values))
    return {
        "method": "paired-session-t-interval",
        "confidence": 0.95,
        "n": len(values),
        "mean": mean,
        "sample_sd": sd,
        "lower": mean - half,
        "upper": mean + half,
    }


def summarize(paths: Iterable[Path]) -> dict[str, Any]:
    inputs = [Path(path).resolve() for path in paths]
    if len(inputs) != SESSION_COUNT:
        raise ValueError("current ImageNette formal aggregate requires exactly six sessions")
    expected_orders = active_williams_orders()
    seen_paths: set[str] = set()
    source_hashes: set[str] = set()
    contract: tuple[Any, ...] | None = None
    session_rows: list[dict[str, Any]] = []
    all_latencies: dict[str, list[float]] = {system: [] for system in SYSTEMS}
    all_misses: dict[str, int] = {system: 0 for system in SYSTEMS}
    all_goodput: dict[str, list[float]] = {system: [] for system in SYSTEMS}
    p99_by_system: dict[str, list[float]] = {system: [] for system in SYSTEMS}

    for session_index, path in enumerate(inputs):
        run, run_sha = _read(path)
        if (
            run.get("kind") != "p9-common-sota-williams-sequence"
            or run.get("active_only") is not True
            or run.get("workload") != "resnet50-classification"
            or run.get("placement_variant") != "fixed-1g-producer-2g-consumer"
            or run.get("deadline_mode") != "wall"
            or run.get("execution_order") != list(expected_orders[session_index % 3])
            or run.get("requests_per_system") != REQUESTS_PER_SESSION
        ):
            raise ValueError(f"formal session {session_index} contract differs")
        lock = run.get("deadline_lock")
        plan = run.get("quiet_plan")
        common = run.get("common_workload")
        if not isinstance(lock, dict) or not isinstance(plan, dict) or not isinstance(common, dict):
            raise ValueError(f"formal session {session_index} lacks provenance")
        lock_path = Path(lock.get("path", "")).resolve()
        plan_path = Path(plan.get("path", "")).resolve()
        if (
            not lock_path.is_file() or not plan_path.is_file()
            or sha256(lock_path) != lock.get("sha256")
            or sha256(plan_path) != plan.get("sha256")
        ):
            raise ValueError(f"formal session {session_index} lock/plan SHA differs")
        current_contract = (
            lock.get("sha256"), plan.get("sha256"), common.get("request_count"),
            common.get("arrival_trace_sha256"), common.get("producer_input_trace_sha256"),
            common.get("operational_arrival_trace_sha256"),
        )
        if contract is None:
            contract = current_contract
        elif current_contract != contract:
            raise ValueError(f"formal session {session_index} common contract differs")
        results = run.get("results")
        evidence_records = run.get("inputs")
        if (
            not isinstance(results, list) or not isinstance(evidence_records, list)
            or tuple(row.get("system") for row in results) != tuple(run["execution_order"])
            or {row.get("system") for row in results} != set(SYSTEMS)
        ):
            raise ValueError(f"formal session {session_index} system rows differ")
        by_system: dict[str, dict[str, Any]] = {}
        for row, record in zip(results, evidence_records, strict=True):
            system = row["system"]
            evidence = Path(record.get("path", "")).resolve()
            if not evidence.is_file() or sha256(evidence) != record.get("sha256"):
                raise ValueError(f"formal session {session_index} {system} evidence SHA differs")
            if sha256(evidence) in source_hashes:
                raise ValueError(f"formal session {session_index} reuses result evidence")
            source_hashes.add(sha256(evidence))
            normalized = _source_row(system, evidence, record["sha256"], lock["sha256"], run["workload"])
            accuracy = normalized.get("application_accuracy")
            if (
                not isinstance(accuracy, dict)
                or accuracy.get("candidate_accuracy", -1.0) < 0.8
                or accuracy.get("accuracy_delta", 1.0) != 0.0
            ):
                raise ValueError(f"formal session {session_index} {system} accuracy gate differs")
            latencies, trace_sha = _raw_latencies(system, evidence, REQUESTS_PER_SESSION)
            normalized["raw_pipeline_sha256"] = trace_sha
            normalized["tail"] = _tail(latencies)
            by_system[system] = normalized
            all_latencies[system].extend(latencies)
            all_misses[system] += normalized["misses"]
            all_goodput[system].append(normalized["background_goodput_rps"])
            p99_by_system[system].append(normalized["tail"]["p99_us"])
        session_rows.append({
            "session_index": session_index,
            "path": str(path),
            "sha256": run_sha,
            "execution_order": run["execution_order"],
            "systems": by_system,
        })

    if contract is None:
        raise ValueError("formal session contract is empty")
    systems: dict[str, Any] = {}
    total_requests = SESSION_COUNT * REQUESTS_PER_SESSION
    for system in SYSTEMS:
        systems[system] = {
            "requests": total_requests,
            "sessions": SESSION_COUNT,
            "misses": all_misses[system],
            "observed_dmr": all_misses[system] / total_requests,
            "dmr_cp95_upper": _cp95(all_misses[system], total_requests),
            "slo_confidence_qualified": (
                all_misses[system] == 0
                and _cp95(all_misses[system], total_requests) <= 0.0005
            ),
            "tail": _tail(all_latencies[system]),
            "session_p99_us": p99_by_system[system],
            "background_goodput_rps": {
                "mean": statistics.fmean(all_goodput[system]),
                "min": min(all_goodput[system]),
                "max": max(all_goodput[system]),
            },
        }
    paired = {
        "QUIET_minus_NVIDIA_MPS_p99_us": _paired_t([
            quiet - mps
            for quiet, mps in zip(p99_by_system["QUIET"], p99_by_system["NVIDIA MPS"], strict=True)
        ]),
        "QUIET_minus_NVIDIA_MPS_goodput_rps": _paired_t([
            quiet - mps
            for quiet, mps in zip(all_goodput["QUIET"], all_goodput["NVIDIA MPS"], strict=True)
        ]),
    }
    return {
        "schema_version": 1,
        "kind": "p9-current-imagenette-formal-production-wall-aggregate",
        "proposed_system": "QUIET",
        "workload": "resnet50-classification",
        "scope": "current-production-wall-labelled-imagenette-formal-six-session-replay",
        "formal": True,
        "ranking_allowed": False,
        "thermal_normalized": False,
        "thermal_claim_allowed": False,
        "production_wall_definition": PRODUCTION_WALL_DEFINITION,
        "deadline_mode": "wall",
        "sessions": SESSION_COUNT,
        "requests_per_session": REQUESTS_PER_SESSION,
        "requests_per_system": total_requests,
        "dmr_target": 0.0005,
        "systems": systems,
        "paired_session_statistics": paired,
        "sessions_input": session_rows,
        "deadline_lock_sha256": contract[0],
        "quiet_plan_sha256": contract[1],
        "common_workload": {
            "request_count": contract[2],
            "arrival_trace_sha256": contract[3],
            "producer_input_trace_sha256": contract[4],
            "operational_arrival_trace_sha256": contract[5],
        },
        "statistical_unit": "request-level DMR with paired independent session tails",
        "claim_guard": (
            "Current binary/source/engine and labelled output gates are bound; "
            "thermal normalization and same-SLO ranking remain separate gates."
        ),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = summarize(args.input)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
