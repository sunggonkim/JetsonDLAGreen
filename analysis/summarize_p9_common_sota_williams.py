#!/usr/bin/env python3
"""Verify and aggregate the six-sequence common-workload SOTA pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_p9_common_sota_williams import (  # noqa: E402
    COMPARISON_CONTRACTS,
    SYSTEMS,
    williams_orders,
)


DMR_TARGET = 0.0005
TRACE_COLUMNS = (
    "request", "producer_compute_us", "producer_copy_us",
    "producer_validation_us", "notification_us", "consumer_validation_us",
    "consumer_copy_us", "edge_transport_us", "consumer_compute_us",
    "output_verification_us", "validation_excluded_end_to_end_us",
    "wall_end_to_end_us", "deadline_miss",
)
TRACE_COLUMNS_WITH_INPUT = (
    "request", "producer_compute_us", "producer_copy_us", "input_sha256",
    "producer_validation_us", "notification_us", "consumer_validation_us",
    "consumer_copy_us", "edge_transport_us", "consumer_compute_us",
    "output_verification_us", "validation_excluded_end_to_end_us",
    "wall_end_to_end_us", "deadline_miss",
)
STAGE_COLUMNS = (
    "producer_compute_us", "notification_us", "edge_transport_us",
    "consumer_compute_us", "output_verification_us",
)
COMMON_WORKLOAD_KEYS = (
    "schema_version", "workload_id", "topology", "placement", "input_tensor",
    "payload_bytes", "arrival_trace_path", "arrival_trace_sha256",
    "dataset_manifest_path", "dataset_manifest_sha256", "contract_path",
    "contract_sha256",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_comparator_manifest() -> tuple[dict[str, Any], str]:
    """Read the publication-facing comparator contract from one byte buffer."""
    path = ROOT / "docs" / "p9-comparator-manifest.json"
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("comparator manifest is not newline-complete")
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("proposed_system") != "QUIET":
        raise ValueError("comparator manifest is malformed")
    if not isinstance(value.get("rows"), dict) or not isinstance(
        value.get("structural_controls"), list
    ):
        raise ValueError("comparator manifest rows are malformed")
    return value, hashlib.sha256(raw).hexdigest()


def manifest_contract(
    manifest: dict[str, Any], system: str
) -> tuple[bool, str]:
    rows = manifest["rows"]
    entry = rows.get(system)
    if isinstance(entry, dict):
        allowed = entry.get("numeric_comparison_allowed")
        status = entry.get("status")
        if not isinstance(allowed, bool) or not isinstance(status, str) or not status:
            raise ValueError(f"manifest contract is malformed for {system}")
        return allowed, status
    if system in manifest["structural_controls"]:
        return False, "structural-only"
    raise ValueError(f"system {system!r} is absent from comparator manifest")


def cp95_upper(misses: int, requests: int) -> float:
    if requests <= 0 or not 0 <= misses <= requests:
        raise ValueError("invalid miss count")
    return 1.0 if misses == requests else float(beta.ppf(0.95, misses + 1, requests - misses))


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return number


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile input is empty")
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def resolve_path(value: str, owner: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for candidate in (ROOT / path, owner.parent / path):
        if candidate.is_file():
            return candidate.resolve()
    return (ROOT / path).resolve()


def validate_common_workload(value: Any, owner: Path) -> dict[str, Any]:
    if not isinstance(value, dict) or any(key not in value for key in COMMON_WORKLOAD_KEYS):
        raise ValueError("common workload contract is malformed")
    if value.get("schema_version") != 1:
        raise ValueError("common workload contract schema differs")
    for path_key, digest_key in (
        ("arrival_trace_path", "arrival_trace_sha256"),
        ("dataset_manifest_path", "dataset_manifest_sha256"),
        ("contract_path", "contract_sha256"),
    ):
        path = resolve_path(value[path_key], owner)
        digest = value[digest_key]
        if (
            not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not path.is_file() or sha256(path) != digest
        ):
            raise ValueError(f"common workload evidence SHA differs: {path_key}")
    return dict(value)


def replay_trace(
    path: Path, expected_hash: str, deadline_us: float,
    latency_field: str = "wall_end_to_end_us",
    expected_inputs: dict[int, str] | None = None,
) -> dict[str, Any]:
    if sha256(path) != expected_hash:
        raise ValueError(f"raw trace hash differs: {path}")
    latencies: list[float] = []
    stages: dict[str, list[float]] = {name: [] for name in STAGE_COLUMNS}
    misses = 0
    input_binding = expected_inputs is None
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        columns = tuple(reader.fieldnames or ())
        if columns not in (TRACE_COLUMNS, TRACE_COLUMNS_WITH_INPUT):
            raise ValueError("raw trace schema differs")
        has_input = columns == TRACE_COLUMNS_WITH_INPUT
        if expected_inputs is not None and not has_input:
            input_binding = False
        elif expected_inputs is not None:
            input_binding = True
        for row in reader:
            request = int(row["request"])
            latency = finite(float(row[latency_field]), "deadline latency")
            recorded = int(row["deadline_miss"])
            if latency <= 0.0 or recorded not in (0, 1) or recorded != int(latency > deadline_us):
                raise ValueError("raw deadline classification differs")
            latencies.append(latency)
            if expected_inputs is not None and has_input:
                digest = row.get("input_sha256", "")
                if (
                    not isinstance(digest, str) or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                    or expected_inputs.get(request) != digest
                ):
                    raise ValueError("raw input binding differs")
            for name in STAGE_COLUMNS:
                stages[name].append(finite(float(row[name]), name))
            misses += recorded
    return {
        "requests": len(latencies), "misses": misses,
        "latencies": latencies, "stages": stages,
        "input_binding": input_binding,
    }


def arrival_inputs(contract: dict[str, Any], owner: Path) -> dict[int, str]:
    path = resolve_path(contract["arrival_trace_path"], owner)
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("common arrival trace is not newline-complete")
    result: dict[int, str] = {}
    arrivals: set[int] = set()
    for line_number, line in enumerate(raw.splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict) or set(value) != {
            "schema_version", "iteration", "request_id", "arrival_sequence",
            "input_sha256", "expected_label",
        } or value.get("schema_version") != 1:
            raise ValueError(f"common arrival trace:{line_number} schema differs")
        iteration = value.get("iteration")
        arrival = value.get("arrival_sequence")
        digest = value.get("input_sha256")
        if (
            isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0
            or iteration in result
            or isinstance(arrival, bool) or not isinstance(arrival, int)
            or arrival < 0 or arrival in arrivals
            or not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"common arrival trace:{line_number} input binding is invalid")
        result[iteration] = digest
        arrivals.add(arrival)
    if not result or sorted(arrivals) != list(range(len(result))):
        raise ValueError("common arrival trace is not dense")
    return result


def trace_for_system(system: str, evidence: Path) -> tuple[Path, str]:
    value = json.loads(evidence.read_text(encoding="utf-8"))
    if system in ("NVIDIA MIG", "NVIDIA MPS", "QUIET"):
        rows = value.get("results")
        if not isinstance(rows, list) or len(rows) != 1 or rows[0].get("system") != system:
            raise ValueError(f"{system} evidence row differs")
        trace = rows[0].get("request_trace")
        if not isinstance(trace, dict):
            raise ValueError(f"{system} trace provenance is missing")
        return resolve_path(trace["path"], evidence), trace["sha256"]
    if system in ("Orion", "XSched"):
        inputs = value.get("inputs_sha256", value.get("inputs"))
        digest = (
            inputs.get("pipeline", inputs.get("pipeline_sha256"))
            if isinstance(inputs, dict) else None
        )
        if not isinstance(digest, str):
            raise ValueError(f"{system} pipeline provenance is missing")
        return evidence.parent / "pipeline.csv", digest
    if system == "gpulet":
        result_path = evidence.parent / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        evaluation = Path(result["evaluation"]["path"])
        trace = evaluation.parent / "gpulet/pipeline.csv"
        digest = value.get("trace_sha256")
        if not isinstance(digest, str):
            raise ValueError("gpulet trace provenance is missing")
        return trace, digest
    raise ValueError(f"unknown public system: {system}")


def summarize(paths: list[Path], expected_offered_rps: float = 250.0) -> dict[str, Any]:
    if len(paths) != 6:
        raise ValueError("Williams aggregate requires exactly six sequences")
    expected_offered_rps = finite(expected_offered_rps, "expected offered rps")
    if expected_offered_rps <= 0.0:
        raise ValueError("expected offered rps must be positive")
    comparator_manifest, comparator_manifest_sha256 = load_comparator_manifest()
    orders = williams_orders()
    seen: set[int] = set()
    lock: dict[str, str] | None = None
    plan: dict[str, str] | None = None
    per_system: dict[str, list[dict[str, Any]]] = {name: [] for name in SYSTEMS}
    inputs: list[dict[str, Any]] = []
    evidence_hashes: set[str] = set()
    requests_per_system: int | None = None
    gpulet_profile_requests: int | None = None
    deadline_us: float | None = None
    workload: str | None = None
    common_workload: dict[str, Any] | None = None
    common_inputs: dict[int, str] | None = None
    for path in paths:
        run = json.loads(path.read_text(encoding="utf-8"))
        index = run.get("sequence_index")
        if (
            run.get("kind") != "p9-common-sota-williams-sequence"
            or run.get("proposed_system") != "QUIET"
            or isinstance(index, bool) or not isinstance(index, int)
            or not 0 <= index < 6 or index in seen
            or tuple(run.get("execution_order", ())) != orders[index]
            or isinstance(run.get("requests_per_system"), bool)
            or not isinstance(run.get("requests_per_system"), int)
            or run.get("requests_per_system") <= 0
            or isinstance(run.get("gpulet_profile_requests_per_partition", 100), bool)
            or not isinstance(run.get("gpulet_profile_requests_per_partition", 100), int)
            or run.get("gpulet_profile_requests_per_partition", 100) <= 0
            or finite(run.get("background_offered_rps"), "offered rps")
            != expected_offered_rps
        ):
            raise ValueError("Williams sequence contract differs")
        seen.add(index)
        current_workload = run.get("workload", "resnet-control")
        if current_workload not in ("resnet-control", "whisper-projection"):
            raise ValueError("Williams workload contract differs")
        if workload is None:
            workload = current_workload
        elif workload != current_workload:
            raise ValueError("Williams workloads differ")
        current_common = run.get("common_workload")
        if current_common is None:
            if common_workload is not None:
                raise ValueError("Williams common workload evidence is incomplete")
        else:
            checked_common = validate_common_workload(current_common, path)
            if common_workload is None:
                common_workload = checked_common
                common_inputs = arrival_inputs(checked_common, path)
            elif checked_common != common_workload:
                raise ValueError("Williams common workload contract differs")
        current_requests = run["requests_per_system"]
        current_profile_requests = run.get("gpulet_profile_requests_per_partition", 100)
        if requests_per_system is None:
            requests_per_system = current_requests
            gpulet_profile_requests = current_profile_requests
        elif (
            current_requests != requests_per_system
            or current_profile_requests != gpulet_profile_requests
        ):
            raise ValueError("Williams request counts differ")
        if lock is None:
            lock, plan = run.get("deadline_lock"), run.get("quiet_plan")
            lock_path = Path(lock["path"])
            lock_value = json.loads(lock_path.read_text(encoding="utf-8"))
            deadline_us = finite(lock_value.get("deadline_us"), "deadline")
            lock_workload = lock_value.get("contract", {}).get("workload")
            if lock_workload is not None and lock_workload != workload:
                raise ValueError("deadline lock workload differs")
        elif run.get("deadline_lock") != lock or run.get("quiet_plan") != plan:
            raise ValueError("Williams lock or plan differs")
        results, evidence = run.get("results"), run.get("inputs")
        if (
            not isinstance(results, list) or not isinstance(evidence, list)
            or tuple(row.get("system") for row in results) != orders[index]
            or tuple(item.get("system") for item in evidence) != orders[index]
        ):
            raise ValueError("Williams rows differ from execution order")
        checked = []
        for row, item in zip(results, evidence, strict=True):
            source = Path(item["path"])
            digest = sha256(source)
            if digest != item["sha256"] or digest in evidence_hashes:
                raise ValueError("system evidence differs or is reused")
            evidence_hashes.add(digest)
            requests, misses = row.get("requests"), row.get("misses")
            if (
                isinstance(requests, bool) or not isinstance(requests, int)
                or requests != requests_per_system
                or isinstance(misses, bool) or not isinstance(misses, int)
                or not 0 <= misses <= requests
            ):
                raise ValueError("request totals differ")
            if row["system"] == "gpulet" and row.get("spatial_schedule_feasible") is not False:
                raise ValueError("gpulet diagnostic row is not planner-infeasible")
            manifest_allowed, manifest_status = manifest_contract(
                comparator_manifest, row["system"]
            )
            declared_allowed = row.get(
                "numeric_comparison_allowed",
                COMPARISON_CONTRACTS[row["system"]]["numeric_comparison_allowed"],
            )
            if not isinstance(declared_allowed, bool):
                raise ValueError("row numeric comparison contract is malformed")
            # Aggregate JSON cannot promote a structural or fidelity-pending
            # comparator.  The repository manifest is the publication authority.
            # A replayed CSV alone cannot prove that every arm consumed the
            # same external inputs and arrival sequence.  Old campaigns may
            # remain useful raw evidence, but only a byte-bound common
            # workload can enter a publication-facing numeric table.
            effective_allowed = (
                declared_allowed and manifest_allowed and common_workload is not None
            )
            if row.get("correctness_validated") is False:
                raise ValueError("formal Williams evidence lacks correctness validation")
            per_system[row["system"]].append({
                "requests": requests,
                "misses": misses,
                "p99_us": finite(row.get("p99_us"), "p99"),
                "goodput_rps": finite(row.get("background_goodput_rps"), "goodput"),
                "trace": None,
                "deadline_mode": row.get("deadline_mode", run.get("deadline_mode", "legacy")),
                "numeric_comparison_allowed": effective_allowed,
                "comparison_status": manifest_status,
                "topology": row.get(
                    "topology", COMPARISON_CONTRACTS[row["system"]]["topology"]
                ),
            })
            trace_path, trace_hash = trace_for_system(row["system"], source)
            deadline_mode = per_system[row["system"]][-1]["deadline_mode"]
            if deadline_mode not in {"wall", "validation-excluded", "legacy"}:
                raise ValueError("unknown deadline mode in Williams evidence")
            latency_field = (
                "validation_excluded_end_to_end_us"
                if deadline_mode == "validation-excluded"
                or (deadline_mode == "legacy" and workload == "whisper-projection")
                else "wall_end_to_end_us"
            )
            replay = replay_trace(
                trace_path, trace_hash, float(deadline_us), latency_field,
                expected_inputs=common_inputs,
            )
            if (
                replay["requests"] != requests or replay["misses"] != misses
                or not math.isclose(
                    percentile(replay["latencies"], 0.99),
                    per_system[row["system"]][-1]["p99_us"], abs_tol=0.01,
                )
            ):
                raise ValueError(f"{row['system']} raw replay differs")
            per_system[row["system"]][-1]["latencies"] = replay["latencies"]
            per_system[row["system"]][-1]["stages"] = replay["stages"]
            per_system[row["system"]][-1]["input_binding"] = replay["input_binding"]
            if not replay["input_binding"]:
                per_system[row["system"]][-1]["numeric_comparison_allowed"] = False
                per_system[row["system"]][-1]["comparison_status"] = (
                    "input-binding-pending-production-trace"
                )
            per_system[row["system"]][-1]["trace"] = {
                "path": str(trace_path), "sha256": trace_hash,
            }
            checked.append({"system": row["system"], "path": str(source), "sha256": digest})
        inputs.append({"sequence": index, "path": str(path), "sha256": sha256(path), "evidence": checked})
    if seen != set(range(6)) or not isinstance(lock, dict) or not isinstance(plan, dict):
        raise ValueError("Williams design is incomplete")
    for item, label in ((lock, "deadline lock"), (plan, "QUIET plan")):
        source = Path(item["path"])
        if sha256(source) != item["sha256"]:
            raise ValueError(f"{label} hash differs")
    systems_out: dict[str, Any] = {}
    for name in SYSTEMS:
        samples = per_system[name]
        requests = sum(item["requests"] for item in samples)
        misses = sum(item["misses"] for item in samples)
        p99s = [item["p99_us"] for item in samples]
        goodputs = [item["goodput_rps"] for item in samples]
        latencies = [latency for item in samples for latency in item["latencies"]]
        stage_values = {
            stage: [value for item in samples for value in item["stages"][stage]]
            for stage in STAGE_COLUMNS
        }
        upper = cp95_upper(misses, requests)
        systems_out[name] = {
            "runs": 6,
            "requests": requests,
            "misses": misses,
            "observed_dmr": misses / requests,
            "dmr_target": DMR_TARGET,
            "dmr_cp95_upper": upper,
            "slo_confidence_qualified": upper <= DMR_TARGET,
            "per_run_p99_us": p99s,
            "p99_us_range": [min(p99s), max(p99s)],
            "pooled_p99_us": percentile(latencies, 0.99),
            "pooled_p999_us": percentile(latencies, 0.999),
            "maximum_us": max(latencies),
            "pooled_stage_p99_us": {
                stage: percentile(values, 0.99)
                for stage, values in stage_values.items()
            },
            "background_goodput_rps_mean": statistics.fmean(goodputs),
            "background_goodput_rps_range": [min(goodputs), max(goodputs)],
            "numeric_comparison_allowed": all(
                item["numeric_comparison_allowed"] for item in samples
            ),
            "input_binding": all(item.get("input_binding", False) for item in samples),
            "comparison_status": samples[0]["comparison_status"],
            "topology": samples[0]["topology"],
            "trace_inputs": [item["trace"] for item in samples],
        }
    qualified = [name for name, value in systems_out.items() if value["slo_confidence_qualified"]]
    # Keep the six-treatment raw replay unchanged, but expose the canonical
    # paper view separately.  Pantheon is intentionally a functional-only row
    # until an accuracy-equivalent numeric adapter is replay-verified.
    headline_systems: dict[str, Any] = {
        name: systems_out[name]
        for name in ("NVIDIA MIG", "NVIDIA MPS", "Orion", "XSched", "QUIET")
    }
    headline_systems["Pantheon"] = {
        "runs": 0,
        "numeric_comparison_allowed": False,
        "slo_confidence_qualified": False,
        "comparison_status": "functional-only-pending-common-workload-adapter",
        "topology": COMPARISON_CONTRACTS["QUIET"]["topology"],
    }
    return {
        "schema_version": 1,
        "kind": "p9-common-sota-williams-aggregate",
        "scope": "order-balanced-raw-replayed-nonthermal-campaign",
        "proposed_system": "QUIET",
        "order_design": "six-treatment-williams",
        "comparison_contract_version": 1,
        "comparator_manifest": {
            "path": str((ROOT / "docs" / "p9-comparator-manifest.json").resolve()),
            "sha256": comparator_manifest_sha256,
        },
        "deadline_lock": lock,
        "quiet_plan": plan,
        "workload": workload,
        "common_workload": common_workload,
        "background_offered_rps": expected_offered_rps,
        "deadline_mode": (
            "wall"
            if all(
                sample["deadline_mode"] == "wall"
                for samples in per_system.values()
                for sample in samples
            )
            else "legacy"
        ),
        "requests_per_system_per_sequence": requests_per_system,
        "gpulet_profile_requests_per_partition": gpulet_profile_requests,
        "inputs": sorted(inputs, key=lambda item: item["sequence"]),
        "systems": systems_out,
        "headline_system_order": [
            "NVIDIA MIG", "NVIDIA MPS", "Orion", "XSched", "Pantheon", "QUIET"
        ],
        "headline_systems": headline_systems,
        "confidence_qualified_systems": qualified,
        "numeric_comparison_systems": [
            name for name, value in systems_out.items()
            if value["numeric_comparison_allowed"]
        ],
        "claim_guard": (
            "Statistical DMR qualification is sample-size valid; thermal normalization remains pending."
            if qualified and common_workload is not None
            else "Common workload contract is missing or no row satisfies the predeclared CP95 DMR target."
        ),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offered-rps", type=float, default=250.0)
    args = parser.parse_args(argv)
    result = summarize(
        [path.resolve() for path in args.input], args.offered_rps
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
