#!/usr/bin/env python3
"""Build replay-verified structural-limit evidence for the QUIET P9 design."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


BOER_COMMIT = "df54815de3b1c9059f873a17c13f7d5203eedd3e"
PARVA_COMMIT = "5f3de1e18582b4c81896a1c3eb0e2915238dfee6"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACE_COLUMNS = (
    "request", "producer_compute_us", "producer_copy_us",
    "producer_validation_us", "notification_us", "consumer_validation_us",
    "consumer_copy_us", "edge_transport_us", "consumer_compute_us",
    "output_verification_us", "validation_excluded_end_to_end_us",
    "wall_end_to_end_us", "deadline_miss",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def resolve_evidence_path(value: str, owner: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [PROJECT_ROOT / path]
    if owner is not None:
        candidates.append(owner.parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty latency evidence")
    position = q * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def replay_boer_hardware(search: dict[str, Any]) -> list[dict[str, Any]]:
    deadline_us = float(search["contract"]["deadline_us"])
    replayed: list[dict[str, Any]] = []
    for observation in search.get("observations", []):
        if observation.get("source") != "hardware":
            continue
        metrics = observation.get("metrics")
        evidence = metrics.get("evidence") if isinstance(metrics, dict) else None
        if not isinstance(evidence, dict):
            raise ValueError("BOER hardware observation lacks evidence")
        root = Path(evidence["result_dir"])
        for name, digest in evidence["sha256"].items():
            if sha256(root / name) != digest:
                raise ValueError("BOER hardware evidence hash differs")
        pipeline = load(root / "pipeline.json")
        if (
            pipeline.get("pipeline") != "resnet10-layer7-cov-to-control-mlp"
            or pipeline.get("payload_bytes") != 14_720
            or pipeline.get("checksum_failures") != 0
            or pipeline.get("unique_payload_checksums", 0) < 2
            or pipeline.get("unique_policy_output_checksums", 0) < 2
            or not math.isclose(float(pipeline.get("deadline_us")), deadline_us)
        ):
            raise ValueError("BOER dependent pipeline contract differs")
        latencies: list[float] = []
        misses = 0
        with (root / "pipeline.csv").open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != TRACE_COLUMNS:
                raise ValueError("BOER pipeline trace schema differs")
            for row in reader:
                latency = float(row["wall_end_to_end_us"])
                miss = int(row["deadline_miss"])
                if not math.isfinite(latency) or latency <= 0 or miss != (latency > deadline_us):
                    raise ValueError("BOER raw deadline classification differs")
                latencies.append(latency)
                misses += miss
        p99 = percentile(latencies, 0.99)
        if (
            not math.isclose(p99 / 1000.0, float(metrics["worst_p99_ms"]), abs_tol=1e-9)
            or not math.isclose(misses / len(latencies), float(metrics["deadline_miss_rate"]), abs_tol=1e-12)
            or observation.get("feasible") != (p99 <= deadline_us)
        ):
            raise ValueError("BOER search metrics differ from raw trace")
        replayed.append({
            "candidate": observation["id"],
            "requests": len(latencies),
            "misses": misses,
            "p99_us": p99,
            "served_pipeline_rps": metrics["served_rps_0"],
            "served_background_rps": metrics["served_rps_1"],
        })
    if not replayed:
        raise ValueError("BOER search has no hardware observations")
    return replayed


def quiet_row(summary: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in summary.get("results", []) if row.get("system") == "QUIET"]
    if len(rows) != 1:
        raise ValueError("summary must contain one QUIET row")
    return rows[0]


def summarize(paths: dict[str, Path]) -> dict[str, Any]:
    boer_independent = load(paths["boer_independent"])
    boer_dependent = load(paths["boer_dependent"])
    if (
        boer_independent.get("status") != "selected"
        or boer_independent.get("provenance", {}).get("upstream_commit") != BOER_COMMIT
        or boer_dependent.get("status") != "no-feasible-configuration"
        or boer_dependent.get("provenance", {}).get("upstream_commit") != BOER_COMMIT
        or boer_dependent.get("contract", {}).get("payload_bytes") != 14_720
    ):
        raise ValueError("BOER positive/dependent contract differs")
    boer_replay = replay_boer_hardware(boer_dependent)

    parva_independent = load(paths["parva_independent"])
    parva_dependent = load(paths["parva_dependent"])
    if (
        parva_independent.get("system") != "ParvaGPU"
        or parva_independent.get("all_slos_met") is not True
        or parva_dependent.get("provenance", {}).get("upstream_commit") != PARVA_COMMIT
        or parva_dependent.get("feasible") is not False
        or parva_dependent.get("reason") != "insufficient fixed MIG segments"
        or parva_dependent.get("contract", {}).get("payload_bytes") != 14_720
    ):
        raise ValueError("ParvaGPU positive/dependent contract differs")
    parva_profile = parva_dependent["provenance"]["thor_profile"]
    if (
        sha256(Path(parva_profile["profile_path"])) != parva_profile["profile_sha256"]
        or any(len(value) != 64 for value in parva_profile["input_sha256"].values())
    ):
        raise ValueError("ParvaGPU profile provenance differs")

    plan = load(paths["quiet_plan"])
    selected = plan.get("selected_plan")
    if (
        plan.get("proposed_system") != "QUIET"
        or plan.get("status") != "selected"
        or not isinstance(selected, dict)
        or selected.get("dag", {}).get("edges", [{}])[0].get("payload_bytes") != 14_720
        or selected.get("dag", {}).get("edges", [{}])[0].get("transport")
        != "registered-shared-sysmem-direct-binding"
        or selected.get("reserved_slack_us", -1) <= 0
    ):
        raise ValueError("QUIET ResNet stage-DAG plan differs")
    common_deadline = float(plan["deadline_us"])
    lock_sha = plan["deadline_lock"]["sha256"]

    base = load(paths["plan_enforced_base"])
    if (
        base.get("deadline_source") != "frozen-independent-pipeline-p99-factor"
        or not math.isclose(float(base.get("deadline_us")), common_deadline)
        or base.get("quiet_plan")
        != {"path": str(paths["quiet_plan"].resolve()), "sha256": sha256(paths["quiet_plan"])}
    ):
        raise ValueError("plan-enforced execution differs")
    base_rows = {row["system"]: row for row in base.get("results", [])}
    if set(base_rows) != {"NVIDIA MIG", "NVIDIA MPS", "QUIET"}:
        raise ValueError("plan-enforced base systems differ")
    for row in base_rows.values():
        trace = resolve_evidence_path(
            row["request_trace"]["path"], paths["plan_enforced_base"]
        )
        if sha256(trace) != row["request_trace"]["sha256"]:
            raise ValueError("base request trace hash differs")
    if base_rows["QUIET"]["deadline_misses"] != 0:
        raise ValueError("plan-enforced QUIET smoke missed its deadline")

    producer_scope = load(paths["producer_scope"])
    pipeline_scope = load(paths["pipeline_scope"])
    producer_row = quiet_row(producer_scope)
    pipeline_row = quiet_row(pipeline_scope)
    if (
        producer_scope.get("quiet_gate_scope") != "producer"
        or pipeline_scope.get("quiet_gate_scope") != "pipeline"
        or producer_row.get("pipeline_requests") != 1500
        or pipeline_row.get("pipeline_requests") != 1500
        or producer_row.get("payload_bytes") != 2_304_000
        or pipeline_row.get("payload_bytes") != 2_304_000
    ):
        raise ValueError("QUIET scope ablation contract differs")

    transport = load(paths["transport"])
    systems = transport.get("systems", {})
    if (
        transport.get("kind") != "p9-whisper-transport-williams-aggregate"
        or transport.get("payload_bytes") != 2_304_000
        or any(row.get("requests") != 2000 for row in systems.values())
    ):
        raise ValueError("transport ablation contract differs")
    registered = systems["cross-mig-registered"]
    pinned = systems["cross-mig-pinned"]
    pageable = systems["cross-mig-pageable"]
    same = systems["same-instance-registered"]

    best_boer = min(boer_replay, key=lambda row: row["p99_us"])
    return {
        "schema_version": 1,
        "kind": "p9-resnet-dependent-structural-limit-evidence",
        "proposed_system": "QUIET",
        "common_deadline_us": common_deadline,
        "deadline_lock_sha256": lock_sha,
        "findings": {
            "BOER": {
                "independent_positive_control": True,
                "dependent_feasible": False,
                "best_hardware_observation": best_boer,
                "reason": "spatial service search cannot reserve precedence-constrained stage slack",
            },
            "ParvaGPU": {
                "independent_positive_control": True,
                "dependent_feasible": False,
                "reason": parva_dependent["reason"],
                "segment_requests": parva_dependent["segment_requests"],
            },
            "QUIET": {
                "plan_enforced": True,
                "response_reservation_us": selected["response_reservation_us"],
                "reserved_slack_us": selected["reserved_slack_us"],
                "requests": base_rows["QUIET"]["pipeline_requests"],
                "misses": base_rows["QUIET"]["deadline_misses"],
                "p99_us": base_rows["QUIET"]["pipeline_p99_us"],
                "background_goodput_rps": base_rows["QUIET"]["background_goodput_rps"],
            },
        },
        "mechanism_ablations": {
            "coherent_data_plane": {
                "payload_bytes": 2_304_000,
                "cross_mig_registered_edge_p99_us": registered["edge_p99_us"],
                "same_instance_registered_edge_p99_us": same["edge_p99_us"],
                "pinned_bounce_edge_p99_us": pinned["edge_p99_us"],
                "pageable_bounce_edge_p99_us": pageable["edge_p99_us"],
                "pinned_minus_registered_paired_t95": transport["paired_comparisons"][
                    "cross-mig-pinned_minus_cross_mig_registered_edge_p99_us"
                ]["mean_t95"],
                "interpretation": "MIG device memory is isolated; registered coherent system memory carries the real payload without a host bounce",
            },
            "stage_scope": {
                "producer_only": {
                    "misses": producer_row["deadline_misses"],
                    "requests": producer_row["pipeline_requests"],
                    "background_goodput_rps": producer_row["background_goodput_rps"],
                },
                "full_pipeline": {
                    "misses": pipeline_row["deadline_misses"],
                    "requests": pipeline_row["pipeline_requests"],
                    "background_goodput_rps": pipeline_row["background_goodput_rps"],
                },
            },
        },
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "boer-independent", "boer-dependent", "parva-independent",
        "parva-dependent", "quiet-plan", "plan-enforced-base",
        "producer-scope", "pipeline-scope", "transport",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {
        name: getattr(args, name).resolve()
        for name in (
            "boer_independent", "boer_dependent", "parva_independent",
            "parva_dependent", "quiet_plan", "plan_enforced_base",
            "producer_scope", "pipeline_scope", "transport",
        )
    }
    result = summarize(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
