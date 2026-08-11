#!/usr/bin/env python3
"""Join current-lock Whisper numeric, planner, and transport evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


PUBLIC_SYSTEMS = {"NVIDIA MIG", "NVIDIA MPS", "Orion", "XSched", "gpulet", "QUIET"}
HEADLINE_SYSTEMS = ("NVIDIA MIG", "NVIDIA MPS", "Orion", "XSched", "Pantheon", "QUIET")


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def evidence(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def summarize(paths: dict[str, Path]) -> dict[str, Any]:
    numeric = load(paths["numeric"])
    systems = numeric.get("systems")
    lock = numeric.get("deadline_lock")
    if (
        numeric.get("kind") != "p9-common-sota-williams-aggregate"
        or numeric.get("workload") != "whisper-projection"
        or numeric.get("proposed_system") != "QUIET"
        or not isinstance(systems, dict) or set(systems) != PUBLIC_SYSTEMS
        or not isinstance(lock, dict) or sha256(Path(lock["path"])) != lock.get("sha256")
    ):
        raise ValueError("invalid current Whisper numeric aggregate")
    lock_sha = lock["sha256"]

    boer_i, boer_d = load(paths["boer_independent"]), load(paths["boer_dependent"])
    if boer_i.get("status") != "selected" or not isinstance(boer_i.get("selected"), dict):
        raise ValueError("BOER independent positive control failed")
    boer_selected = boer_i["selected"]
    boer_independent_metrics = boer_selected.get("metrics")
    if not isinstance(boer_independent_metrics, dict):
        raise ValueError("BOER independent metrics are missing")
    boer_hw = [
        item for item in boer_d.get("observations", []) if item.get("source") == "hardware"
    ]
    if (
        boer_d.get("status") != "no-feasible-configuration"
        or boer_d.get("selected") is not None or not boer_hw
        or boer_d.get("contract", {}).get("deadline_lock_sha256") != lock_sha
        or any(item.get("feasible") is not False for item in boer_hw)
    ):
        raise ValueError("invalid BOER current-lock dependent evidence")

    parva_i, parva_d = load(paths["parva_independent"]), load(paths["parva_dependent"])
    parva_services = parva_i.get("services")
    if (
        parva_i.get("all_slos_met") is not True
        or not isinstance(parva_services, list) or len(parva_services) != 2
    ):
        raise ValueError("ParvaGPU independent positive control failed")
    if (
        parva_d.get("system") != "ParvaGPU" or parva_d.get("feasible") is not False
        or parva_d.get("contract", {}).get("deadline_lock_sha256") != lock_sha
    ):
        raise ValueError("invalid ParvaGPU current-lock dependent evidence")

    transport = load(paths["transport"])
    transport_systems = transport.get("systems")
    if (
        transport.get("kind") != "p9-whisper-transport-williams-aggregate"
        or transport.get("payload_bytes") != 2_304_000
        or not isinstance(transport_systems, dict)
    ):
        raise ValueError("invalid Whisper transport evidence")

    quiet = systems["QUIET"]
    stage_rows = {
        name: {
            "misses": row["misses"],
            "requests": row["requests"],
            "pooled_p99_us": finite(row["pooled_p99_us"], f"{name} p99"),
            "producer_compute_p99_us": finite(
                row["pooled_stage_p99_us"]["producer_compute_us"], f"{name} producer"
            ),
            "edge_transport_p99_us": finite(
                row["pooled_stage_p99_us"]["edge_transport_us"], f"{name} edge"
            ),
            "consumer_compute_p99_us": finite(
                row["pooled_stage_p99_us"]["consumer_compute_us"], f"{name} consumer"
            ),
        }
        for name, row in systems.items()
    }
    advertised_headline = numeric.get("headline_systems")
    if advertised_headline is not None:
        if (
            not isinstance(advertised_headline, list)
            or tuple(row.get("system") for row in advertised_headline) != HEADLINE_SYSTEMS
        ):
            raise ValueError("headline system contract differs")
        headline_stage_rows = {
            row["system"]: (
                {
                    "numeric_comparison_allowed": row.get("numeric_comparison_allowed") is True,
                    "comparison_status": row.get("comparison_status", "functional-only"),
                    **stage_rows[row["system"]],
                }
                if row["system"] in stage_rows and "pooled_p99_us" in stage_rows[row["system"]]
                else {
                    "numeric_comparison_allowed": False,
                    "comparison_status": row.get("comparison_status", "functional-only"),
                }
            )
            for row in advertised_headline
        }
    else:
        headline_stage_rows = {
            name: dict(stage_rows[name])
            for name in ("NVIDIA MIG", "NVIDIA MPS", "Orion", "XSched", "QUIET")
            if name in stage_rows
        }
        headline_stage_rows["Pantheon"] = {
            "numeric_comparison_allowed": False,
            "comparison_status": "functional-only-pending-common-workload-adapter",
        }
    return {
        "schema_version": 1,
        "kind": "p9-current-whisper-structural-evidence",
        "proposed_system": "QUIET",
        "workload": "Whisper Tiny -> 2.304MB coherent edge -> TensorRT projection",
        "deadline_lock": lock,
        "numeric_stage_replay": stage_rows,
        "headline_stage_replay": headline_stage_rows,
        "headline_system_order": list(HEADLINE_SYSTEMS),
        "published_system_boundaries": {
            "BOER": {
                "independent_positive_control": True,
                "independent_selected": {
                    "candidate_id": boer_selected.get("id"),
                    "sm_percent": int(boer_selected["sm_percent"]),
                    "offered_rps_per_service": finite(
                        boer_selected["offered_rps"], "BOER offered rate"
                    ),
                    "worst_p99_ms": finite(
                        boer_independent_metrics["worst_p99_ms"], "BOER independent p99"
                    ),
                    "served_rps": [
                        finite(boer_independent_metrics["served_rps_0"], "BOER served 0"),
                        finite(boer_independent_metrics["served_rps_1"], "BOER served 1"),
                    ],
                    "deadline_miss_rate": finite(
                        boer_independent_metrics["deadline_miss_rate"], "BOER independent DMR"
                    ),
                },
                "dependent_feasible": False,
                "best_measured_p99_ms": min(
                    finite(item["metrics"]["worst_p99_ms"], "BOER p99") for item in boer_hw
                ),
                "hardware_observations": len(boer_hw),
                "limitation": "independent-client configuration search has no stage-DAG slack state",
            },
            "ParvaGPU": {
                "independent_positive_control": True,
                "independent_services": [
                    {
                        "model": item["model"],
                        "served_rps": finite(item["served_rps"], "ParvaGPU served rate"),
                        "p99_ms": finite(item["p99_ms"], "ParvaGPU p99"),
                        "slo_ms": finite(item["slo_ms"], "ParvaGPU SLO"),
                    }
                    for item in parva_services
                ],
                "dependent_feasible": False,
                "reason": parva_d.get("reason"),
                "limitation": "per-service segment admission rejects the producer before dependent placement",
            },
            "QUIET": {
                "observed_zero_miss": quiet["misses"] == 0,
                "confidence_qualified": quiet["slo_confidence_qualified"],
                "mechanism": "communication-aware stage placement and explicit response-slack reservation",
            },
        },
        "transport": {
            "cross_mig_registered_edge_p99_us": transport_systems["cross-mig-registered"]["edge_p99_us"],
            "cross_mig_pinned_edge_p99_us": transport_systems["cross-mig-pinned"]["edge_p99_us"],
            "same_instance_registered_edge_p99_us": transport_systems["same-instance-registered"]["edge_p99_us"],
            "interpretation": "registered coherent system memory avoids a mandatory host bounce",
        },
        "numeric_stage_replay_scope": (
            "legacy-six-system-replay-compatibility; use headline_stage_replay for paper tables"
        ),
        "inputs": {name: evidence(path) for name, path in paths.items()},
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "numeric", "boer-independent", "boer-dependent",
        "parva-independent", "parva-dependent", "transport",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {
        name: getattr(args, name).resolve()
        for name in (
            "numeric", "boer_independent", "boer_dependent",
            "parva_independent", "parva_dependent", "transport",
        )
    }
    result = summarize(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
