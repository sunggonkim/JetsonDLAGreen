#!/usr/bin/env python3
"""Join positive controls, dependent failures, and transport evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


DEADLINE_SHA = "d3da4431a4f047ee133649a51dbd8ccc8716318fccfe86dc4d6ae0e34d1d8fc0"
PAYLOAD_BYTES = 2_304_000


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def evidence(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def summarize(paths: dict[str, Path]) -> dict[str, Any]:
    numeric = load(paths["numeric"])
    systems = numeric.get("systems", {})
    if (
        numeric.get("kind") != "p9-numeric-mechanism-williams-aggregate"
        or numeric.get("contract", {}).get("deadline_lock", {}).get("sha256")
        != DEADLINE_SHA
        or systems.get("QUIET", {}).get("requests") != 9000
    ):
        raise ValueError("invalid balanced numeric evidence")

    boer_i = load(paths["boer_independent"])
    boer_d = load(paths["boer_dependent"])
    boer_selected = boer_i.get("selected")
    if boer_i.get("status") != "selected" or not isinstance(boer_selected, dict):
        raise ValueError("BOER independent positive control did not select a point")
    if boer_d.get("status") != "no-feasible-configuration" or boer_d.get("selected") is not None:
        raise ValueError("BOER dependent evidence is not a no-feasible search")
    boer_hw = [
        item for item in boer_d.get("observations", []) if item.get("source") == "hardware"
    ]
    if not boer_hw or any(item.get("feasible") is not False for item in boer_hw):
        raise ValueError("BOER dependent hardware observations are malformed")

    parva_i = load(paths["parva_independent"])
    parva_d = load(paths["parva_dependent"])
    if parva_i.get("system") != "ParvaGPU" or parva_i.get("all_slos_met") is not True:
        raise ValueError("ParvaGPU independent positive control failed")
    if (
        parva_d.get("system") != "ParvaGPU"
        or parva_d.get("feasible") is not False
        or parva_d.get("contract", {}).get("payload_bytes") != PAYLOAD_BYTES
        or parva_d.get("contract", {}).get("deadline_lock_sha256") != DEADLINE_SHA
    ):
        raise ValueError("invalid ParvaGPU dependent evidence")

    registered = load(paths["registered"])
    pinned = load(paths["pinned"])
    pageable = load(paths["pageable"])
    same = load(paths["same_instance"])
    transports = {"registered": registered, "pinned": pinned, "pageable": pageable}
    for label, raw in transports.items():
        if raw.get("status") != "ok" or raw.get("payload_bytes") != PAYLOAD_BYTES:
            raise ValueError(f"invalid {label} transport evidence")
        if raw.get("checksum_failures") != 0:
            raise ValueError(f"{label} transport has checksum failures")
    if same.get("producer_uuid") != same.get("consumer_uuid"):
        raise ValueError("same-instance control does not use one GPU instance")

    direct_p99 = finite(
        registered["stage_latency_us"]["validation_excluded_end_to_end_p99"],
        "registered p99",
    )
    balanced_transport = None
    if "transport_balanced" in paths:
        balanced_transport = load(paths["transport_balanced"])
        balanced_systems = balanced_transport.get("systems", {})
        if (
            balanced_transport.get("kind")
            != "p9-whisper-transport-williams-aggregate"
            or balanced_transport.get("payload_bytes") != PAYLOAD_BYTES
            or set(balanced_systems)
            != {
                "cross-mig-registered", "cross-mig-pinned",
                "cross-mig-pageable", "same-instance-registered",
            }
            or any(row.get("requests") != 2000 for row in balanced_systems.values())
        ):
            raise ValueError("invalid balanced transport evidence")
    return {
        "schema_version": 1,
        "kind": "p9-structural-limit-evidence",
        "proposed_system": "QUIET",
        "workload": "Whisper producer -> 2.304MB coherent edge -> projection consumer",
        "deadline_lock_sha256": DEADLINE_SHA,
        "findings": [
            {
                "system": "BOER",
                "independent_positive_control": {
                    "feasible": True,
                    "worst_p99_ms": finite(
                        boer_selected["metrics"]["worst_p99_ms"], "BOER independent p99"
                    ),
                    "served_rps": [
                        finite(boer_selected["metrics"]["served_rps_0"], "BOER served 0"),
                        finite(boer_selected["metrics"]["served_rps_1"], "BOER served 1"),
                    ],
                },
                "dependent_dag": {
                    "feasible": False,
                    "best_measured_p99_ms": min(
                        finite(item["metrics"]["worst_p99_ms"], "BOER dependent p99")
                        for item in boer_hw
                    ),
                    "limitation": "independent-client spatial search does not model stage precedence or reclaim producer slack",
                },
            },
            {
                "system": "ParvaGPU",
                "independent_positive_control": {
                    "feasible": True,
                    "services": parva_i["services"],
                },
                "dependent_dag": {
                    "feasible": False,
                    "reason": parva_d["reason"],
                    "limitation": "segment admission treats stages as independent services and rejects the producer before DAG placement",
                },
            },
            {
                "system": "QUIET",
                "dependent_dag": {
                    "feasible": systems["QUIET"]["same_slo_goodput_comparable"],
                    "requests": systems["QUIET"]["requests"],
                    "misses": systems["QUIET"]["misses"],
                    "p99_us": systems["QUIET"]["pooled_p99_us"],
                    "background_goodput_rps": systems["QUIET"]["background_goodput_rps_mean"],
                    "mechanism": "stage-aware quiescence preserves the coherent edge and reclaims only safe DAG slack",
                },
            },
        ],
        "transport": {
            "evidence_scope": (
                "four-sequence-williams-2000-requests-per-treatment"
                if balanced_transport is not None
                else "functional-smoke"
            ),
            "registered_direct_p99_us": direct_p99,
            "pinned_bounce_p99_us": finite(
                pinned["stage_latency_us"]["validation_excluded_end_to_end_p99"],
                "pinned p99",
            ),
            "pageable_bounce_p99_us": finite(
                pageable["stage_latency_us"]["validation_excluded_end_to_end_p99"],
                "pageable p99",
            ),
            "cross_mig_edge_p99_us": finite(
                registered["stage_latency_us"]["edge_transport_p99"], "cross-MIG edge p99"
            ),
            "same_instance_edge_p99_us": finite(
                same["stage_latency_us"]["edge_transport_p99"], "same-instance edge p99"
            ),
            "interpretation": "MIG isolation forbids device-memory sharing, but coherent registered system memory avoids an explicit bounce; residual slowdown is shared-SoC interference, not mandatory payload copying",
            "balanced_raw_replay": (
                None
                if balanced_transport is None
                else {
                    "cross_mig_registered_edge_p99_us": balanced_transport["systems"]["cross-mig-registered"]["edge_p99_us"],
                    "same_instance_registered_edge_p99_us": balanced_transport["systems"]["same-instance-registered"]["edge_p99_us"],
                    "pinned_edge_p99_us": balanced_transport["systems"]["cross-mig-pinned"]["edge_p99_us"],
                    "pageable_edge_p99_us": balanced_transport["systems"]["cross-mig-pageable"]["edge_p99_us"],
                    "paired_comparisons": balanced_transport["paired_comparisons"],
                }
            ),
        },
        "inputs": {name: evidence(path) for name, path in paths.items()},
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "numeric", "boer-independent", "boer-dependent", "parva-independent",
        "parva-dependent", "registered", "pinned", "pageable", "same-instance",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--transport-balanced", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {name.replace("-", "_"): getattr(args, name.replace("-", "_")).resolve()
             for name in (
                 "numeric", "boer-independent", "boer-dependent", "parva-independent",
                 "parva-dependent", "registered", "pinned", "pageable", "same-instance",
             )}
    if args.transport_balanced is not None:
        paths["transport_balanced"] = args.transport_balanced.resolve()
    result = summarize(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
