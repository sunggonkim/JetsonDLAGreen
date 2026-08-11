#!/usr/bin/env python3
"""Build a fast P9 comparison matrix with honest executable-status labels."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


CORE_ROWS = {
    "static-mig": "NVIDIA MIG isolation",
    "same-mig": "NVIDIA MPS spatial sharing",
    "mig-governor": "QUIET",
}


def load(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def metrics(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "pressure_goodput_per_second": policy["pressure_goodput_per_second"],
        "critical_requests": policy["critical_requests"],
        "deadline_misses": policy["deadline_misses"],
        "deadline_miss_rate": policy["deadline_miss_rate"],
        "critical_p99_ms": policy["critical_p99_ms_max"],
    }


def measured_rows(core: dict[str, Any], boer: dict[str, Any], scenario: str) -> list[dict[str, Any]]:
    for summary in (core, boer):
        config = summary.get("config", {})
        if (
            config.get("scenario") != scenario
            or config.get("epochs") != 1
            or config.get("samples_per_epoch") != 80
            or config.get("pressure_rps_per_tenant") != 200.0
        ):
            raise ValueError(f"{scenario} smoke contract differs")
    policies = core.get("policies")
    if not isinstance(policies, list):
        raise ValueError("core smoke lacks policies")
    by_id = {policy.get("name"): policy for policy in policies}
    rows = []
    for policy_id, system in CORE_ROWS.items():
        if policy_id not in by_id:
            raise ValueError(f"core smoke lacks {policy_id}")
        rows.append(
            {
                "scenario": scenario,
                "system": system,
                "status": "measured-smoke",
                "metrics": metrics(by_id[policy_id]),
            }
        )
    boer_policies = boer.get("policies")
    if not isinstance(boer_policies, list) or len(boer_policies) != 1:
        raise ValueError("BOER smoke must contain one result")
    rows.append(
        {
            "scenario": scenario,
            "system": "BOER (Thor port)",
            "status": "measured-smoke",
            "metrics": metrics(boer_policies[0]),
        }
    )
    return rows


def summarize(
    independent_core: dict[str, Any],
    dependent_core: dict[str, Any],
    independent_boer: dict[str, Any],
    dependent_boer: dict[str, Any],
    parvagpu: dict[str, Any],
    orion: dict[str, Any],
) -> dict[str, Any]:
    rows = measured_rows(independent_core, independent_boer, "independent")
    rows += measured_rows(dependent_core, dependent_boer, "dependent")
    if parvagpu.get("feasible") is not False:
        raise ValueError("ParvaGPU smoke did not preserve fixed-layout infeasibility")
    if orion.get("numeric_comparison_allowed") is not False:
        raise ValueError("Orion probe unexpectedly permits numeric comparison")
    for scenario in ("independent", "dependent"):
        rows.extend(
            [
                {
                    "scenario": scenario,
                    "system": "ParvaGPU (Thor port)",
                    "status": "configuration-infeasible",
                    "reason": parvagpu.get("reason"),
                    "metrics": None,
                },
                {
                    "scenario": scenario,
                    "system": "Orion (Thor probe)",
                    "status": "managed-client-integration-required",
                    "reason": orion.get("reason"),
                    "metrics": None,
                },
            ]
        )
    return {
        "schema_version": 1,
        "scope": "fixed-2g+1g-functional-smoke",
        "proposed_system": "QUIET",
        "workloads": {
            "critical": "ResNet-50 vision on 2g, burst 8 every 20 ms",
            "independent": "concurrent Whisper audio and DistilBERT language",
            "dependent": "control token only; producer tensor is not transferred",
            "deadline_ms": independent_core["deadline_ms"],
            "pressure_rps_per_tenant": 200,
        },
        "warning": (
            "80-request functional smoke; not a statistical performance claim. "
            "Dependent rows validate control ordering only and are invalid for "
            "cross-MIG dataflow claims."
        ),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent-core", type=pathlib.Path, required=True)
    parser.add_argument("--dependent-core", type=pathlib.Path, required=True)
    parser.add_argument("--independent-boer", type=pathlib.Path, required=True)
    parser.add_argument("--dependent-boer", type=pathlib.Path, required=True)
    parser.add_argument("--parvagpu", type=pathlib.Path, required=True)
    parser.add_argument("--orion", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = summarize(
        load(args.independent_core), load(args.dependent_core),
        load(args.independent_boer), load(args.dependent_boer),
        load(args.parvagpu), load(args.orion),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
