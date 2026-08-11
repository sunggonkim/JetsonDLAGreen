#!/usr/bin/env python3
"""Summarize independent/dependent P9 performance without renaming controls."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


PRESENTATION = {
    "static-mig": ("NVIDIA MIG isolation", "baseline"),
    "resident-full-gate": ("Resident-only quiescence", "ablation"),
    "fixed-full-gate": ("Static full gating", "ablation"),
    "mig-governor": ("QUIET", "proposed"),
}


def load(path: pathlib.Path, scenario: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("config", {}).get("scenario") != scenario:
        raise ValueError(f"{path} is not the {scenario} scenario")
    policies = value.get("policies")
    if not isinstance(policies, list):
        raise ValueError(f"{path} lacks policies")
    rows = []
    for policy in policies:
        policy_id = policy.get("name")
        if policy_id not in PRESENTATION:
            continue
        label, role = PRESENTATION[policy_id]
        rows.append(
            {
                "policy_id": policy_id,
                "label": label,
                "role": role,
                "critical_requests": policy.get("critical_requests"),
                "deadline_misses": policy.get("deadline_misses"),
                "deadline_miss_rate": policy.get("deadline_miss_rate"),
                "critical_p99_ms_max": policy.get("critical_p99_ms_max"),
                "pressure_goodput_per_second": policy.get(
                    "pressure_goodput_per_second"
                ),
                "goodput_by_modality": policy.get("goodput_by_modality"),
            }
        )
    return {
        "scenario": scenario,
        "deadline_ms": value.get("deadline_ms"),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent", type=pathlib.Path, required=True)
    parser.add_argument("--dependent", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    independent = load(args.independent, "independent")
    dependent = load(args.dependent, "dependent")
    if abs(float(independent["deadline_ms"]) - float(dependent["deadline_ms"])) > 1e-9:
        raise ValueError("scenario runs used different deadlines")
    result = {
        "schema_version": 1,
        "proposed_system": "QUIET",
        "scope": "fixed-2g+1g-MIG",
        "temperature_role": "passive-safety-log-only",
        "deadline_ms": independent["deadline_ms"],
        "scenarios": [independent, dependent],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
