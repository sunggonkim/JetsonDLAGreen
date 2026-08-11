#!/usr/bin/env python3
"""Summarize positive and negative controls for published P9 comparators."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def evidence(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def summarize(paths: dict[str, Path]) -> dict[str, Any]:
    raw = {name: load(path) for name, path in paths.items()}
    boer_i, boer_d = raw["boer_independent"], raw["boer_dependent"]
    parva_i, parva_d = raw["parva_independent"], raw["parva_dependent"]
    orion, quiet = raw["orion"], raw["quiet"]
    if boer_i.get("system") != "BOER" or boer_i.get("status") != "selected":
        raise ValueError("BOER independent positive control is missing")
    if boer_d.get("system") != "BOER" or boer_d.get("status") != "no-feasible-configuration":
        raise ValueError("BOER dependent negative control is missing")
    if parva_i.get("system") != "ParvaGPU" or parva_i.get("all_slos_met") is not True:
        raise ValueError("ParvaGPU independent positive control is missing")
    if parva_d.get("system") != "ParvaGPU" or parva_d.get("feasible") is not False:
        raise ValueError("ParvaGPU dependent negative control is missing")
    if orion.get("system") != "Orion" or orion.get("numeric_comparison_allowed") is not False:
        raise ValueError("Orion compatibility evidence is missing")
    if quiet.get("proposed_system") != "QUIET" or quiet.get("status") != "selected":
        raise ValueError("QUIET dependent positive control is missing")
    selected = boer_i["selected"]
    return {
        "schema_version": 1,
        "kind": "p9-sota-workload-scope-smoke",
        "scope": "functional-positive-negative-controls-not-formal-statistics",
        "proposed_system": "QUIET",
        "comparators": [
            {
                "system": "BOER",
                "independent": {
                    "status": "supported-measured",
                    "selected_quota_pair": [
                        selected["sm_percent"], selected["complement_sm_percent"]
                    ],
                    "offered_rps_per_service": selected["offered_rps"],
                    "worst_p99_ms": selected["metrics"]["worst_p99_ms"],
                },
                "dependent": {
                    "status": "no-feasible-configuration",
                    "reason": "per-service complementary-MPS search does not reserve a tensor DAG path",
                },
            },
            {
                "system": "ParvaGPU",
                "independent": {
                    "status": "supported-measured",
                    "services": parva_i["services"],
                },
                "dependent": {
                    "status": "allocation-infeasible",
                    "reason": parva_d["reason"],
                },
            },
            {
                "system": "Orion",
                "independent": {"status": "unsupported-tensorrt-surface"},
                "dependent": {"status": "unsupported-tensorrt-surface"},
                "reason": orion["reason"],
            },
        ],
        "quiet_dependent": {
            "status": "supported-measured",
            "selected_plan": quiet["selected_plan"],
        },
        "inputs": {name: evidence(path) for name, path in paths.items()},
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "boer-independent", "boer-dependent", "parva-independent",
        "parva-dependent", "orion", "quiet",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize({
        "boer_independent": args.boer_independent.resolve(),
        "boer_dependent": args.boer_dependent.resolve(),
        "parva_independent": args.parva_independent.resolve(),
        "parva_dependent": args.parva_dependent.resolve(),
        "orion": args.orion.resolve(),
        "quiet": args.quiet.resolve(),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
