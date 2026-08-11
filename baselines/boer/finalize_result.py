#!/usr/bin/env python3
"""Bind a BOER search, selected replay, and Thor profiles into a comparison row."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
from typing import Any


UPSTREAM_COMMIT = "df54815de3b1c9059f873a17c13f7d5203eedd3e"
FIDELITY = "upstream-ei-discrete-thor-adapter"


def load(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_close(actual: Any, expected: Any, label: str) -> None:
    if (
        isinstance(actual, bool)
        or isinstance(expected, bool)
        or not isinstance(actual, (int, float))
        or not isinstance(expected, (int, float))
        or not math.isfinite(float(actual))
        or not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9)
    ):
        raise ValueError(f"{label} differs")


def finalize(
    search: dict[str, Any], replay: dict[str, Any], profile_paths: list[pathlib.Path]
) -> dict[str, Any]:
    provenance = search.get("provenance")
    if provenance != {"upstream_commit": UPSTREAM_COMMIT, "fidelity": FIDELITY}:
        raise ValueError("BOER search provenance differs")
    selected = search.get("selected")
    config = replay.get("config")
    policies = replay.get("policies")
    if not isinstance(selected, dict) or not isinstance(config, dict):
        raise ValueError("BOER selected replay is incomplete")
    if not isinstance(policies, list) or len(policies) != 1:
        raise ValueError("BOER replay must contain one policy")
    policy = policies[0]
    if not isinstance(policy, dict) or policy.get("name") != "uncoordinated-borrow":
        raise ValueError("BOER replay used a different runtime mechanism")
    if config.get("borrower_quota") != selected.get("sm_percent"):
        raise ValueError("BOER replay quota differs from the selected point")
    require_close(
        config.get("pressure_rps_per_tenant"), selected.get("offered_rps"),
        "BOER replay offered RPS"
    )
    by_modality = policy.get("goodput_by_modality")
    if not isinstance(selected.get("metrics"), dict) or not isinstance(by_modality, dict):
        raise ValueError("BOER selected or final replay metrics are incomplete")

    profile_hashes: dict[str, str] = {}
    for path in profile_paths:
        resolved = path.resolve()
        evidence_name = f"{resolved.parent.name}/{resolved.name}"
        if not resolved.is_file() or evidence_name in profile_hashes:
            raise ValueError("BOER profile evidence is missing or duplicated")
        profile_hashes[evidence_name] = sha256(resolved)
    if not profile_hashes:
        raise ValueError("BOER result needs Thor profile evidence")
    deadline = replay.get("deadline_ms")
    return {
        "schema_version": 1,
        "system": "BOER",
        "provenance": {
            "upstream_commit": UPSTREAM_COMMIT,
            "fidelity": FIDELITY,
            "thor_profile_sha256": profile_hashes,
            "search_sha256": sha256(pathlib.Path(search["_path"]))
            if isinstance(search.get("_path"), str)
            else None,
            "final_replay_sha256": sha256(pathlib.Path(replay["_path"]))
            if isinstance(replay.get("_path"), str)
            else None,
        },
        "contract": {
            "scenario": config.get("scenario"),
            "epochs": config.get("epochs"),
            "samples_per_epoch": config.get("samples_per_epoch"),
            "period_ms": config.get("period_ms"),
            "pressure_rps_per_tenant": config.get("pressure_rps_per_tenant", 0.0),
            "burst_size": config.get("burst_size"),
            "deadline_ms": deadline,
            "dmr_target": config.get("dmr_target"),
        },
        "metrics": {
            "pressure_goodput_per_second": policy.get("pressure_goodput_per_second"),
            "deadline_miss_rate": policy.get("deadline_miss_rate"),
            "p99_ms": policy.get("critical_p99_ms_max"),
            "critical_requests": policy.get("critical_requests"),
            "deadline_misses": policy.get("deadline_misses"),
            "goodput_by_modality": by_modality,
        },
        "measurement_stage": "final-replay",
        "numeric_comparison_allowed": False,
        "comparison_status": "structural-only-discrete-domain-adapter",
        "selected": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=pathlib.Path, required=True)
    parser.add_argument("--replay", type=pathlib.Path, required=True)
    parser.add_argument("--profile", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    search = load(args.search)
    search["_path"] = str(args.search.resolve())
    replay = load(args.replay)
    replay["_path"] = str(args.replay.resolve())
    result = finalize(search, replay, args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
