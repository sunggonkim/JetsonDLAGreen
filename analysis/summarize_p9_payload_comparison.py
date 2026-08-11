#!/usr/bin/env python3
"""Build the dependent-payload smoke comparison and its canonical headline view.

The legacy ``systems`` array is retained for replay compatibility with old
artifacts.  New consumers must use ``headline_systems``: it contains exactly
the proposed system (QUIET), the two executable vendor controls, and the
three pinned published comparators.  A functional gate is never silently
promoted to a numeric SOTA result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


PUBLIC_SYSTEMS = (
    "NVIDIA MIG",
    "NVIDIA MPS",
    "Orion",
    "XSched",
    "gpulet",
    "QUIET",
)

# Stable presentation contract for new tables and paper generators.  Keep the
# legacy tuple above because old raw summaries and tests are immutable evidence.
HEADLINE_SYSTEMS = (
    "NVIDIA MIG",
    "NVIDIA MPS",
    "Orion",
    "XSched",
    "Pantheon",
    "QUIET",
)

HEADLINE_NUMERIC_DEFAULTS = {
    "NVIDIA MIG": False,
    "NVIDIA MPS": True,
    "Orion": False,
    "XSched": True,
    "Pantheon": False,
    "QUIET": True,
}

NATIVE_GATES = {
    "Orion": {
        "kind": "orion-thor-native-positive-control-verification",
        "commit": "20f9469764fb96d94ce23a8e70615196e9ce4ba1",
    },
    "XSched": {
        "kind": "xsched-thor-native-positive-control",
        "commit": "bd494cb7a72958cd11900243a0798df00d856c6e",
    },
    "Pantheon": {
        "kind": "pantheon-thor-native-positive-control",
        "commit": "1caa4321fe9f9902ffacb78978f11a32a7a62f64",
    },
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measured_row(frontier: dict[str, Any], system: str, offered: float) -> dict[str, Any]:
    if "results" in frontier:
        actual_offered = frontier.get("background_offered_rps")
        if actual_offered != offered:
            raise ValueError("stress summary offered load differs")
        matches = [
            row for row in frontier["results"] if row.get("system") == system
        ]
        if len(matches) != 1:
            raise ValueError(f"stress summary lacks one {system} row")
        row = matches[0]
        requests = row["pipeline_requests"]
        misses = row["deadline_misses"]
        return {
            "system": system,
            "status": "measured-smoke",
            "offered_rps": offered,
            "background_goodput_rps": row["background_goodput_rps"],
            "requests": requests,
            "misses": misses,
            "dmr": misses / requests,
            "deadline_p99_us": row["pipeline_p99_us"],
            "deadline_us": frontier.get("deadline_us"),
            "deadline_mode": row["deadline_mode"],
            "gate_scope": row.get("gate_scope"),
            "observed_deadline_feasible": misses == 0,
        }
    if "systems" in frontier:
        if frontier.get("offered_background_rps") != offered:
            raise ValueError("prior comparison offered load differs")
        matches = [
            row for row in frontier["systems"] if row.get("system") == system
        ]
        if len(matches) != 1 or matches[0].get("status") != "measured-smoke":
            raise ValueError(f"prior comparison lacks one measured {system} row")
        return dict(matches[0])
    matches = [
        row
        for row in frontier.get("rows", [])
        if row.get("system") == system and row.get("offered_rps") == offered
    ]
    if len(matches) != 1:
        raise ValueError(f"frontier lacks one {system} row at {offered} RPS")
    row = matches[0]
    return {
        "system": system,
        "status": "measured-smoke",
        "offered_rps": offered,
        "background_goodput_rps": row["background_goodput_rps"],
        "requests": row["requests"],
        "misses": row["misses"],
        "dmr": row["dmr"],
        "post_release_p99_us": row["post_release_p99_us"],
        "arrival_bound_feasible": row["arrival_bound_feasible"],
    }


def native_gate_row(system: str, gate: dict[str, Any]) -> dict[str, Any]:
    spec = NATIVE_GATES[system]
    if gate.get("kind") != spec["kind"]:
        raise ValueError(f"invalid {system} native gate kind")
    if gate.get("upstream_commit") != spec["commit"]:
        raise ValueError(f"{system} native gate does not use the pinned upstream")
    if gate.get("numeric_comparison_allowed") is not False:
        raise ValueError(f"{system} functional gate cannot be used as a numeric result")

    if system == "Orion":
        if gate.get("status") != "passed":
            raise ValueError("Orion native gate did not pass")
        for key in ("driver_launches", "reordered_decisions"):
            value = gate.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"Orion native gate lacks {key}")
    elif system == "XSched":
        for key in (
            "be_completed_requests",
            "hp_completed_requests",
            "be_suspend_transitions",
            "be_resume_transitions",
            "measurement_overlap_ns",
        ):
            value = gate.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"XSched native gate lacks {key}")
    elif system == "Pantheon":
        full = gate.get("full_exit_job")
        early = gate.get("early_exit_job")
        if not isinstance(full, dict) or full.get("last_block") != 1:
            raise ValueError("Pantheon native gate lacks the full-exit path")
        if not isinstance(early, dict) or early.get("last_block") != 0:
            raise ValueError("Pantheon native gate lacks the early-exit path")

    return {
        "system": system,
        "status": "native-functional-gate-passed",
        "numeric_result": None,
        "reason": "same-workload numeric campaign is pending",
        "upstream_commit": spec["commit"],
    }


def bless_gate_row(gate: dict[str, Any]) -> dict[str, Any]:
    required = {
        "precreated-2-4-6-8-sm-tensorrt-context-replicas",
        "selected-only-logical-tensorrt-launch-admission",
        "shadow-replica-advancement",
        "safe-boundary-restricted-to-unrestricted-switch",
        "independent-safe-boundary-profile-and-held-out-replay",
    }
    completed = gate.get("completed_gates")
    if (
        gate.get("kind") != "bless-thor-tensorrt-fidelity-gate"
        or gate.get("status") != "passed-functional-gates"
        or gate.get("numeric_comparison_allowed") is not False
        or not isinstance(completed, list)
        or not required.issubset(set(completed))
        or gate.get("remaining_gate")
        != "drive the frozen safe boundaries with BLESS relative-progress squad scheduling on the common workload"
    ):
        raise ValueError("invalid BLESS TensorRT fidelity gate")
    return {
        "system": "BLESS",
        "status": "native-functional-gate-passed",
        "numeric_result": None,
        "reason": "common-workload BLESS scheduler-driven numeric campaign is pending",
        "completed_fidelity_gates": len(completed),
    }


def orion_numeric_row(value: dict[str, Any], offered: float) -> dict[str, Any]:
    if (
        value.get("kind")
        != "orion-thor-resnet-control-numeric-smoke-verification"
        or value.get("system") != "Orion (Thor port)"
        or value.get("status") != "passed-smoke"
        or value.get("workload") != "resnet10-layer7-cov-to-control-mlp"
        or value.get("payload_bytes") != 14_720
        or value.get("requests") != 100
        or value.get("checksum_failures") != 0
        or value.get("unique_payload_checksums", 0) < 2
        or value.get("unique_policy_output_checksums", 0) < 2
        or value.get("token_only") is not False
        or value.get("scheduler", {}).get("event_records", 0) <= 0
    ):
        raise ValueError("Orion same-workload numeric evidence differs")
    return {
        "system": "Orion",
        "status": "measured-smoke",
        "offered_rps": offered,
        "background_goodput_rps": value["background_goodput_rps"],
        "requests": value["requests"],
        "misses": value["misses"],
        "dmr": value["dmr"],
        "deadline_p99_us": value["p99_us"],
        "deadline_us": value["deadline_us"],
        "deadline_mode": "wall",
        "observed_deadline_feasible": value["misses"] == 0,
    }


def xsched_numeric_row(value: dict[str, Any], offered: float) -> dict[str, Any]:
    scheduler = value.get("scheduler")
    background = value.get("background_window")
    if (
        value.get("kind")
        != "xsched-thor-resnet-control-numeric-smoke-verification"
        or value.get("system") != "XSched (Thor port)"
        or value.get("upstream_commit") != NATIVE_GATES["XSched"]["commit"]
        or value.get("status") != "passed-smoke"
        or value.get("workload") != "resnet10-layer7-cov-to-control-mlp"
        or value.get("payload_bytes") != 14_720
        or value.get("requests") != 100
        or value.get("checksum_failures") != 0
        or value.get("unique_payload_checksums", 0) < 2
        or value.get("unique_policy_output_checksums", 0) < 2
        or value.get("token_only") is not False
        or not isinstance(scheduler, dict)
        or scheduler.get("connected_clients") != 3
        or scheduler.get("xqueue_clients") != 3
        or scheduler.get("be_suspend_transitions", 0) <= 0
        or scheduler.get("be_resume_transitions", 0) <= 0
        or not isinstance(background, dict)
        or background.get("completed_requests", 0) <= 0
    ):
        raise ValueError("XSched same-workload numeric evidence differs")
    return {
        "system": "XSched",
        "status": "measured-smoke",
        "offered_rps": offered,
        "background_goodput_rps": value["background_goodput_rps"],
        "requests": value["requests"],
        "misses": value["misses"],
        "dmr": value["dmr"],
        "deadline_p99_us": value["p99_us"],
        "deadline_us": value["deadline_us"],
        "deadline_mode": "wall",
        "observed_deadline_feasible": value["misses"] == 0,
    }


def gpulet_numeric_row(value: dict[str, Any], offered: float) -> dict[str, Any]:
    action = value.get("selected_action")
    if (
        value.get("kind")
        != "gpulet-thor-resnet-control-numeric-smoke-verification"
        or value.get("system") != "gpulet (Thor port)"
        or value.get("upstream_commit")
        != "3c1c2aad3b33edcef20e549d5093c43af497e6ae"
        or value.get("status") != "passed-smoke"
        or value.get("workload") != "resnet10-layer7-cov-to-control-mlp"
        or value.get("payload_bytes") != 14_720
        or value.get("requests") != 100
        or value.get("checksum_failures") != 0
        or value.get("unique_payload_checksums", 0) < 2
        or value.get("unique_policy_output_checksums", 0) < 2
        or value.get("token_only") is not False
        or not isinstance(value.get("spatial_schedule_feasible"), bool)
        or not isinstance(action, dict)
        or action.get("semantics") not in {
            "gpulet-best-fit", "diagnostic-largest-critical-partition"
        }
    ):
        raise ValueError("gpulet same-workload numeric evidence differs")
    feasible = value["spatial_schedule_feasible"]
    return {
        "system": "gpulet",
        "status": "measured-smoke" if feasible else "structural-diagnostic",
        "numeric_comparison_allowed": feasible,
        "comparison_status": (
            "numeric-smoke" if feasible else "structural-only-until-spatial-search-feasible"
        ),
        "offered_rps": offered,
        "background_goodput_rps": value["background_goodput_rps"],
        "requests": value["requests"],
        "misses": value["misses"],
        "dmr": value["dmr"],
        "deadline_p99_us": value["p99_us"],
        "deadline_us": value["deadline_us"],
        "deadline_mode": "wall",
        "observed_deadline_feasible": value["misses"] == 0,
        "spatial_schedule_feasible": feasible,
        "selected_action": action,
    }


def headline_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a legacy row into the explicit public presentation contract."""

    system = row.get("system")
    if system not in HEADLINE_SYSTEMS:
        raise ValueError(f"{system!r} is not a headline system")
    result = dict(row)
    result.setdefault(
        "numeric_comparison_allowed", HEADLINE_NUMERIC_DEFAULTS[system]
    )
    result.setdefault(
        "comparison_status",
        "numeric-smoke" if result["numeric_comparison_allowed"] else "functional-only",
    )
    result["public_system_name"] = "QUIET" if system == "QUIET" else system
    return result


def common_numeric_deadline(rows: list[dict[str, Any]]) -> float | None:
    values = [row.get("deadline_us") for row in rows]
    if not all(value is not None for value in values):
        return None
    deadline = float(values[0])
    if any(
        not math.isclose(float(value), deadline, abs_tol=1e-9)
        for value in values[1:]
    ):
        raise ValueError("numeric comparison rows use different deadlines")
    return deadline


def summarize(
    frontier_path: Path,
    boer_path: Path,
    parva_path: Path,
    orion_path: Path,
    xsched_path: Path,
    bless_path: Path,
    pantheon_path: Path,
    gpulet_numeric_path: Path,
    offered_rps: float,
    payload_gate_path: Path | None = None,
    orion_numeric_path: Path | None = None,
    xsched_numeric_path: Path | None = None,
) -> dict[str, Any]:
    if not math.isfinite(offered_rps) or offered_rps <= 0:
        raise ValueError("offered_rps must be positive and finite")
    frontier = load(frontier_path)
    boer = load(boer_path)
    parva = load(parva_path)
    orion = load(orion_path)
    xsched = load(xsched_path)
    bless = load(bless_path)
    pantheon = load(pantheon_path)
    gpulet_numeric = load(gpulet_numeric_path)
    payload_gate = load(payload_gate_path) if payload_gate_path is not None else None
    orion_numeric = load(orion_numeric_path) if orion_numeric_path is not None else None
    xsched_numeric = load(xsched_numeric_path) if xsched_numeric_path is not None else None
    if "results" in frontier:
        if [row.get("system") for row in frontier["results"]].count("QUIET") != 1:
            raise ValueError("stress summary does not expose one QUIET row")
    elif frontier.get("proposed_system") != "QUIET":
        raise ValueError("frontier does not expose QUIET as the proposed system")
    if boer.get("system") != "BOER" or boer.get("status") not in {
        "selected",
        "no-feasible-configuration",
    }:
        raise ValueError("invalid BOER search result")
    if parva.get("system") != "ParvaGPU" or not isinstance(
        parva.get("feasible"), bool
    ):
        raise ValueError("invalid ParvaGPU result")

    legacy_rows = [
        measured_row(frontier, "NVIDIA MIG", offered_rps),
        measured_row(frontier, "NVIDIA MPS", offered_rps),
        orion_numeric_row(orion_numeric, offered_rps)
        if orion_numeric is not None
        else native_gate_row("Orion", orion),
        xsched_numeric_row(xsched_numeric, offered_rps)
        if xsched_numeric is not None
        else native_gate_row("XSched", xsched),
        gpulet_numeric_row(gpulet_numeric, offered_rps),
        measured_row(frontier, "QUIET", offered_rps),
    ]
    if tuple(row["system"] for row in legacy_rows) != PUBLIC_SYSTEMS:
        raise AssertionError("public comparison system order changed")
    row_by_system = {row["system"]: row for row in legacy_rows}
    row_by_system["Pantheon"] = native_gate_row("Pantheon", pantheon)
    headline_rows = [headline_row(row_by_system[name]) for name in HEADLINE_SYSTEMS]
    common_deadline_us = common_numeric_deadline(legacy_rows)
    workload_name = frontier.get("workload", "resnet-control")
    if workload_name != "whisper-projection":
        if payload_gate is None or (
            payload_gate.get("kind")
            != "p9-resnet-layer7-control-mlp-cross-mig-smoke"
            or payload_gate.get("status") != "passed"
            or payload_gate.get("requests") != 100
            or payload_gate.get("edge", {}).get("producer_tensor") != "Layer7_cov"
            or payload_gate.get("edge", {}).get("consumer_tensor") != "features"
            or payload_gate.get("edge", {}).get("shape") != [1, 4, 23, 40]
            or payload_gate.get("edge", {}).get("bytes") != 14_720
            or payload_gate.get("checksum_failures") != 0
            or payload_gate.get("unique_payload_checksums", 0) < 2
            or payload_gate.get("unique_policy_output_checksums", 0) < 2
            or payload_gate.get("token_only") is not False
        ):
            raise ValueError("dependent payload gate differs")
    if isinstance(workload_name, dict):
        workload = dict(workload_name)
    elif workload_name == "whisper-projection":
        workload = {
            "producer": "TensorRT Whisper Tiny encoder",
            "edge_payload_bytes": 2304000,
            "consumer": "TensorRT projection MLP",
            "background": "TensorRT DistilBERT SST-2",
            "layout": "fixed Thor MIG 2g+1g",
            "deadline_us": frontier.get("deadline_us"),
            "deadline_mode": "validation-excluded",
        }
    else:
        workload = {
            "producer": "TensorRT ResNet10 Layer7_cov",
            "edge_payload_bytes": 14720,
            "consumer": "TensorRT control MLP",
            "background": "TensorRT DistilBERT SST-2",
            "layout": "fixed Thor MIG 2g+1g",
            "payload_gate_sha256": digest(payload_gate_path),
        }
    return {
        "schema_version": 5,
        "kind": "p9-dependent-payload-six-system-smoke",
        "workload": workload,
        "proposed_system": "QUIET",
        "offered_background_rps": offered_rps,
        "common_deadline_us": common_deadline_us,
        "scope": "functional-smoke-not-formal-statistics",
        "inputs": {
            "frontier": {"path": str(frontier_path.resolve()), "sha256": digest(frontier_path)},
            "boer": {"path": str(boer_path.resolve()), "sha256": digest(boer_path)},
            "parvagpu": {"path": str(parva_path.resolve()), "sha256": digest(parva_path)},
            "orion": {"path": str(orion_path.resolve()), "sha256": digest(orion_path)},
            "xsched": {"path": str(xsched_path.resolve()), "sha256": digest(xsched_path)},
            "bless": {"path": str(bless_path.resolve()), "sha256": digest(bless_path)},
            "pantheon": {"path": str(pantheon_path.resolve()), "sha256": digest(pantheon_path)},
            "gpulet_numeric": {
                "path": str(gpulet_numeric_path.resolve()),
                "sha256": digest(gpulet_numeric_path),
            },
            **(
                {
                    "payload_gate": {
                        "path": str(payload_gate_path.resolve()),
                        "sha256": digest(payload_gate_path),
                    }
                }
                if payload_gate_path is not None
                else {}
            ),
            **(
                {
                    "orion_numeric": {
                        "path": str(orion_numeric_path.resolve()),
                        "sha256": digest(orion_numeric_path),
                    }
                }
                if orion_numeric_path is not None
                else {}
            ),
            **(
                {
                    "xsched_numeric": {
                        "path": str(xsched_numeric_path.resolve()),
                        "sha256": digest(xsched_numeric_path),
                    }
                }
                if xsched_numeric_path is not None
                else {}
            ),
        },
        # ``systems`` is the immutable legacy replay view.  New tables must
        # consume ``headline_systems`` and cannot accidentally rank gpulet.
        "systems": legacy_rows,
        "headline_systems": headline_rows,
        "headline_contract": {
            "system_order": list(HEADLINE_SYSTEMS),
            "proposed_system": "QUIET",
            "numeric_comparison_systems": [
                row["system"]
                for row in headline_rows
                if row.get("numeric_comparison_allowed") is True
            ],
            "functional_only_systems": [
                row["system"]
                for row in headline_rows
                if row.get("numeric_comparison_allowed") is not True
            ],
            "gpulet_role": "structural-diagnostic-excluded-from-headline",
        },
        "edge_secondary": native_gate_row("Pantheon", pantheon),
        "literature_functional": [bless_gate_row(bless)],
        "structural_controls": [
            {
                "system": "BOER",
                "status": boer["status"],
                "numeric_result": boer.get("selected"),
                "role": "MIG+MPS provisioning control; not an online runtime row",
            },
            {
                "system": "ParvaGPU",
                "status": "allocation-feasible" if parva["feasible"] else "allocation-infeasible",
                "numeric_result": None,
                "reason": parva.get("reason"),
                "role": "MIG allocation control; not an online runtime row",
            },
        ],
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--boer", type=Path, required=True)
    parser.add_argument("--parvagpu", type=Path, required=True)
    parser.add_argument("--orion", type=Path, required=True)
    parser.add_argument("--xsched", type=Path, required=True)
    parser.add_argument("--bless", type=Path, required=True)
    parser.add_argument("--pantheon", type=Path, required=True)
    parser.add_argument("--gpulet-numeric", type=Path, required=True)
    parser.add_argument("--payload-gate", type=Path)
    parser.add_argument("--orion-numeric", type=Path)
    parser.add_argument("--xsched-numeric", type=Path)
    parser.add_argument("--offered-rps", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(
        args.frontier.resolve(),
        args.boer.resolve(),
        args.parvagpu.resolve(),
        args.orion.resolve(),
        args.xsched.resolve(),
        args.bless.resolve(),
        args.pantheon.resolve(),
        args.gpulet_numeric.resolve(),
        args.offered_rps,
        args.payload_gate.resolve() if args.payload_gate is not None else None,
        args.orion_numeric.resolve() if args.orion_numeric is not None else None,
        args.xsched_numeric.resolve() if args.xsched_numeric is not None else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
