#!/usr/bin/env python3
"""Summarize intended-domain positive controls for the published ports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(
    boer_path: Path,
    parvagpu_path: Path,
    orion_path: Path,
    xsched_path: Path,
    bless_path: Path,
) -> dict[str, Any]:
    boer = load(boer_path)
    selected = boer.get("selected")
    if (
        boer.get("system") != "BOER"
        or boer.get("status") != "selected"
        or boer.get("provenance", {}).get("fidelity")
        != "algorithm-preserving-thor-port"
        or not isinstance(selected, dict)
        or selected.get("metrics", {}).get("feasible") != 1.0
        or selected.get("metrics", {}).get("deadline_miss_rate") != 0.0
    ):
        raise ValueError("BOER positive control differs")

    parvagpu = load(parvagpu_path)
    services = parvagpu.get("services")
    if (
        parvagpu.get("system") != "ParvaGPU"
        or parvagpu.get("status") != "measured-smoke"
        or parvagpu.get("all_slos_met") is not True
        or not isinstance(services, list)
        or len(services) != 2
        or any(service.get("slo_met") is not True for service in services)
    ):
        raise ValueError("ParvaGPU positive control differs")

    orion = load(orion_path)
    if (
        orion.get("kind") != "orion-thor-profile-aware-verification"
        or orion.get("functional_gate_passed") is not True
        or orion.get("complementary_admissions", 0) <= 0
        or orion.get("decisions", 0) <= 0
    ):
        raise ValueError("Orion positive control differs")

    xsched = load(xsched_path)
    if (
        xsched.get("kind") != "xsched-thor-native-positive-control"
        or xsched.get("be_completed_requests", 0) <= 0
        or xsched.get("hp_completed_requests", 0) <= 0
        or xsched.get("be_suspend_transitions", 0) <= 0
        or xsched.get("be_resume_transitions", 0) <= 0
        or xsched.get("measurement_overlap_ns", 0) <= 0
    ):
        raise ValueError("XSched positive control differs")

    bless = load(bless_path)
    if (
        bless.get("kind") != "bless-thor-tensorrt-fidelity-gate"
        or bless.get("status") != "passed-functional-gates"
        or bless.get("native_squad", {}).get("kernels") != 24
        or bless.get("native_squad", {}).get("estimators")
        != ["interference-free", "workload-equivalence"]
        or bless.get("tensorrt_replicas", {}).get("driver_launch_records", 0) <= 0
        or bless.get("activation_replication", {}).get("post_copy_inference_passed")
        is not True
    ):
        raise ValueError("BLESS positive control differs")

    parva_worst = max(float(service["p99_ms"]) for service in services)
    parva_goodput = sum(float(service["served_rps"]) for service in services)
    return {
        "schema_version": 1,
        "kind": "p9-published-system-positive-controls",
        "scope": "intended-domain-functional-and-smoke-validation",
        "interpretation": (
            "These controls establish that a dependent-workload failure is not "
            "explained solely by a dead port. They are not cross-system rankings."
        ),
        "rows": [
            {
                "system": "BOER (Thor port)",
                "status": "passed-independent-numeric-smoke",
                "evidence": "Bayesian search selected a feasible configuration",
                "p99_ms": float(selected["metrics"]["worst_p99_ms"]),
                "deadline_miss_rate": 0.0,
                "aggregate_goodput_rps": (
                    float(selected["metrics"]["served_rps_0"])
                    + float(selected["metrics"]["served_rps_1"])
                ),
            },
            {
                "system": "ParvaGPU (Thor port)",
                "status": "passed-independent-numeric-smoke",
                "evidence": "planned 2g+1g allocation served both services",
                "p99_ms": parva_worst,
                "deadline_miss_rate": 0.0,
                "aggregate_goodput_rps": parva_goodput,
            },
            {
                "system": "Orion (Thor port)",
                "status": "passed-profile-aware-functional-gate",
                "evidence": "real TensorRT operations received complementary admissions",
                "decisions": orion["decisions"],
                "complementary_admissions": orion["complementary_admissions"],
            },
            {
                "system": "XSched (Thor port)",
                "status": "passed-native-functional-gate",
                "evidence": "overlapping XQueues suspended and resumed BE work",
                "be_completed_requests": xsched["be_completed_requests"],
                "hp_completed_requests": xsched["hp_completed_requests"],
                "suspend_transitions": xsched["be_suspend_transitions"],
                "resume_transitions": xsched["be_resume_transitions"],
            },
            {
                "system": "BLESS (Thor reimplementation)",
                "status": "passed-native-and-tensorrt-functional-gates",
                "evidence": (
                    "ordered native squads plus selected-only TensorRT logical launches "
                    "in reused affinity contexts"
                ),
                "kernels": bless["native_squad"]["kernels"],
                "squads": bless["native_squad"]["squads"],
                "driver_launch_records": bless["tensorrt_replicas"]["driver_launch_records"],
                "activation_bytes": bless["activation_replication"]["activation_bytes"],
                "logical_launches": bless["logical_squad_admission"]["logical_launches"],
                "physical_launches": bless["logical_squad_admission"]["physical_launches"],
                "shadow_launches": bless["logical_squad_admission"]["shadow_launches"],
            },
        ],
        "input_sha256": {
            "boer": sha256(boer_path),
            "parvagpu": sha256(parvagpu_path),
            "orion": sha256(orion_path),
            "xsched": sha256(xsched_path),
            "bless": sha256(bless_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("boer", "parvagpu", "orion", "xsched", "bless"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.boer.resolve(), args.parvagpu.resolve(), args.orion.resolve(),
        args.xsched.resolve(), args.bless.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
