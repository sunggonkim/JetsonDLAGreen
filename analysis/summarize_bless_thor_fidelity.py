#!/usr/bin/env python3
"""Join the native-squad and TensorRT-context gates for the BLESS port."""

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
    native_path: Path,
    replica_path: Path,
    activation_path: Path,
    squad_path: Path,
) -> dict[str, Any]:
    native = load(native_path)
    replica = load(replica_path)
    activation = load(activation_path)
    squad = load(squad_path)
    if (
        native.get("kind") != "bless-thor-native-squad-functional-gate"
        or native.get("status") != "passed"
        or native.get("numeric_comparison_allowed") is not False
        or native.get("kernels") != 24
        or native.get("squads") != 4
        or native.get("estimators")
        != ["interference-free", "workload-equivalence"]
    ):
        raise ValueError("BLESS native-squad gate differs")
    if (
        replica.get("kind")
        != "bless-thor-trt-context-replica-functional-gate"
        or replica.get("status") != "passed"
        or replica.get("numeric_comparison_allowed") is not False
        or replica.get("affinity_domain_sms") != [2, 4, 6, 8]
        or replica.get("replica_rounds") != 2
        or replica.get("contexts_precreated_and_reused") is not True
        or replica.get("launches_per_replica", 0) <= 0
        or replica.get("replica_launch_sequences_identical") is not True
        or replica.get("driver_launch_records", 0) <= 0
        or replica.get("driver_api_counts", {}).get("cuLaunchKernelEx", 0) <= 0
    ):
        raise ValueError("BLESS TensorRT-context gate differs")
    if (
        activation.get("kind")
        != "bless-thor-trt-activation-replica-functional-gate"
        or activation.get("status") != "passed"
        or activation.get("numeric_comparison_allowed") is not False
        or activation.get("affinity_domain_sms") != [2, 4, 6, 8]
        or activation.get("activation_bytes", 0) <= 0
        or activation.get("restricted_to_unrestricted_peer_copy") is not True
        or activation.get("post_copy_inference_passed") is not True
        or activation.get("replica_launch_sequences_identical") is not True
        or activation.get("engine", {}).get("sha256")
        != replica.get("engine", {}).get("sha256")
    ):
        raise ValueError("BLESS TensorRT-activation gate differs")
    if (
        squad.get("kind") != "bless-thor-trt-squad-replica-functional-gate"
        or squad.get("status") != "passed"
        or squad.get("numeric_comparison_allowed") is not False
        or squad.get("logical_launches", 0) <= 0
        or squad.get("physical_launches") != squad.get("logical_launches")
        or squad.get("shadow_launches") != squad.get("logical_launches") * 3
        or squad.get("safe_switch_operation") != 23
        or squad.get("activation_copies") != 1
        or not isinstance(squad.get("boundary_lock", {}).get("sha256"), str)
        or squad.get("engine", {}).get("sha256")
        != replica.get("engine", {}).get("sha256")
    ):
        raise ValueError("BLESS TensorRT-squad gate differs")
    return {
        "schema_version": 1,
        "kind": "bless-thor-tensorrt-fidelity-gate",
        "status": "passed-functional-gates",
        "numeric_comparison_allowed": False,
        "completed_gates": [
            "relative-progress-kernel-squad-selection",
            "interference-free-and-workload-equivalence-estimators",
            "restricted-and-unrestricted-native-context-execution",
            "precreated-2-4-6-8-sm-tensorrt-context-replicas",
            "measured-tensorrt-driver-launches-in-every-replica",
            "user-managed-activation-peer-copy-restricted-to-unrestricted",
            "post-copy-tensorrt-output-correctness",
            "selected-only-logical-tensorrt-launch-admission",
            "shadow-replica-advancement",
            "safe-boundary-restricted-to-unrestricted-switch",
            "independent-safe-boundary-profile-and-held-out-replay",
        ],
        "remaining_gate": (
            "drive the frozen safe boundaries with BLESS relative-progress squad "
            "scheduling on the common workload"
        ),
        "native_squad": {
            "kernels": native["kernels"],
            "squads": native["squads"],
            "estimators": native["estimators"],
        },
        "tensorrt_replicas": {
            "affinity_domain_sms": replica["affinity_domain_sms"],
            "replica_rounds": replica["replica_rounds"],
            "driver_launch_records": replica["driver_launch_records"],
            "launches_per_replica": replica["launches_per_replica"],
            "replica_launch_sequences_identical": True,
            "driver_api_counts": replica["driver_api_counts"],
            "engine": replica["engine"],
        },
        "activation_replication": {
            "activation_bytes": activation["activation_bytes"],
            "restricted_to_unrestricted_peer_copy": True,
            "post_copy_inference_passed": True,
            "driver_launch_records": activation["driver_launch_records"],
            "launches_per_inference": activation["launches_per_inference"],
        },
        "logical_squad_admission": {
            "logical_launches": squad["logical_launches"],
            "physical_launches": squad["physical_launches"],
            "shadow_launches": squad["shadow_launches"],
            "safe_switch_operation": squad["safe_switch_operation"],
            "activation_copies": squad["activation_copies"],
            "output_checksum": squad["output_checksum"],
            "boundary_lock": squad["boundary_lock"],
        },
        "inputs": {
            "native_squad": {"path": str(native_path), "sha256": sha256(native_path)},
            "tensorrt_replicas": {
                "path": str(replica_path),
                "sha256": sha256(replica_path),
            },
            "activation_replication": {
                "path": str(activation_path),
                "sha256": sha256(activation_path),
            },
            "logical_squad_admission": {
                "path": str(squad_path),
                "sha256": sha256(squad_path),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--replicas", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--squad", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.native.resolve(),
        args.replicas.resolve(),
        args.activation.resolve(),
        args.squad.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
