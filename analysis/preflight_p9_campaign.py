#!/usr/bin/env python3
"""Fail-fast preflight for the QUIET common-workload campaign.

This command never creates workload evidence and never substitutes a synthetic
trace.  It reports exactly which external inputs are missing or stale before a
long hardware run is started.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    raw = path.resolve().read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("JSON file is not newline-complete")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def bound_file(record: Any, label: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"{label} is missing")
    path_value, expected = record.get("path"), record.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str):
        raise ValueError(f"{label} lacks path/sha256")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise ValueError(f"{label} file does not exist: {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} sha256 differs")
    return {"path": str(path), "sha256": actual}


def verify_accuracy_gate(
    path: Path, *, common_workload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the byte-bound accuracy gate before formal preflight passes.

    Merely supplying a JSON file named ``accuracy-gate`` is not evidence.  The
    gate must be the output of the strict verifier and all request/output
    evidence referenced by it must still hash to the recorded bytes.
    """
    resolved = path.resolve()
    raw = resolved.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("accuracy gate is not newline-complete")
    try:
        gate = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("accuracy gate is invalid JSON") from error
    if not isinstance(gate, dict):
        raise ValueError("accuracy gate must be a JSON object")
    if (
        gate.get("kind") != "p9-application-accuracy-gate"
        or gate.get("status") != "passed"
        or gate.get("numeric_comparison_allowed") is not True
        or gate.get("application_output_trace_required") is not True
        or gate.get("application_output_trace_contract") != "passed"
        or gate.get("application_input_binding_required") is not True
        or gate.get("application_input_binding_contract") != "passed"
    ):
        raise ValueError("accuracy gate is not a passed formal output/input-bound gate")
    requests = gate.get("requests")
    if isinstance(requests, bool) or not isinstance(requests, int) or requests <= 0:
        raise ValueError("accuracy gate request count is invalid")
    tolerance = gate.get("accuracy_tolerance")
    delta = gate.get("accuracy_delta")
    minimum = gate.get("minimum_accuracy")
    reference_accuracy = gate.get("reference_accuracy")
    candidate_accuracy = gate.get("candidate_accuracy")
    if (
        isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
        or isinstance(delta, bool) or not isinstance(delta, (int, float))
        or not math.isfinite(float(tolerance)) or not math.isfinite(float(delta))
        or float(tolerance) < 0.0 or float(delta) < 0.0
        or float(delta) > float(tolerance)
        or isinstance(minimum, bool) or not isinstance(minimum, (int, float))
        or not math.isfinite(float(minimum)) or not 0.0 <= float(minimum) <= 1.0
        or isinstance(reference_accuracy, bool)
        or not isinstance(reference_accuracy, (int, float))
        or not math.isfinite(float(reference_accuracy))
        or isinstance(candidate_accuracy, bool)
        or not isinstance(candidate_accuracy, (int, float))
        or not math.isfinite(float(candidate_accuracy))
        or float(reference_accuracy) < float(minimum)
        or float(candidate_accuracy) < float(minimum)
    ):
        raise ValueError("accuracy gate tolerance/delta/minimum is invalid")

    def verify_bound(path_key: str, digest_key: str, label: str) -> dict[str, Any]:
        record = {"path": gate.get(path_key), "sha256": gate.get(digest_key)}
        return bound_file(record, label)

    bound = {
        "dataset_manifest": verify_bound(
            "dataset_manifest_path", "dataset_manifest_sha256", "accuracy dataset manifest"
        ),
        "reference_trace": verify_bound(
            "reference_trace_path", "reference_trace_sha256", "accuracy reference trace"
        ),
        "candidate_trace": verify_bound(
            "candidate_trace_path", "candidate_trace_sha256", "accuracy candidate trace"
        ),
    }
    for prefix, label in (("reference", "reference"), ("candidate", "candidate")):
        record = gate.get(f"{prefix}_pipeline_csv")
        if not isinstance(record, dict):
            raise ValueError(f"accuracy {label} pipeline CSV is missing")
        bound[f"{prefix}_pipeline_csv"] = bound_file(
            record, f"accuracy {label} pipeline CSV"
        )
    for prefix, label in (("reference", "reference"), ("candidate", "candidate")):
        record = gate.get(f"{prefix}_output_trace")
        if not isinstance(record, dict):
            raise ValueError(f"accuracy {label} output trace is missing")
        output = bound_file(record, f"accuracy {label} output trace")
        if record.get("capture_boundary") != "post-completion":
            raise ValueError(f"accuracy {label} output trace is not post-completion")
        count = record.get("record_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < requests:
            raise ValueError(f"accuracy {label} output trace count is invalid")
        bound[f"{prefix}_output_trace"] = output
    if common_workload is not None:
        workload_id = common_workload.get("workload_id")
        request_count = common_workload.get("request_count")
        common_dataset = common_workload.get("dataset_manifest")
        if not isinstance(workload_id, str) or not workload_id:
            raise ValueError("common workload lacks workload_id")
        if isinstance(request_count, bool) or not isinstance(request_count, int) or request_count <= 0:
            raise ValueError("common workload request_count is invalid")
        if not isinstance(common_dataset, dict):
            raise ValueError("common workload lacks dataset manifest binding")
        if gate.get("workload") != workload_id:
            raise ValueError("accuracy gate workload differs from common workload")
        if requests != request_count:
            raise ValueError("accuracy gate request count differs from common workload")
        if bound["dataset_manifest"] != common_dataset:
            raise ValueError("accuracy gate dataset manifest differs from common workload")
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "requests": requests,
        "accuracy_delta": float(delta),
        "accuracy_tolerance": float(tolerance),
        "evidence": bound,
    }


def check_common_contract(path: Path) -> dict[str, Any]:
    contract = load_json(path)
    required = {
        "schema_version", "workload_id", "topology", "placement", "input_tensor",
        "payload_bytes", "request_count", "arrival_trace_path",
        "arrival_trace_sha256", "dataset_manifest_path", "dataset_manifest_sha256",
    }
    if contract.get("schema_version") != 1 or not required <= set(contract):
        raise ValueError("common workload contract schema differs")
    arrival = bound_file({"path": contract["arrival_trace_path"],
                          "sha256": contract["arrival_trace_sha256"]}, "arrival trace")
    dataset = bound_file({"path": contract["dataset_manifest_path"],
                          "sha256": contract["dataset_manifest_sha256"]}, "dataset manifest")
    if contract["request_count"] <= 0:
        raise ValueError("common workload request_count must be positive")
    return {
        "path": str(path.resolve()), "sha256": sha256(path),
        "workload_id": contract["workload_id"], "request_count": contract["request_count"],
        "topology": contract["topology"], "placement": contract["placement"],
        "arrival_trace": arrival, "dataset_manifest": dataset,
    }


def verify_thermal_lock(path: Path) -> dict[str, Any]:
    lock = load_json(path)
    module_path = ROOT / "analysis/freeze_p9_thermal.py"
    spec = importlib.util.spec_from_file_location("p9_freeze_thermal_preflight", module_path)
    if spec is None or spec.loader is None:
        raise ValueError("thermal freezer is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_lock(lock)
    return {"path": str(path.resolve()), "sha256": sha256(path),
            "schema_version": lock.get("schema_version"),
            "stability_sensor": lock.get("stability_sensor"),
            "safety_sensor": lock.get("safety_sensor")}


def verify_deadline_lock(path: Path) -> dict[str, Any]:
    module_path = ROOT / "analysis/freeze_p9_deadline.py"
    spec = importlib.util.spec_from_file_location("p9_freeze_deadline_preflight", module_path)
    if spec is None or spec.loader is None:
        raise ValueError("deadline freezer is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lock = load_json(path)
    module.verify_lock(lock)
    return {"path": str(path.resolve()), "sha256": sha256(path),
            "schema_version": lock.get("schema_version"),
            "deadline_ms": lock.get("deadline_ms")}


def check_hardware() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia-smi is not installed"}
    completed = subprocess.run([executable, "-L"], capture_output=True, text=True, check=False)
    text = completed.stdout + completed.stderr
    mig = [line.strip() for line in text.splitlines() if "MIG" in line]
    return {"available": completed.returncode == 0, "mig_instances": mig,
            "raw_sha256": hashlib.sha256(text.encode()).hexdigest()}


def check_mig_env(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"MIG environment is missing: {path}")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    required = ("JDG_MIG_SMALL_UUID", "JDG_MIG_BIG_UUID", "JDG_MPS_PIPE_DIRECTORY")
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise ValueError(f"MIG environment lacks {','.join(missing)}")
    pipe = Path(values["JDG_MPS_PIPE_DIRECTORY"])
    if not pipe.is_dir():
        raise ValueError(f"MPS pipe directory is missing: {pipe}")
    ps = shutil.which("ps")
    process_listing = subprocess.run(
        [ps or "ps", "-eo", "args"], capture_output=True, text=True, check=False
    ).stdout
    server_present = any(
        line.strip().startswith("nvidia-cuda-mps-server")
        or " nvidia-cuda-mps-server " in f" {line.strip()} "
        for line in process_listing.splitlines()
    )
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "small_uuid": values["JDG_MIG_SMALL_UUID"],
        "big_uuid": values["JDG_MIG_BIG_UUID"],
        "pipe_directory": str(pipe.resolve()),
        "mps_server_present": server_present,
    }


def preflight(*, common: Path, thermal: Path, deadline: Path | None,
              accuracy: Path | None,
              mig_env: Path = Path("/tmp/jdg-mps-1g/mig.env"),
              check_gpu: bool = True) -> dict[str, Any]:
    missing: list[str] = []
    warnings: list[str] = []
    artifacts: dict[str, Any] = {}
    common_ok = True
    mig_ok = True
    try:
        artifacts["common_workload"] = check_common_contract(common)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        common_ok = False
        missing.append(f"common workload: {error}")
    try:
        artifacts["mig_env"] = check_mig_env(mig_env)
    except (OSError, ValueError) as error:
        mig_ok = False
        missing.append(f"MIG environment: {error}")
    manifest = ROOT / "results/p9-real-resnet-head-artifacts-20260810/manifest.json"
    for label, path in (("learned artifact manifest", manifest),
                        ("producer engine", ROOT / "models/engines/mig-1g-q100/resnet10-backbone.engine"),
                        ("consumer engine", ROOT / "results/p9-real-resnet-head-artifacts-20260810/resnet10-detection-head.engine")):
        if not path.is_file():
            missing.append(f"{label} is missing: {path}")
        else:
            artifacts[label.replace(" ", "_")] = {"path": str(path.resolve()), "sha256": sha256(path)}
    if accuracy is None:
        missing.append("application accuracy gate manifest was not supplied")
    elif not accuracy.is_file():
        missing.append(f"application accuracy gate is missing: {accuracy}")
    else:
        try:
            artifacts["accuracy_gate"] = verify_accuracy_gate(
                accuracy,
                common_workload=artifacts.get("common_workload"),
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            missing.append(f"application accuracy gate: {error}")
    try:
        artifacts["thermal_lock"] = verify_thermal_lock(thermal)
    except (OSError, ValueError, json.JSONDecodeError, ImportError) as error:
        missing.append(f"thermal lock: {error}")
    if deadline is None:
        missing.append("deadline lock was not supplied")
    else:
        try:
            artifacts["deadline_lock"] = verify_deadline_lock(deadline)
        except (OSError, ValueError, json.JSONDecodeError, ImportError) as error:
            missing.append(f"deadline lock: {error}")
    hardware = check_hardware() if check_gpu else {"available": None, "skipped": True}
    if hardware.get("available") is False:
        warnings.append(hardware.get("reason", "GPU query failed"))
    engine_ok = all(key in artifacts for key in ("producer_engine", "consumer_engine"))
    hardware_ok = hardware.get("available", True) is not False
    if not common_ok:
        warnings.append("common workload is absent; exploratory legacy smoke may run, but no numeric promotion is allowed")
    if mig_ok and artifacts["mig_env"].get("mps_server_present") is False:
        warnings.append("nvidia-cuda-mps-server is not running; start it before the learned-head fast pair")
    if accuracy is None or not (accuracy and accuracy.is_file()):
        warnings.append("application accuracy is absent; all learned-DAG results remain exploratory")
    return {
        "schema_version": 1,
        "kind": "p9-campaign-preflight",
        "proposed_system": "QUIET",
        "exploratory_ready": engine_ok and hardware_ok and mig_ok,
        "formal_ready": not missing and hardware_ok,
        "missing": missing,
        "warnings": warnings,
        "artifacts": artifacts,
        "hardware": hardware,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-workload-contract", type=Path,
                        default=ROOT / "results/common-workload.json")
    parser.add_argument("--mig-env", type=Path,
                        default=Path("/tmp/jdg-mps-1g/mig.env"))
    parser.add_argument("--thermal-lock", type=Path, required=True)
    parser.add_argument("--deadline-lock", type=Path)
    parser.add_argument("--accuracy-gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-gpu-check", action="store_true")
    args = parser.parse_args()
    result = preflight(common=args.common_workload_contract, thermal=args.thermal_lock,
                       deadline=args.deadline_lock, accuracy=args.accuracy_gate,
                       mig_env=args.mig_env,
                       check_gpu=not args.skip_gpu_check)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["formal_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
