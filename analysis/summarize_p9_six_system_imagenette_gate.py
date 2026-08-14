#!/usr/bin/env python3
"""Build the fixed-roster, single-session ImageNette comparator gate.

The gate exists to make comparator coverage explicit.  Every displayed row
uses the same 90 labelled inputs, operational arrival trace, current runtime
lock, and 2,224.448-us deadline.  It is deliberately not a formal ranking:
the run has one directional session, the fixed-stage MIG row is a no-BE
capacity endpoint, and Orion and Pantheon retain their native-port
timing/fidelity scopes.  A separate partial control co-locates the critical
DAG on 2g and reserves 1g for BE; it never enters the fixed six-system rank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ORDER = (
    "QUIET",
    "NVIDIA MIG",
    "NVIDIA MPS",
    "XSched",
    "Orion",
    "Pantheon",
)
REQUESTS = 90
PRODUCTION_WALL = "arrival-to-consumer-completion-excludes-correctness-validation"
COMMON_KEYS = (
    "schema_version",
    "workload_id",
    "topology",
    "placement",
    "input_tensor",
    "payload_bytes",
    "request_count",
    "arrival_trace_sha256",
    "dataset_manifest_sha256",
    "producer_input_trace_sha256",
    "operational_arrival_trace_sha256",
)
DEFAULT_COMMON = ROOT / "results/p9-resnet50-imagenette-gate100-20260811/common-workload.json"
DEFAULT_LOCK = ROOT / "results/p9-resnet50-imagenette-calibration-current-r03-20260811/deadline-lock.json"
DEFAULT_QUIET = ROOT / "results/p9-active-resnet50-imagenette-frontier-r07-20260811/sequence-0/03-quiet/summary.json"
DEFAULT_MIG = ROOT / "results/p9-resnet50-imagenette-six-system-gate90-20260814/nvidia-mig/summary.json"
DEFAULT_MIG_COLOCATED = ROOT / "results/p9-mig-colocated-imagenette-gate90-20260814-r06/summary.json"
DEFAULT_MPS = ROOT / "results/p9-active-resnet50-imagenette-frontier-r07-20260811/sequence-0/01-nvidia-mps/summary.json"
DEFAULT_XSCHED = ROOT / "results/p9-active-resnet50-imagenette-frontier-r07-20260811/sequence-0/02-xsched/verification.json"
DEFAULT_ORION = ROOT / "results/p9-resnet50-imagenette-six-system-gate90-20260814/orion-r02/verification.json"
DEFAULT_PANTHEON = ROOT / "results/p9-pantheon-resnet50-imagenette-gate100-r01-20260811/verification.json"
DEFAULT_OUTPUT = ROOT / "paper/eurosys27/generated/p9-six-system-imagenette-gate.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _resolve(raw: str) -> Path:
    path = Path(raw)
    if path.is_file():
        return path.resolve()
    if not path.is_absolute() and (ROOT / path).is_file():
        return (ROOT / path).resolve()
    for marker in ("results", "models", "paper", "docs", "baselines"):
        if marker in path.parts:
            candidate = ROOT.joinpath(*path.parts[path.parts.index(marker):])
            if candidate.is_file():
                return candidate.resolve()
    raise ValueError(f"recorded evidence is missing: {raw}")


def _read(path: Path, label: str) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{label} is not newline-complete: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _record(record: Any, label: str) -> dict[str, str]:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError(f"{label} path record is missing")
    expected = record.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} SHA-256 is missing")
    path = _resolve(record["path"])
    if sha256(path) != expected:
        raise ValueError(f"{label} SHA-256 differs")
    return {"path": _portable(path), "sha256": expected}


def _cp95(misses: int, requests: int) -> float:
    if misses == requests:
        return 1.0
    return float(beta.ppf(0.95, misses + 1, requests - misses))


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _common(path: Path) -> tuple[dict[str, Any], str]:
    value, digest = _read(path, "common workload")
    if (
        value.get("schema_version") != 1
        or value.get("workload_id") != "resnet50-classification"
        or value.get("topology") != "fixed-2g+1g"
        or value.get("placement") != "fixed-1g-producer-2g-consumer"
        or value.get("input_tensor") != "gpu_0/res4_5_branch2c_bn_2"
        or value.get("payload_bytes") != 802816
        or value.get("request_count") != REQUESTS
    ):
        raise ValueError("common ImageNette contract differs")
    for path_key, sha_key in (
        ("arrival_trace_path", "arrival_trace_sha256"),
        ("dataset_manifest_path", "dataset_manifest_sha256"),
        ("producer_input_trace_path", "producer_input_trace_sha256"),
        ("operational_arrival_trace_path", "operational_arrival_trace_sha256"),
    ):
        evidence = _resolve(str(value.get(path_key, "")))
        if sha256(evidence) != value.get(sha_key):
            raise ValueError(f"common workload {path_key} SHA-256 differs")
    return value, digest


def _check_common(value: Any, common: dict[str, Any], digest: str, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} common workload is missing")
    if "workload_id" in value:
        if any(value.get(key) != common.get(key) for key in COMMON_KEYS):
            raise ValueError(f"{label} common workload differs")
        declared = value.get("contract_sha256")
        if declared is not None and declared != digest:
            raise ValueError(f"{label} common workload contract SHA-256 differs")
        return
    path_record = {"path": value.get("path"), "sha256": value.get("sha256")}
    checked = _record(path_record, f"{label} common workload")
    if checked["sha256"] != digest:
        raise ValueError(f"{label} common workload SHA-256 differs")
    if (
        value.get("input_trace_sha256") != common["producer_input_trace_sha256"]
        or value.get("arrival_trace_sha256") != common["operational_arrival_trace_sha256"]
        or value.get("dataset_manifest_sha256") != common["dataset_manifest_sha256"]
    ):
        raise ValueError(f"{label} common trace bindings differ")


def _deadline(path: Path) -> tuple[dict[str, Any], str, float]:
    value, digest = _read(path, "deadline lock")
    deadline = _finite(value.get("deadline_us"), "deadline")
    if (
        value.get("kind") != "p9-dependent-pipeline-deadline-lock"
        or value.get("contract", {}).get("workload") != "resnet50-classification"
        or deadline <= 0.0
    ):
        raise ValueError("deadline lock contract differs")
    for label in ("binary", "source", "engine", "consumer_engine"):
        _record(value.get("artifacts", {}).get(label), f"deadline {label}")
    return value, digest, deadline


def _check_deadline_record(record: Any, digest: str, label: str) -> None:
    checked = _record(record, f"{label} deadline lock")
    if checked["sha256"] != digest:
        raise ValueError(f"{label} deadline lock SHA-256 differs")


def _accuracy(
    path: Path,
    common: dict[str, Any],
    deadline: float,
    label: str,
    *,
    maximum_delta: float = 0.0,
) -> dict[str, Any]:
    value, digest = _read(path, f"{label} accuracy gate")
    recorded_tolerance = _finite(value.get("accuracy_tolerance"), label)
    accuracy_delta = _finite(value.get("accuracy_delta"), label)
    if (
        value.get("kind") != "p9-application-accuracy-gate"
        or value.get("status") != "passed"
        or value.get("numeric_comparison_allowed") is not True
        or value.get("workload") != "resnet50-classification"
        or value.get("requests") != REQUESTS
        or not math.isclose(_finite(value.get("deadline_us"), label), deadline, abs_tol=1e-9)
        or value.get("application_input_binding_contract") != "passed"
        or value.get("application_output_trace_contract") != "passed"
        or value.get("dataset_manifest_sha256") != common["dataset_manifest_sha256"]
        or _finite(value.get("reference_accuracy"), label) < 0.8
        or _finite(value.get("candidate_accuracy"), label) < 0.8
        or recorded_tolerance > maximum_delta + 1e-12
        or accuracy_delta > maximum_delta + 1e-12
    ):
        raise ValueError(f"{label} accuracy gate differs")
    return {
        "reference": float(value["reference_accuracy"]),
        "candidate": float(value["candidate_accuracy"]),
        "delta": accuracy_delta,
        "tolerance": recorded_tolerance,
        "gate": {"path": _portable(path), "sha256": digest},
    }


def _metric_row(
    *, system: str, requests: int, misses: int, p99_us: float, goodput: float,
    accuracy: dict[str, Any], source: Path, source_sha: str, topology: str,
    evidence_scope: str, timing_scope: str,
    background_goodput_applicable: bool = True,
) -> dict[str, Any]:
    if requests != REQUESTS or not 0 <= misses <= requests or p99_us <= 0.0 or goodput < 0.0:
        raise ValueError(f"{system} metric values are invalid")
    return {
        "system": system,
        "requests": requests,
        "misses": misses,
        "observed_dmr": misses / requests,
        "dmr_cp95_upper": _cp95(misses, requests),
        "p99_us": p99_us,
        "background_goodput_rps": goodput,
        "background_goodput_applicable": background_goodput_applicable,
        "reference_accuracy": accuracy["reference"],
        "candidate_accuracy": accuracy["candidate"],
        "accuracy_delta": accuracy["delta"],
        "topology": topology,
        "evidence_scope": evidence_scope,
        "timing_scope": timing_scope,
        "observed_slo_feasible": misses == 0,
        "confidence_qualified": False,
        "source": {"path": _portable(source), "sha256": source_sha},
        "accuracy_gate": accuracy["gate"],
    }


def _pipeline_row(
    system: str, path: Path, common: dict[str, Any], common_sha: str,
    lock_sha: str, deadline: float,
) -> dict[str, Any]:
    value, source_sha = _read(path, f"{system} evidence")
    rows = value.get("results")
    if (
        value.get("kind") != "p9-dependent-small-stress-smoke"
        or value.get("workload") != "resnet50-classification"
        or not isinstance(rows, list) or len(rows) != 1
        or rows[0].get("system") != system
        or not math.isclose(_finite(value.get("deadline_us"), system), deadline, abs_tol=1e-9)
    ):
        raise ValueError(f"{system} evidence contract differs")
    _check_common(value.get("common_workload"), common, common_sha, system)
    _check_deadline_record(value.get("deadline_lock"), lock_sha, system)
    row = rows[0]
    if (
        row.get("deadline_mode") != "wall"
        or row.get("latency_contract") != "production-wall-arrival-to-completion"
        or row.get("production_wall_definition") != PRODUCTION_WALL
        or row.get("correctness_validation_placement") != "post-completion"
        or row.get("correctness_validated") is not True
    ):
        raise ValueError(f"{system} production-wall contract differs")
    _record(row.get("request_trace"), f"{system} request trace")
    _record(row.get("application_output_trace"), f"{system} output trace")
    if system == "NVIDIA MIG" and (
        row.get("best_effort_admitted") is not False
        or row.get("best_effort_status") != "rejected-no-best-effort-slice"
        or _finite(row.get("background_goodput_rps"), system) != 0.0
    ):
        raise ValueError("NVIDIA MIG is not the pure capacity endpoint")
    accuracy_path = path.parent / "application-accuracy/accuracy-gate.json"
    accuracy = _accuracy(accuracy_path, common, deadline, system)
    scopes = {
        "QUIET": ("fixed-2g+1g-dependent-dag-quiescence", "proposed-system-common-gate"),
        "NVIDIA MIG": ("fixed-2g+1g-physical-isolation", "capacity-endpoint-no-best-effort-slice"),
        "NVIDIA MPS": ("1g-shared-MPS-with-2g-reserved", "vendor-baseline-common-gate"),
    }
    topology, evidence_scope = scopes[system]
    return _metric_row(
        system=system,
        requests=int(row["pipeline_requests"]),
        misses=int(row["deadline_misses"]),
        p99_us=_finite(row["pipeline_p99_us"], system),
        goodput=_finite(row["background_goodput_rps"], system),
        accuracy=accuracy,
        source=path,
        source_sha=source_sha,
        topology=topology,
        evidence_scope=evidence_scope,
        timing_scope="arrival-to-consumer-completion",
        background_goodput_applicable=system != "NVIDIA MIG",
    )


def _plain_evidence(path: Path, label: str) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file() or not path.read_bytes().endswith(b"\n"):
        raise ValueError(f"{label} is missing or not newline-complete")
    return {"path": _portable(path), "sha256": sha256(path)}


def _colocated_mig_row(
    path: Path,
    common: dict[str, Any],
    common_sha: str,
    lock_sha: str,
    deadline: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    label = "NVIDIA MIG (2g DAG + 1g BE)"
    value, source_sha = _read(path, "colocated MIG evidence")
    rows = value.get("results")
    if (
        value.get("kind") != "p9-dependent-small-stress-smoke"
        or value.get("claim_status") != "partial-topology-comparator"
        or value.get("deadline_source")
        != "replayed-common-gate-slo-topology-variant"
        or value.get("deadline_lock") is not None
        or value.get("workload") != "resnet50-classification"
        or not isinstance(rows, list)
        or len(rows) != 1
        or rows[0].get("system") != label
        or not math.isclose(
            _finite(value.get("deadline_us"), label), deadline, abs_tol=1e-9
        )
    ):
        raise ValueError("colocated MIG evidence contract differs")
    _check_common(value.get("common_workload"), common, common_sha, label)
    _check_deadline_record(value.get("deadline_reference"), lock_sha, label)
    row = rows[0]
    if (
        row.get("deadline_mode") != "wall"
        or row.get("latency_contract")
        != "production-wall-arrival-to-completion"
        or row.get("production_wall_definition") != PRODUCTION_WALL
        or row.get("correctness_validation_placement") != "post-completion"
        or row.get("correctness_validated") is not True
        or row.get("best_effort_admitted") is not True
        or row.get("best_effort_status") != "completed"
        or row.get("comparison_scope") != "partial-topology-variant"
        or row.get("placement_variant")
        != "colocated-2g-critical-dag-1g-best-effort"
        or row.get("workload_contract_placement")
        != "fixed-1g-producer-2g-consumer"
        or row.get("producer_uuid") != row.get("consumer_uuid")
        or row.get("producer_sms") != 12
        or row.get("consumer_sms") != 12
    ):
        raise ValueError("colocated MIG topology or runtime contract differs")
    _record(row.get("request_trace"), "colocated MIG request trace")
    _record(row.get("application_output_trace"), "colocated MIG output trace")
    accuracy_path = path.parent / "application-accuracy/accuracy-gate.json"
    accuracy = _accuracy(
        accuracy_path,
        common,
        deadline,
        label,
        maximum_delta=0.02,
    )
    if accuracy["candidate"] < accuracy["reference"]:
        raise ValueError("colocated MIG loses application accuracy")

    placements_path = path.parent / "mig-possible-placements.txt"
    active_path = path.parent / "active-mig-instances.txt"
    placements = placements_path.read_text(encoding="utf-8")
    active = active_path.read_text(encoding="utf-8")
    required_placements = (
        "Profile ID 78 Placement : {2}:1",
        "Profile ID 80 Placement : {2}:1",
        "Profile ID 81 Placement : {2}:1",
        "Profile ID 82 Placement : {0}:2",
        "Profile ID 83 Placement : {0}:2",
    )
    if any(item not in placements for item in required_placements):
        raise ValueError("Thor MIG placement evidence differs")
    if (
        "MIG 1g.0gb+me" not in active
        or "MIG 2g.0gb+gfx" not in active
        or active.count("Placement") != 1
    ):
        raise ValueError("active Thor MIG topology differs")
    constraints = {
        "maximum_simultaneous_instances": 2,
        "three_way_1g_supported": False,
        "reason": (
            "all 1g profiles have the sole placement {2}:1 while all 2g "
            "profiles occupy {0}:2"
        ),
        "possible_placements": _plain_evidence(
            placements_path, "Thor MIG possible placements"
        ),
        "active_instances": _plain_evidence(
            active_path, "Thor active MIG instances"
        ),
    }
    metric = _metric_row(
        system=label,
        requests=int(row["pipeline_requests"]),
        misses=int(row["deadline_misses"]),
        p99_us=_finite(row["pipeline_p99_us"], label),
        goodput=_finite(row["background_goodput_rps"], label),
        accuracy=accuracy,
        source=path,
        source_sha=source_sha,
        topology="colocated-2g-critical-dag-with-isolated-1g-best-effort",
        evidence_scope="partial-topology-variant-same-input-arrival-deadline",
        timing_scope="arrival-to-consumer-completion",
    )
    return metric, constraints


def _xsched_row(
    path: Path, common: dict[str, Any], common_sha: str, lock_sha: str,
    deadline: float,
) -> dict[str, Any]:
    value, source_sha = _read(path, "XSched evidence")
    if (
        value.get("kind") != "xsched-dependent-resnet50-classification-numeric-smoke-verification"
        or value.get("system") != "XSched (Thor port)"
        or value.get("workload") != "resnet50-classification"
        or value.get("deadline_mode") != "wall"
        or value.get("latency_contract") != "production-wall-arrival-to-completion"
        or value.get("production_wall_definition") != PRODUCTION_WALL
        or value.get("correctness_validation_placement") != "post-completion"
        or value.get("inputs", {}).get("deadline_lock_sha256") != lock_sha
        or not math.isclose(_finite(value.get("deadline_us"), "XSched"), deadline, abs_tol=1e-9)
    ):
        raise ValueError("XSched evidence contract differs")
    _check_common(value.get("common_workload"), common, common_sha, "XSched")
    pipeline = path.parent / "pipeline.csv"
    if sha256(pipeline) != value["inputs"].get("pipeline_sha256"):
        raise ValueError("XSched pipeline SHA-256 differs")
    _record(value.get("application_output_trace"), "XSched output trace")
    accuracy = _accuracy(path.parent / "application-accuracy/accuracy-gate.json", common, deadline, "XSched")
    goodput = value.get("background_goodput_rps")
    if goodput is None:
        goodput = value.get("background_window", {}).get("completion_goodput_rps")
    return _metric_row(
        system="XSched", requests=int(value["requests"]), misses=int(value["misses"]),
        p99_us=_finite(value["p99_us"], "XSched"), goodput=_finite(goodput, "XSched"),
        accuracy=accuracy, source=path, source_sha=source_sha,
        topology="fixed-2g+1g-native-xqueue-port",
        evidence_scope="published-native-runtime-common-gate",
        timing_scope="arrival-to-consumer-completion",
    )


def _orion_row(
    path: Path, common: dict[str, Any], common_sha: str, lock_sha: str,
    deadline: float,
) -> dict[str, Any]:
    value, source_sha = _read(path, "Orion evidence")
    if (
        value.get("kind") != "orion-dependent-resnet50-imagenette-numeric-smoke-verification"
        or value.get("system") != "Orion (Thor port)"
        or value.get("functional_gate_passed") is not True
        or value.get("numeric_smoke_valid") is not True
        or value.get("formal_claim_allowed") is not False
        or "upstream-differential-fidelity" not in value.get("formal_blockers", [])
        or value.get("inputs", {}).get("deadline_lock_sha256") != lock_sha
        or not math.isclose(_finite(value.get("deadline_us"), "Orion"), deadline, abs_tol=1e-9)
    ):
        raise ValueError("Orion evidence contract differs")
    _check_common(value.get("common_workload"), common, common_sha, "Orion")
    pipeline = path.parent / "pipeline.csv"
    if sha256(pipeline) != value["inputs"].get("pipeline_sha256"):
        raise ValueError("Orion pipeline SHA-256 differs")
    _record(value.get("application_output_trace"), "Orion output trace")
    accuracy = _accuracy(path.parent / "accuracy-gate.json", common, deadline, "Orion")
    return _metric_row(
        system="Orion", requests=int(value["requests"]), misses=int(value["misses"]),
        p99_us=_finite(value["p99_us"], "Orion"),
        goodput=_finite(value["background_goodput_rps"], "Orion"),
        accuracy=accuracy, source=path, source_sha=source_sha,
        topology="fixed-2g+1g-native-interposition-port",
        evidence_scope="published-port-common-gate-differential-fidelity-open",
        timing_scope="arrival-to-consumer-completion",
    )


def _pantheon_row(
    path: Path, common: dict[str, Any], common_sha: str, lock_sha: str,
    deadline: float,
) -> dict[str, Any]:
    value, source_sha = _read(path, "Pantheon evidence")
    if (
        value.get("kind") != "pantheon-resnet50-imagenette-common-workload-fidelity-gate"
        or value.get("system") != "Pantheon (Thor port)"
        or value.get("status") != "passed"
        or value.get("numeric_comparison_allowed") is not True
        or not math.isclose(_finite(value.get("deadline_us"), "Pantheon"), deadline, abs_tol=1e-9)
        or value.get("effective_pantheon_deadline_us") != math.floor(deadline)
    ):
        raise ValueError("Pantheon evidence contract differs")
    _check_common(value.get("common_workload"), common, common_sha, "Pantheon")
    _check_deadline_record(value.get("deadline_lock"), lock_sha, "Pantheon")
    for key in ("pantheon_output_trace", "runtime_log", "background", "runtime_binary"):
        _record(value.get(key), f"Pantheon {key}")
    for index, record in enumerate(value.get("upstream_sources", [])):
        _record(record, f"Pantheon source {index}")
    _record(value.get("adapter"), "Pantheon adapter")
    reference = _finite(value.get("reference_accuracy"), "Pantheon")
    candidate = _finite(value.get("pantheon_accuracy"), "Pantheon")
    delta = _finite(value.get("accuracy_delta"), "Pantheon")
    if reference < 0.8 or candidate < 0.8 or not math.isclose(delta, 0.0, abs_tol=1e-12):
        raise ValueError("Pantheon accuracy gate differs")
    accuracy = {
        "reference": reference,
        "candidate": candidate,
        "delta": delta,
        "gate": {"path": _portable(path), "sha256": source_sha},
    }
    return _metric_row(
        system="Pantheon", requests=int(value["requests"]),
        misses=int(value["deadline_misses"]), p99_us=_finite(value["p99_us"], "Pantheon"),
        goodput=_finite(value["background_goodput_rps"], "Pantheon"),
        accuracy=accuracy, source=path, source_sha=source_sha,
        topology="fixed-2g+1g-native-block-exit-port",
        evidence_scope="published-native-runtime-common-gate",
        timing_scope="actual-release-to-native-exit-completion-integer-deadline",
    )


def summarize(
    *, common_path: Path, deadline_path: Path, quiet_path: Path, mig_path: Path,
    mig_colocated_path: Path, mps_path: Path, xsched_path: Path,
    orion_path: Path, pantheon_path: Path,
) -> dict[str, Any]:
    common, common_sha = _common(common_path)
    _, lock_sha, deadline = _deadline(deadline_path)
    systems = {
        "QUIET": _pipeline_row("QUIET", quiet_path.resolve(), common, common_sha, lock_sha, deadline),
        "NVIDIA MIG": _pipeline_row("NVIDIA MIG", mig_path.resolve(), common, common_sha, lock_sha, deadline),
        "NVIDIA MPS": _pipeline_row("NVIDIA MPS", mps_path.resolve(), common, common_sha, lock_sha, deadline),
        "XSched": _xsched_row(xsched_path.resolve(), common, common_sha, lock_sha, deadline),
        "Orion": _orion_row(orion_path.resolve(), common, common_sha, lock_sha, deadline),
        "Pantheon": _pantheon_row(pantheon_path.resolve(), common, common_sha, lock_sha, deadline),
    }
    if tuple(systems) != SYSTEM_ORDER:
        raise ValueError("fixed comparator display order differs")
    colocated_mig, mig_constraints = _colocated_mig_row(
        mig_colocated_path.resolve(), common, common_sha, lock_sha, deadline
    )
    return {
        "schema_version": 1,
        "kind": "p9-six-system-imagenette-common-gate",
        "proposed_system": "QUIET",
        "system_order": list(SYSTEM_ORDER),
        "workload": "resnet50-classification",
        "scope": "single-directional-common-input-arrival-deadline-gate",
        "formal": False,
        "ranking_allowed": False,
        "requests_per_system": REQUESTS,
        "deadline_us": deadline,
        "dmr_target": 0.0005,
        "common_workload": {"path": _portable(common_path), "sha256": common_sha},
        "deadline_lock": {"path": _portable(deadline_path), "sha256": lock_sha},
        "contract_hashes": {
            "arrival_trace_sha256": common["arrival_trace_sha256"],
            "operational_arrival_trace_sha256": common["operational_arrival_trace_sha256"],
            "producer_input_trace_sha256": common["producer_input_trace_sha256"],
            "dataset_manifest_sha256": common["dataset_manifest_sha256"],
        },
        "systems": systems,
        "partial_topology_comparisons": {
            colocated_mig["system"]: colocated_mig,
        },
        "mig_topology_constraints": mig_constraints,
        "claim_guard": (
            "Every row is measured on the same labelled 90-input contract and frozen deadline. "
            "The fixed figure reports coverage and SLO feasibility, not a formal ranking: the "
            "fixed-stage MIG row reserves both slices and its BE metric is not applicable. The "
            "separate colocated-MIG control admits BE by changing stage placement and therefore "
            "remains partial evidence. Orion's differential fidelity gate remains open, Pantheon "
            "floors its integer-microsecond deadline, and no row has session/thermal power."
        ),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common", type=Path, default=DEFAULT_COMMON)
    parser.add_argument("--deadline-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--quiet", type=Path, default=DEFAULT_QUIET)
    parser.add_argument("--mig", type=Path, default=DEFAULT_MIG)
    parser.add_argument(
        "--mig-colocated", type=Path, default=DEFAULT_MIG_COLOCATED
    )
    parser.add_argument("--mps", type=Path, default=DEFAULT_MPS)
    parser.add_argument("--xsched", type=Path, default=DEFAULT_XSCHED)
    parser.add_argument("--orion", type=Path, default=DEFAULT_ORION)
    parser.add_argument("--pantheon", type=Path, default=DEFAULT_PANTHEON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    value = summarize(
        common_path=args.common.resolve(), deadline_path=args.deadline_lock.resolve(),
        quiet_path=args.quiet.resolve(), mig_path=args.mig.resolve(),
        mig_colocated_path=args.mig_colocated.resolve(), mps_path=args.mps.resolve(),
        xsched_path=args.xsched.resolve(), orion_path=args.orion.resolve(),
        pantheon_path=args.pantheon.resolve(),
    )
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
