#!/usr/bin/env python3
"""Compare QUIET with a same-contract SOTA port without inventing numbers.

The competitor file is deliberately an adapter output, not a local policy
summary.  A missing adapter is reported as ``not-run`` rather than being
silently replaced by an MPS or fixed-gating ablation.  The comparator manifest
also decides whether an accepted adapter may contribute numeric SLO/goodput
claims; structural and functional evidence is retained without promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
from typing import Any

MANIFEST_PATH = pathlib.Path(__file__).resolve().parents[1] / "docs" / "p9-comparator-manifest.json"


def _parse_output_trace(path: pathlib.Path) -> dict[str, Any]:
    module_path = pathlib.Path(__file__).with_name("read_application_output_trace.py")
    spec = importlib.util.spec_from_file_location("p9_output_trace_reader_compare", module_path)
    if spec is None or spec.loader is None:
        raise ValueError("application output trace parser is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse(path)

SYSTEMS = {
    "BOER": {
        "paper": "SC 2025",
        "source": "https://github.com/TsingYiPainter/SC25_BOER",
        "source_commit": "df54815de3b1c9059f873a17c13f7d5203eedd3e",
        "mechanism": "MIG+MPS spatial configuration with Bayesian optimization",
        "required_fidelity": "algorithm-preserving-thor-port",
    },
    "ParvaGPU": {
        "paper": "SC 2024",
        "source": "https://github.com/MunQ-Lee/ParvaGPU_SC24",
        "source_commit": "5f3de1e18582b4c81896a1c3eb0e2915238dfee6",
        "mechanism": "MIG segment placement with MPS replicas",
        "required_fidelity": "algorithm-preserving-thor-port",
    },
    "Orion": {
        "paper": "EuroSys 2024",
        "source": "https://github.com/eth-easl/orion",
        "source_commit": "20f9469764fb96d94ce23a8e70615196e9ce4ba1",
        "mechanism": "fine-grained CUDA library interception and sharing",
        "required_fidelity": "native-interposition-port",
    },
    "XSched": {
        "paper": "OSDI 2025",
        "source": "https://github.com/XpuOS/xsched-artifacts",
        "source_commit": "bd494cb7a72958cd11900243a0798df00d856c6e",
        "mechanism": "preemptible command queues with priority scheduling",
        "required_fidelity": "native-xqueue-port",
    },
    "Pantheon": {
        "paper": "MobiSys 2024",
        "source": "https://github.com/PantheonInfer/Pantheon",
        "source_commit": "1caa4321fe9f9902ffacb78978f11a32a7a62f64",
        "mechanism": "deadline-aware block scheduling and early exits for edge inference",
        "required_fidelity": "native-block-runtime-port",
    },
}


def _manifest_row(system: str) -> dict[str, Any]:
    try:
        value = json.loads(MANIFEST_PATH.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("comparator manifest cannot be loaded") from error
    if not isinstance(value, dict) or value.get("proposed_system") != "QUIET":
        raise ValueError("comparator manifest is not a QUIET manifest")
    rows = value.get("rows")
    if not isinstance(rows, dict) or not isinstance(rows.get(system), dict):
        # Structural controls are intentionally not headline rows, but they
        # still need an explicit claim boundary when queried by the adapter.
        controls = value.get("same_slo_baselines", {})
        if not isinstance(controls, dict) or not isinstance(controls.get(system), dict):
            structural = value.get("structural_controls", [])
            if isinstance(structural, list) and system in structural:
                return {
                    "numeric_comparison_allowed": False,
                    "status": "structural-only",
                }
            raise ValueError(f"comparator manifest has no row for {system}")
        return controls[system]
    return rows[system]


def claim_contract(system: str) -> dict[str, Any]:
    if system == "QUIET":
        return {
            "numeric_comparison_allowed": True,
            "claim_level": "numeric-frontier",
            "manifest_status": "proposed-system",
        }
    if system not in SYSTEMS:
        raise ValueError(f"unknown comparator: {system}")
    row = _manifest_row(system)
    allowed = row.get("numeric_comparison_allowed") is True
    result = {
        "numeric_comparison_allowed": allowed,
        "claim_level": "numeric-frontier" if allowed else "functional-or-structural-only",
        "manifest_status": row.get("status", "unknown"),
    }
    return result


def load_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _output_trace_bound(gate: dict[str, Any], key: str) -> bool:
    trace = gate.get(key)
    if not isinstance(trace, dict):
        return False
    path = trace.get("path")
    expected = trace.get("sha256")
    count = trace.get("record_count")
    if (
        not isinstance(path, str) or not path
        or not isinstance(expected, str) or len(expected) != 64
        or not isinstance(count, int) or isinstance(count, bool) or count <= 0
        or trace.get("capture_boundary") != "post-completion"
    ):
        return False
    try:
        actual = hashlib.sha256(pathlib.Path(path).resolve().read_bytes()).hexdigest()
    except OSError:
        return False
    if actual != expected:
        return False
    try:
        parsed = _parse_output_trace(pathlib.Path(path))
    except (OSError, ValueError):
        return False
    return parsed.get("record_count") == count


def application_accuracy_gate_bound(
    summary: dict[str, Any], *, require_output_traces: bool = False,
    require_input_binding: bool = True,
) -> bool:
    """Return true only for a passed, byte-bound trained-application gate."""
    record = summary.get("application_accuracy_gate")
    if not isinstance(record, dict):
        return False
    path = record.get("path")
    expected = record.get("sha256")
    if not isinstance(path, str) or not path or not isinstance(expected, str) or len(expected) != 64:
        return False
    try:
        gate_bytes = pathlib.Path(path).resolve().read_bytes()
        actual = hashlib.sha256(gate_bytes).hexdigest()
        gate = json.loads(gate_bytes)
    except (OSError, json.JSONDecodeError):
        return False
    passed = (
        actual == expected
        and isinstance(gate, dict)
        and gate.get("kind") == "p9-application-accuracy-gate"
        and gate.get("status") == "passed"
        and gate.get("numeric_comparison_allowed") is True
    )
    if not passed:
        return False
    minimum_accuracy = gate.get("minimum_accuracy")
    reference_accuracy = gate.get("reference_accuracy")
    candidate_accuracy = gate.get("candidate_accuracy")
    if (
        isinstance(minimum_accuracy, bool)
        or not isinstance(minimum_accuracy, (int, float))
        or not math.isfinite(float(minimum_accuracy))
        or not 0.0 <= float(minimum_accuracy) <= 1.0
        or isinstance(reference_accuracy, bool)
        or not isinstance(reference_accuracy, (int, float))
        or not math.isfinite(float(reference_accuracy))
        or isinstance(candidate_accuracy, bool)
        or not isinstance(candidate_accuracy, (int, float))
        or not math.isfinite(float(candidate_accuracy))
        or float(reference_accuracy) < float(minimum_accuracy)
        or float(candidate_accuracy) < float(minimum_accuracy)
    ):
        return False

    if require_input_binding:
        if (
            gate.get("application_input_binding_required") is not True
            or gate.get("application_input_binding_contract") != "passed"
        ):
            return False

    # The gate must remain tied to the exact input evidence after it is
    # produced.  A digest without a path cannot detect post-gate replacement
    # of the reference/candidate request traces or dataset manifest.
    for path_key, digest_key in (
        ("dataset_manifest_path", "dataset_manifest_sha256"),
        ("reference_trace_path", "reference_trace_sha256"),
        ("candidate_trace_path", "candidate_trace_sha256"),
    ):
        evidence_path = gate.get(path_key)
        evidence_digest = gate.get(digest_key)
        if (
            not isinstance(evidence_path, str)
            or not evidence_path
            or not isinstance(evidence_digest, str)
            or len(evidence_digest) != 64
            or any(character not in "0123456789abcdef" for character in evidence_digest)
        ):
            return False
        try:
            observed = hashlib.sha256(pathlib.Path(evidence_path).resolve().read_bytes()).hexdigest()
        except OSError:
            return False
        if observed != evidence_digest:
            return False

    if require_input_binding:
        for prefix in ("reference", "candidate"):
            record = gate.get(f"{prefix}_pipeline_csv")
            if not isinstance(record, dict):
                return False
            path_value, expected = record.get("path"), record.get("sha256")
            if not isinstance(path_value, str) or not isinstance(expected, str):
                return False
            try:
                observed = hashlib.sha256(pathlib.Path(path_value).resolve().read_bytes()).hexdigest()
            except OSError:
                return False
            if observed != expected:
                return False

    if not require_output_traces:
        return passed
    return (
        gate.get("application_output_trace_required") is True
        and gate.get("application_output_trace_contract") == "passed"
        and _output_trace_bound(gate, "reference_output_trace")
        and _output_trace_bound(gate, "candidate_output_trace")
    )


def numeric_accuracy_contract_bound(
    proposed: dict[str, Any], competitor: dict[str, Any],
    *, require_output_traces: bool = False, require_input_binding: bool = True,
) -> bool:
    """Require task-accuracy evidence on both sides of a numeric comparison."""
    return (
        application_accuracy_gate_bound(
            proposed, require_output_traces=require_output_traces,
            require_input_binding=require_input_binding,
        )
        and application_accuracy_gate_bound(
            competitor, require_output_traces=require_output_traces,
            require_input_binding=require_input_binding,
        )
    )


def formal_evidence_bound(summary: dict[str, Any]) -> bool:
    """Require the frozen evidence bundle before allowing numeric ranking.

    Accuracy and byte-bound output traces establish semantic equivalence, but
    they do not establish a paper-level SLO comparison.  The latter also needs
    a declared formal result, thermal normalization, independent session
    statistics, frozen lock provenance, and a one-sided CP95 miss bound.
    Exploratory summaries deliberately fail this predicate.
    """
    if (
        summary.get("formal") is not True
        or summary.get("thermal_normalized") is not True
        or summary.get("ranking_allowed") is not True
    ):
        return False
    sessions = summary.get("session_level_statistics")
    if not isinstance(sessions, dict):
        return False
    run_count = sessions.get("run_count")
    if (
        isinstance(run_count, bool)
        or not isinstance(run_count, int)
        or run_count < 14
        or sessions.get("unit") not in {"independent-session", "run"}
        or sessions.get("paired_williams") is not True
    ):
        return False
    for key in ("deadline_lock_sha256", "thermal_lock_sha256"):
        digest = summary.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            return False
    certification = summary.get("slo_certification")
    config = summary.get("config")
    target = config.get("dmr_target") if isinstance(config, dict) else None
    upper = certification.get("cp95_upper_dmr") if isinstance(certification, dict) else None
    if (
        not isinstance(certification, dict)
        or certification.get("method") != "one-sided-clopper-pearson-95"
        or isinstance(upper, bool)
        or not isinstance(upper, (int, float))
        or not math.isfinite(float(upper))
        or isinstance(target, bool)
        or not isinstance(target, (int, float))
        or not math.isfinite(float(target))
        or float(upper) > float(target)
    ):
        return False
    return True


def _bound_artifact(path_value: Any, digest_value: Any) -> bool:
    """Re-hash a comparator artifact from the path recorded by its gate."""
    if (
        not isinstance(path_value, str)
        or not path_value
        or not isinstance(digest_value, str)
        or len(digest_value) != 64
        or any(character not in "0123456789abcdef" for character in digest_value)
    ):
        return False
    try:
        observed = hashlib.sha256(pathlib.Path(path_value).resolve().read_bytes()).hexdigest()
    except OSError:
        return False
    return observed == digest_value


def quiet_contract(summary: dict[str, Any], scenario: str) -> dict[str, Any]:
    config = summary.get("config")
    if not isinstance(config, dict):
        raise ValueError("QUIET summary lacks config")
    actual_scenario = config.get("scenario", "independent")
    if actual_scenario != scenario:
        raise ValueError(
            f"QUIET summary scenario={actual_scenario!r}, expected {scenario!r}"
        )
    required = {
        "critical_placement": "2g",
        "resident_placement": "1g",
        "borrower_placement": "2g",
        "includes_transfers": True,
        "worker_max_inflight": 1,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"QUIET contract mismatch: {key} != {expected!r}")
    workload = summary.get("common_workload")
    if not isinstance(workload, dict):
        raise ValueError("QUIET summary lacks common workload contract")
    workload_required = {
        "workload_id": str,
        "topology": str,
        "placement": str,
        "input_tensor": str,
        "payload_bytes": int,
        "arrival_trace_sha256": str,
        "dataset_manifest_sha256": str,
    }
    if workload.get("workload_id") in {"resnet-detection-head", "resnet50-classification"}:
        workload_required.update({
            "producer_input_trace_path": str,
            "producer_input_trace_sha256": str,
        })
    for key, expected_type in workload_required.items():
        value = workload.get(key)
        if expected_type is int:
            valid = isinstance(value, int) and not isinstance(value, bool) and value > 0
        else:
            valid = isinstance(value, expected_type) and bool(value)
            if key.endswith("_sha256"):
                valid = valid and len(value) == 64 and all(
                    character in "0123456789abcdef" for character in value
                )
        if not valid:
            raise ValueError(f"QUIET workload contract field is invalid: {key}")
        if key == "producer_input_trace_path":
            try:
                if not pathlib.Path(value).resolve().is_file():
                    raise ValueError("missing producer input trace")
            except OSError as error:
                raise ValueError("producer input trace path is invalid") from error
        if key == "producer_input_trace_sha256":
            try:
                observed = hashlib.sha256(
                    pathlib.Path(workload["producer_input_trace_path"]).resolve().read_bytes()
                ).hexdigest()
            except OSError as error:
                raise ValueError("producer input trace cannot be hashed") from error
            if observed != value:
                raise ValueError("producer input trace SHA mismatches")
    production_wall = summary.get("latency_contract") == (
        "production-wall-arrival-to-completion"
    )
    if production_wall and summary.get("production_wall_definition") != (
        "arrival-to-consumer-completion-excludes-correctness-validation"
    ):
        raise ValueError("QUIET production-wall contract is stale or unbound")
    result = {
        "scenario": scenario,
        "epochs": config.get("epochs"),
        "samples_per_epoch": config.get("samples_per_epoch"),
        "period_ms": config.get("period_ms"),
        "pressure_rps_per_tenant": config.get("pressure_rps_per_tenant", 0.0),
        "burst_size": config.get("burst_size"),
        "deadline_ms": summary.get("deadline_ms"),
        "dmr_target": config.get("dmr_target"),
        "critical_placement": "2g",
        "pressure_layout": "1g+2g",
        "includes_transfers": True,
        "worker_max_inflight": 1,
        "common_workload": {
            key: workload[key] for key in workload_required
        },
    }
    if production_wall:
        result["latency_contract"] = summary.get("latency_contract")
        result["correctness_validation_placement"] = summary.get(
            "correctness_validation_placement"
        )
        result["production_wall_definition"] = (
            "arrival-to-consumer-completion-excludes-correctness-validation"
        )
    return result


def quiet_metrics(summary: dict[str, Any]) -> dict[str, float | int]:
    policies = summary.get("policies")
    if not isinstance(policies, list):
        raise ValueError("QUIET summary lacks policy results")
    matches = [policy for policy in policies if policy.get("name") == "mig-governor"]
    if len(matches) != 1:
        raise ValueError("QUIET summary must contain exactly one proposed-policy result")
    policy = matches[0]
    result: dict[str, float | int] = {}
    for output_name, input_name in (
        ("pressure_goodput_per_second", "pressure_goodput_per_second"),
        ("deadline_miss_rate", "deadline_miss_rate"),
        ("p99_ms", "critical_p99_ms_max"),
    ):
        value = policy.get(input_name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"QUIET metric {input_name} is invalid")
        result[output_name] = float(value)
    requests = policy.get("critical_requests")
    misses = policy.get("deadline_misses")
    if (
        not isinstance(requests, int)
        or isinstance(requests, bool)
        or requests <= 0
        or not isinstance(misses, int)
        or isinstance(misses, bool)
        or misses < 0
        or misses > requests
    ):
        raise ValueError("QUIET critical request counts are invalid")
    if not math.isclose(
        float(result["deadline_miss_rate"]), misses / requests,
        rel_tol=1e-9, abs_tol=1e-12,
    ):
        raise ValueError("QUIET deadline miss count and rate differ")
    result["critical_requests"] = requests
    result["deadline_misses"] = misses
    return result


def validate_competitor(
    value: dict[str, Any], system: str, contract: dict[str, Any]
) -> None:
    if value.get("system") != system:
        raise ValueError("competitor system name does not match adapter selection")
    provenance = value.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("competitor adapter lacks provenance")
    spec = SYSTEMS[system]
    if provenance.get("upstream_commit") != spec["source_commit"]:
        raise ValueError("competitor upstream commit is not the pinned artifact")
    if provenance.get("fidelity") != spec["required_fidelity"]:
        raise ValueError("competitor port fidelity is insufficient")
    profile_hashes = provenance.get("thor_profile_sha256")
    if not isinstance(profile_hashes, dict) or not profile_hashes:
        raise ValueError("competitor lacks regenerated Thor profile hashes")
    for name, digest in profile_hashes.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("competitor Thor profile hash is invalid")
    for key in (
        "scenario",
        "epochs",
        "samples_per_epoch",
        "period_ms",
        "pressure_rps_per_tenant",
        "burst_size",
        "deadline_ms",
        "dmr_target",
        "common_workload",
        "latency_contract",
        "production_wall_definition",
        "correctness_validation_placement",
    ):
        if value.get("contract", {}).get(key) != contract.get(key):
            raise ValueError(f"competitor contract mismatch: {key}")
    # A pinned source tree and a native wrapper are necessary but not
    # sufficient for a published-system numeric claim.  These two systems
    # require an explicit same-workload fidelity gate before their metrics can
    # enter a comparison table.
    if system == "Orion":
        gate = provenance.get("differential_gate")
        required_workload = contract.get("common_workload")
        gate_workload = gate.get("common_workload") if isinstance(gate, dict) else None
        def valid_sha(value: Any) -> bool:
            return (
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
            )
        if (
            not isinstance(gate, dict)
            or gate.get("schema_version") != 1
            or gate.get("kind") != "orion-differential-fidelity-gate"
            or gate.get("system") != "Orion"
            or gate.get("status") != "passed"
            or gate.get("reference") != "pinned-upstream-scheduler"
            or gate.get("upstream_commit") != spec["source_commit"]
            or not isinstance(gate.get("decision_cases"), int)
            or isinstance(gate.get("decision_cases"), bool)
            or gate.get("decision_cases") <= 0
            or not isinstance(gate.get("mismatch_cases"), int)
            or isinstance(gate.get("mismatch_cases"), bool)
            or gate.get("mismatch_cases") != 0
            or gate.get("numeric_comparison_allowed") is not True
            or gate.get("reference_checkout_verified") is not True
            or gate.get("reference_git_head") != spec["source_commit"]
            or not isinstance(gate.get("reference_git_root"), str)
            or not gate.get("reference_git_root")
            or not isinstance(gate.get("reference_source_relative_path"), str)
            or not gate.get("reference_source_relative_path")
            or not valid_sha(gate.get("reference_source_sha256"))
            or not isinstance(gate.get("reference_source_path"), str)
            or not gate.get("reference_source_path")
            or gate.get("reference_source_verified") is not True
            or gate.get("upstream_runtime_binary_verified") is not True
            or not _bound_artifact(
                gate.get("upstream_runtime_binary_path"),
                gate.get("upstream_runtime_binary_sha256"),
            )
            or not _bound_artifact(
                gate.get("reference_trace_path"), gate.get("reference_trace_sha256")
            )
            or not _bound_artifact(
                gate.get("port_trace_path"), gate.get("port_trace_sha256")
            )
            or not isinstance(gate.get("reference_trace_provenance"), dict)
            or gate["reference_trace_provenance"].get("generator")
                != "pinned-upstream-orion-runtime"
            or not valid_sha(gate["reference_trace_provenance"].get("sha256"))
            or gate["reference_trace_provenance"].get("reference_trace_sha256")
                != gate.get("reference_trace_sha256")
            or gate["reference_trace_provenance"].get(
                "upstream_runtime_binary_path"
            ) != gate.get("upstream_runtime_binary_path")
            or gate["reference_trace_provenance"].get(
                "upstream_runtime_binary_sha256"
            ) != gate.get("upstream_runtime_binary_sha256")
            or not valid_sha(
                gate["reference_trace_provenance"].get("common_workload_sha256")
            )
            or not isinstance(required_workload, dict)
            or not isinstance(gate_workload, dict)
            or any(
                gate_workload.get(key) != required_workload.get(key)
                for key in (
                    "workload_id", "topology", "placement", "input_tensor",
                    "payload_bytes", "arrival_trace_sha256", "dataset_manifest_sha256",
                )
            )
        ):
            raise ValueError("Orion differential fidelity or workload gate is missing or failed")
    elif system == "Pantheon":
        adapter = provenance.get("common_workload_adapter")
        required_workload = contract.get("common_workload")
        adapter_workload = adapter.get("common_workload") if isinstance(adapter, dict) else None
        def valid_sha(value: Any) -> bool:
            return (
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
            )
        numeric_fields_valid = all(
            isinstance(adapter.get(name), (int, float))
            and not isinstance(adapter.get(name), bool)
            and math.isfinite(float(adapter.get(name)))
            for name in (
                "reference_accuracy", "pantheon_accuracy", "accuracy_delta",
                "accuracy_tolerance",
            )
        ) if isinstance(adapter, dict) else False
        if (
            not isinstance(adapter, dict)
            or adapter.get("schema_version") != 1
            or adapter.get("kind") != "pantheon-common-workload-accuracy-gate"
            or adapter.get("system") != "Pantheon"
            or adapter.get("status") != "passed"
            or adapter.get("upstream_commit") != spec["source_commit"]
            or adapter.get("workload") != "p9-dependent-tensorrt-dag"
            or not isinstance(adapter.get("deadline_us"), (int, float))
            or isinstance(adapter.get("deadline_us"), bool)
            or not math.isfinite(float(adapter.get("deadline_us")))
            or not math.isclose(
                float(adapter.get("deadline_us")),
                float(contract.get("deadline_ms")) * 1000.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or adapter.get("accuracy_equivalent") is not True
            or adapter.get("shared_arrival_trace") is not True
            or adapter.get("numeric_comparison_allowed") is not True
            or not isinstance(adapter.get("decision_cases"), int)
            or isinstance(adapter.get("decision_cases"), bool)
            or adapter.get("decision_cases") <= 0
            or not valid_sha(adapter.get("reference_trace_sha256"))
            or not valid_sha(adapter.get("port_trace_sha256"))
            or not _bound_artifact(
                adapter.get("reference_trace_path"), adapter.get("reference_trace_sha256")
            )
            or not _bound_artifact(
                adapter.get("port_trace_path"), adapter.get("port_trace_sha256")
            )
            or not isinstance(adapter.get("upstream_source_path"), str)
            or not adapter.get("upstream_source_path")
            or not valid_sha(adapter.get("upstream_source_sha256"))
            or adapter.get("upstream_source_verified") is not True
            or adapter.get("runtime_binary_verified") is not True
            or not _bound_artifact(
                adapter.get("runtime_binary_path"),
                adapter.get("runtime_binary_sha256"),
            )
            or adapter.get("upstream_checkout_verified") is not True
            or adapter.get("upstream_git_head") != spec["source_commit"]
            or not isinstance(adapter.get("upstream_git_root"), str)
            or not adapter.get("upstream_git_root")
            or not isinstance(adapter.get("upstream_source_relative_path"), str)
            or not adapter.get("upstream_source_relative_path")
            or not isinstance(adapter.get("training_result_path"), str)
            or not adapter.get("training_result_path")
            or not valid_sha(adapter.get("training_result_sha256"))
            or adapter.get("training_artifact_verified") is not True
            or not numeric_fields_valid
            or float(adapter["accuracy_delta"]) > float(adapter["accuracy_tolerance"])
            or not isinstance(required_workload, dict)
            or not isinstance(adapter_workload, dict)
            or any(
                adapter_workload.get(key) != required_workload.get(key)
                for key in (
                    "workload_id", "topology", "placement", "input_tensor",
                    "payload_bytes", "arrival_trace_sha256", "dataset_manifest_sha256",
                )
            )
        ):
            raise ValueError("Pantheon accuracy gate or workload contract is missing or failed")
    metrics = value.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("competitor adapter must provide metrics")
    for key in ("pressure_goodput_per_second", "deadline_miss_rate", "p99_ms"):
        number = metrics.get(key)
        if (
            not isinstance(number, (int, float))
            or isinstance(number, bool)
            or not math.isfinite(float(number))
        ):
            raise ValueError(f"competitor metric {key} is missing or non-numeric")
    requests = metrics.get("critical_requests")
    misses = metrics.get("deadline_misses")
    if (
        not isinstance(requests, int)
        or isinstance(requests, bool)
        or requests <= 0
        or not isinstance(misses, int)
        or isinstance(misses, bool)
        or misses < 0
        or misses > requests
        or not math.isclose(
            float(metrics["deadline_miss_rate"]), misses / requests,
            rel_tol=1e-9, abs_tol=1e-12,
        )
    ):
        raise ValueError("competitor deadline miss count and rate differ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet-summary", type=pathlib.Path, required=True)
    parser.add_argument("--system", choices=tuple(SYSTEMS), required=True)
    parser.add_argument("--scenario", choices=("independent", "dependent"), required=True)
    parser.add_argument("--competitor-summary", type=pathlib.Path)
    parser.add_argument(
        "--require-output-traces", action="store_true",
        help="require raw post-completion traces in both accuracy gates",
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    quiet_summary = load_json(args.quiet_summary)
    contract = quiet_contract(quiet_summary, args.scenario)
    proposed_metrics = quiet_metrics(quiet_summary)
    spec = SYSTEMS[args.system]
    claim = claim_contract(args.system)
    result: dict[str, Any] = {
        "schema_version": 1,
        "proposed_system": "QUIET",
        "competitor": args.system,
        "competitor_spec": spec,
        "contract": contract,
        "proposed_metrics": proposed_metrics,
        "claim_contract": claim,
        "status": "not-run",
        "reason": "no same-contract adapter output was supplied",
    }
    proposed_accuracy_bound = application_accuracy_gate_bound(
        quiet_summary, require_output_traces=args.require_output_traces
    )
    result["application_accuracy_gate_bound"] = proposed_accuracy_bound
    proposed_formal_bound = formal_evidence_bound(quiet_summary)
    result["proposed_formal_evidence_bound"] = proposed_formal_bound
    if claim["numeric_comparison_allowed"] and not proposed_accuracy_bound:
        result["claim_contract"] = {
            **claim,
            "numeric_comparison_allowed": False,
            "claim_level": "exploratory-until-application-accuracy-gate",
        }
    if args.competitor_summary is not None:
        competitor = load_json(args.competitor_summary)
        validate_competitor(competitor, args.system, contract)
        competitor_accuracy_bound = application_accuracy_gate_bound(
            competitor, require_output_traces=args.require_output_traces
        )
        both_accuracy_bound = numeric_accuracy_contract_bound(
            quiet_summary, competitor,
            require_output_traces=args.require_output_traces,
        )
        result["competitor_application_accuracy_gate_bound"] = competitor_accuracy_bound
        competitor_formal_bound = formal_evidence_bound(competitor)
        result["competitor_formal_evidence_bound"] = competitor_formal_bound
        both_formal_bound = proposed_formal_bound and competitor_formal_bound
        if claim["numeric_comparison_allowed"]:
            result["status"] = (
                "measured"
                if both_accuracy_bound and both_formal_bound
                else (
                    "measured-accuracy-gate-pending"
                    if not both_accuracy_bound
                    else "measured-formal-evidence-pending"
                )
            )
        else:
            result["status"] = "measured-structural-only"
        result.pop("reason")
        result["competitor_metrics"] = competitor["metrics"]
        if claim["numeric_comparison_allowed"] and both_accuracy_bound and both_formal_bound:
            result["ranking_allowed"] = True
            competitor_goodput = float(
                competitor["metrics"]["pressure_goodput_per_second"]
            )
            result["goodput_ratio_quiet_over_competitor"] = (
                proposed_metrics["pressure_goodput_per_second"] / competitor_goodput
                if competitor_goodput > 0.0
                else None
            )
            target = contract.get("dmr_target")
            if (
                not isinstance(target, (int, float))
                or isinstance(target, bool)
                or not math.isfinite(float(target))
                or float(target) < 0.0
            ):
                raise ValueError("comparison DMR target is invalid")
            result["slo_assessment"] = {
                "criterion": "observed-deadline-miss-rate",
                "dmr_target": float(target),
                "quiet_observed_target_met": (
                    float(proposed_metrics["deadline_miss_rate"]) <= float(target)
                ),
                "competitor_observed_target_met": (
                    float(competitor["metrics"]["deadline_miss_rate"]) <= float(target)
                ),
                "certification": "not-certified; repeated samples and a confidence bound are required",
            }
            if not proposed_accuracy_bound or not competitor_accuracy_bound:
                result["claim_contract"] = {
                    **claim,
                    "numeric_comparison_allowed": False,
                    "claim_level": "exploratory-until-both-application-accuracy-gates",
                }
        elif claim["numeric_comparison_allowed"]:
            result["ranking_allowed"] = False
            result["claim_contract"] = {
                **claim,
                "numeric_comparison_allowed": False,
                "claim_level": "exploratory-until-formal-evidence-bundle",
            }
        else:
            result["claim_guard"] = (
                "Numeric ranking is withheld until both the comparator fidelity gate "
                "and QUIET's byte-bound application-accuracy gate pass."
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
