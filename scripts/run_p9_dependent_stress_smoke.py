#!/usr/bin/env python3
"""Run the payload-valid dependent-small interference smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Scenario:
    name: str
    same_instance: bool
    gate_mode: str | None
    gate_scope: str = "producer"
    best_effort_admitted: bool = True


DEFAULT_SCENARIOS = (
    # Pure MIG reserves both fixed slices for the dependent critical DAG. It
    # must not silently start an MPS BE client and then call the result MIG.
    Scenario("nvidia-mig-isolation", False, None, "producer", False),
    # MPS is the quota/stream baseline on the same fixed 1g+2g placement as
    # QUIET. Putting both critical stages in one 1g slice changes the
    # dependent workload and makes the comparison invalid.
    Scenario("nvidia-mps-spatial-sharing", False, None),
    # A same-SLO comparator: stop the complete best-effort pipeline around
    # every critical request, without QUIET's adaptive placement/quota logic.
    Scenario("static-full-gate", False, "stop", "pipeline"),
    Scenario("process-stop-ablation", False, "stop"),
    Scenario("quiet", False, "cooperative"),
)
# Legacy scenario IDs are retained for trace compatibility. These are local
# mechanism ablations, not implementations of the similarly named papers.
MECHANISM_SCENARIOS = (
    Scenario("gslice", False, None),
    Scenario("gpulet", False, None),
    Scenario("orion", False, "cooperative", "pipeline"),
)
SCENARIOS = DEFAULT_SCENARIOS + MECHANISM_SCENARIOS

# A placement variant changes the MIG slice that hosts the critical producer
# and consumer.  The best-effort client remains on the 1g slice, so reverse
# placement exercises contention on the consumer side instead of silently
# reusing the forward placement contract.
PLACEMENT_VARIANTS = {
    "fixed-1g-producer-2g-consumer": {
        "producer_slice": "1g",
        "consumer_slice": "2g",
        "producer_uuid_key": "JDG_MIG_SMALL_UUID",
        "consumer_uuid_key": "JDG_MIG_BIG_UUID",
    },
    "fixed-2g-producer-1g-consumer": {
        "producer_slice": "2g",
        "consumer_slice": "1g",
        "producer_uuid_key": "JDG_MIG_BIG_UUID",
        "consumer_uuid_key": "JDG_MIG_SMALL_UUID",
    },
}

PUBLIC_SYSTEM_NAMES = {
    "nvidia-mig-isolation": "NVIDIA MIG",
    "nvidia-mps-spatial-sharing": "NVIDIA MPS",
    "static-full-gate": "Static full gating",
    "process-stop-ablation": "Process-stop ablation",
    "quiet": "QUIET",
    "gslice": "Quota-only provisioning",
    "gpulet": "Partition-only planning",
    "orion": "Full-DAG quiescence",
}

WORKLOAD_PAYLOAD_BYTES = {
    "resnet-control": 14_720,
    "resnet-detection-head": 512 * 23 * 40 * 4,
    "resnet50-classification": 1024 * 14 * 14 * 4,
    "whisper-projection": 2_304_000,
}

RESNET50_SPLIT_TENSOR = "gpu_0/res4_5_branch2c_bn_2"


def load_env(path: pathlib.Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or not value or key in result:
            raise ValueError("invalid MIG environment file")
        result[key] = value
    for key in (
        "JDG_MIG_SMALL_UUID",
        "JDG_MIG_BIG_UUID",
        "JDG_MPS_PIPE_DIRECTORY",
        "JDG_MPS_LOG_DIRECTORY",
    ):
        if key not in result:
            raise ValueError(f"MIG environment lacks {key}")
    return result


def state(pid: int) -> str:
    text = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    end = text.rfind(")")
    if end < 0 or end + 2 >= len(text):
        raise RuntimeError("malformed process state")
    return text[end + 2]


def wait_paused(process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"background exited {process.returncode}: {stdout} {stderr}"
            )
        if state(process.pid) in {"T", "t"}:
            return
        time.sleep(0.02)
    raise TimeoutError("background start barrier timed out")


def stop(process: subprocess.Popen[str]) -> tuple[str, str]:
    if process.poll() is None:
        for number in (signal.SIGCONT, signal.SIGINT):
            try:
                os.kill(process.pid, number)
            except ProcessLookupError:
                break
    try:
        return process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.communicate()


def summarize(
    scenario: Scenario,
    pipeline: dict[str, Any],
    background: dict[str, Any],
    producer_quota: int = 100,
    background_quota: int = 100,
    trace_csv: pathlib.Path | None = None,
    placement_variant: str = "fixed-1g-producer-2g-consumer",
    application_output_trace: pathlib.Path | None = None,
    event_trace_csv: pathlib.Path | None = None,
) -> dict[str, Any]:
    if pipeline.get("status") != "ok" or pipeline.get("checksum_failures") != 0:
        raise ValueError(f"{scenario.name} pipeline correctness failed")
    deadline_mode = pipeline.get("deadline_mode", "wall")
    wall_p99 = pipeline["end_to_end_us"]["p99"]
    deadline_p99 = (
        pipeline["stage_latency_us"]["validation_excluded_end_to_end_p99"]
        if deadline_mode == "validation-excluded"
        else wall_p99
    )
    result = {
        "system": PUBLIC_SYSTEM_NAMES[scenario.name],
        "pipeline_requests": pipeline["iterations"],
        "deadline_misses": pipeline["deadline_misses"],
        "pipeline_p99_us": deadline_p99,
        "wall_pipeline_p99_us": wall_p99,
        "deadline_mode": deadline_mode,
        "production_wall_definition": pipeline.get(
            "production_wall_definition",
            "legacy-pre-validation-wall-unknown",
        ),
        "correctness_validation_placement": pipeline.get(
            "correctness_validation_placement", "unknown"
        ),
        "latency_contract": pipeline.get(
            "latency_contract",
            "production-wall-arrival-to-completion"
            if deadline_mode == "wall"
            else "audit-validation-excluded",
        ),
        "transport": pipeline.get("transport", "unknown"),
        "checksum_mode": pipeline.get("checksum_mode", "inline"),
        "correctness_validated": pipeline.get("correctness_validated", True),
        "activation_replay_trace": pipeline.get("activation_replay_trace"),
        "activation_replay_verified_requests": pipeline.get(
            "activation_replay_verified_requests", 0
        ),
        "stage_latency_us": pipeline["stage_latency_us"],
        "payload_bytes": pipeline["payload_bytes"],
        "unique_payload_checksums": pipeline["unique_payload_checksums"],
        "unique_policy_output_checksums": pipeline[
            "unique_policy_output_checksums"
        ],
        "background_goodput_rps": background["throughput_per_second"],
        "best_effort_admitted": scenario.best_effort_admitted,
        "best_effort_status": background.get("status", "completed"),
        "gate_p99_us": pipeline.get("gate_us", {}).get("p99"),
        "gate_scope": pipeline.get("gate_scope", "producer"),
        "producer_quota_percent": producer_quota,
        "background_quota_percent": background_quota,
        "placement_variant": placement_variant,
        # Bind the observed MIG topology into each row; placement_variant alone
        # is intent, while these fields are what the benchmark actually used.
        "producer_uuid": pipeline.get("producer_uuid"),
        "consumer_uuid": pipeline.get("consumer_uuid"),
        "producer_sms": pipeline.get("producer_sms"),
        "consumer_sms": pipeline.get("consumer_sms"),
        "consumer_engine_mode": pipeline.get(
            "consumer_engine_mode", "generated-control-policy"
        ),
        "consumer_input_tensor": pipeline.get("consumer_input_tensor", "features"),
    }
    if trace_csv is not None:
        result["request_trace"] = {
            "path": str(trace_csv),
            "sha256": sha256(trace_csv),
        }
    if application_output_trace is not None:
        if not application_output_trace.is_file():
            raise ValueError("application output trace is missing")
        result["application_output_trace"] = {
            "path": str(application_output_trace),
            "sha256": sha256(application_output_trace),
            "capture_boundary": "post-completion",
        }
    if event_trace_csv is not None:
        if not event_trace_csv.is_file():
            raise ValueError("event trace is missing")
        result["event_trace"] = {
            "path": str(event_trace_csv),
            "sha256": sha256(event_trace_csv),
            "fields": "arrival-publication-resume-gate-hold",
        }
    return result


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_common_workload_contract(
    path: pathlib.Path,
    *,
    workload: str,
    placement: str,
    input_tensor: str,
    payload_bytes: int,
) -> dict[str, Any]:
    """Load and revalidate the shared workload evidence used by all arms."""
    resolved = path.resolve()
    raw = resolved.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("common workload contract is not newline-complete")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("common workload contract is invalid JSON") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("common workload contract schema differs")
    expected = {
        "workload_id": workload,
        "topology": "fixed-2g+1g",
        "placement": placement,
        "input_tensor": input_tensor,
        "payload_bytes": payload_bytes,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"common workload contract differs at {key}")
    for path_key, digest_key in (
        ("arrival_trace_path", "arrival_trace_sha256"),
        ("dataset_manifest_path", "dataset_manifest_sha256"),
    ):
        evidence_path = value.get(path_key)
        expected_digest = value.get(digest_key)
        if (
            not isinstance(evidence_path, str)
            or not evidence_path
            or not isinstance(expected_digest, str)
            or len(expected_digest) != 64
            or any(character not in "0123456789abcdef" for character in expected_digest)
        ):
            raise ValueError(f"common workload contract field is invalid: {path_key}")
        evidence = pathlib.Path(evidence_path).resolve()
        if not evidence.is_file() or sha256(evidence) != expected_digest:
            raise ValueError(f"common workload evidence SHA mismatches: {path_key}")
        value[path_key] = str(evidence)
    producer_path_value = value.get("producer_input_trace_path")
    producer_digest = value.get("producer_input_trace_sha256")
    if producer_path_value is not None or producer_digest is not None:
        if (
            not isinstance(producer_path_value, str)
            or not producer_path_value
            or not isinstance(producer_digest, str)
            or len(producer_digest) != 64
            or any(character not in "0123456789abcdef" for character in producer_digest)
        ):
            raise ValueError("common workload producer input trace binding is invalid")
        producer_path = pathlib.Path(producer_path_value).resolve()
        if not producer_path.is_file() or sha256(producer_path) != producer_digest:
            raise ValueError("common workload producer input trace SHA mismatches")
        value["producer_input_trace_path"] = str(producer_path)
    operational_path_value = value.get("operational_arrival_trace_path")
    operational_digest = value.get("operational_arrival_trace_sha256")
    if operational_path_value is not None or operational_digest is not None:
        if (
            not isinstance(operational_path_value, str)
            or not operational_path_value
            or not isinstance(operational_digest, str)
            or len(operational_digest) != 64
            or any(character not in "0123456789abcdef" for character in operational_digest)
        ):
            raise ValueError("common workload operational arrival binding is invalid")
        operational_path = pathlib.Path(operational_path_value).resolve()
        if not operational_path.is_file() or sha256(operational_path) != operational_digest:
            raise ValueError("common workload operational arrival SHA mismatches")
        value["operational_arrival_trace_path"] = str(operational_path)
    value["contract_path"] = str(resolved)
    value["contract_sha256"] = hashlib.sha256(raw).hexdigest()
    return value


def validate_quiet_plan(
    path: pathlib.Path,
    deadline_us: float,
    deadline_lock: dict[str, str] | None,
    producer_quota: int,
    background_quota: int,
    gate_scope: str,
    workload: str,
    placement_variant: str = "fixed-1g-producer-2g-consumer",
    transport: str = "registered-direct",
) -> tuple[dict[str, Any], dict[str, str]]:
    placement = PLACEMENT_VARIANTS.get(placement_variant)
    if placement is None:
        raise ValueError("unknown QUIET placement variant")
    path = path.resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    selected = raw.get("selected_plan")
    if (
        raw.get("proposed_system") != "QUIET"
        or raw.get("status") != "selected"
        or not isinstance(selected, dict)
        or selected.get("feasible") is not True
        or not math.isclose(float(raw.get("deadline_us", -1)), deadline_us)
        or raw.get("deadline_lock") != deadline_lock
        or selected.get("placement")
        != {
            "producer": f"{placement['producer_slice']}-q{producer_quota}",
            "consumer": f"{placement['consumer_slice']}-q100",
        }
        or selected.get("placement_variant", placement_variant)
        != placement_variant
        or selected.get("candidate_id") not in {
            f"q{producer_quota}-q{background_quota}",
            f"q{producer_quota}-q{background_quota}@{placement_variant}",
        }
        or selected.get("reserved_slack_us", -1) < 0
        or selected.get("uncovered_guard_us") != 0.0
        or selected.get("protection_scope", "producer") != gate_scope
    ):
        raise ValueError("QUIET plan differs from the requested execution contract")
    expected_payload = WORKLOAD_PAYLOAD_BYTES[workload]
    edges = selected.get("dag", {}).get("edges")
    if (
        not isinstance(edges, list)
        or len(edges) != 1
        or edges[0].get("payload_bytes") != expected_payload
        or {
            "registered-shared-sysmem-direct-binding": "registered-direct",
            "registered-direct": "registered-direct",
            "full-coherent-registered-system-memory": "registered-direct",
            "pinned-shared-sysmem-d2h-h2d": "pinned-bounce",
            "pinned-bounce": "pinned-bounce",
            "pageable-shared-sysmem-d2h-h2d": "pageable-bounce",
            "pageable-bounce": "pageable-bounce",
        }.get(edges[0].get("transport"))
        != transport
    ):
        raise ValueError("QUIET plan does not bind the coherent payload edge")
    return raw, {"path": str(path), "sha256": sha256(path)}


def quiet_plan_protection_scope(path: pathlib.Path) -> str:
    """Return the selected plan's scope before any GPU process is launched."""
    try:
        raw = json.loads(path.resolve().read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("QUIET plan is not readable JSON") from error
    selected = raw.get("selected_plan")
    scope = selected.get("protection_scope", "producer") if isinstance(selected, dict) else None
    if scope not in {"producer", "consumer", "pipeline"}:
        raise ValueError("QUIET plan has an invalid protection_scope")
    return scope


def validate_deadline_placement(
    lock: dict[str, Any], mig: dict[str, str], placement_variant: str
) -> None:
    contract = lock.get("contract", {})
    if lock.get("kind") == "p9-common-placement-deadline-lock":
        allowed = contract.get("allowed_placements", lock.get("allowed_placements", []))
        if placement_variant not in allowed:
            raise ValueError("common deadline lock does not allow requested placement")
        return
    placement = PLACEMENT_VARIANTS.get(placement_variant)
    if placement is None:
        raise ValueError("unknown placement variant")
    if (
        contract.get("producer_uuid") != mig[placement["producer_uuid_key"]]
        or contract.get("consumer_uuid") != mig[placement["consumer_uuid_key"]]
    ):
        raise ValueError(
            "deadline lock topology differs from the requested placement variant"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument(
        "--mig-env", type=pathlib.Path, default=pathlib.Path("/tmp/jdg-mps-1g/mig.env")
    )
    parser.add_argument("--result-dir", type=pathlib.Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--deadline-us", type=float, default=760.0)
    parser.add_argument(
        "--deadline-lock",
        type=pathlib.Path,
        help="verified actual-pipeline deadline lock; overrides --deadline-us",
    )
    parser.add_argument(
        "--background-period-ms",
        type=float,
        default=0.0,
        help="background request period; zero means saturated",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=tuple(item.name for item in SCENARIOS),
        help="run only this scenario; repeat to select multiple scenarios",
    )
    parser.add_argument("--producer-quota", type=int, default=100)
    parser.add_argument("--background-quota", type=int, default=100)
    parser.add_argument(
        "--no-background",
        action="store_true",
        help=(
            "disable the best-effort pressure client for a short correctness/"
            "latency sanity run; never treat this as an interference result"
        ),
    )
    parser.add_argument(
        "--placement-variant",
        choices=tuple(PLACEMENT_VARIANTS),
        default="fixed-1g-producer-2g-consumer",
        help="fixed MIG placement; each variant requires its own deadline lock",
    )
    parser.add_argument(
        "--workload",
        choices=("resnet-control", "resnet-detection-head", "resnet50-classification", "whisper-projection"),
        default="resnet-control",
    )
    parser.add_argument(
        "--consumer-engine",
        type=pathlib.Path,
        help="external serialized downstream TensorRT engine; omission uses generated control policy",
    )
    parser.add_argument(
        "--producer-engine",
        type=pathlib.Path,
        help="explicit serialized upstream TensorRT engine bound by the deadline lock",
    )
    parser.add_argument(
        "--consumer-input-tensor",
        default="features",
        help="input tensor name for --consumer-engine",
    )
    parser.add_argument(
        "--producer-input-trace",
        type=pathlib.Path,
        help="JDGINT1 fixed-size preprocessed producer input trace",
    )
    parser.add_argument(
        "--activation-replay-trace",
        type=pathlib.Path,
        help=(
            "JDGACT1 request-indexed producer activation trace; required for "
            "the independent arm"
        ),
    )
    parser.add_argument(
        "--common-workload-contract",
        type=pathlib.Path,
        help=(
            "newline-complete shared workload contract containing the common "
            "arrival trace and dataset-manifest paths/SHA-256 digests"
        ),
    )
    parser.add_argument(
        "--require-common-workload",
        action="store_true",
        help="fail unless --common-workload-contract is supplied and valid",
    )
    parser.add_argument(
        "--operational-arrival-trace", "--arrival-trace",
        dest="operational_arrival_trace", type=pathlib.Path,
        help="JDGARR1 operational release schedule consumed by the pipeline",
    )
    parser.add_argument(
        "--require-operational-arrival-trace", action="store_true",
        help="fail unless every pipeline arm consumes a JDGARR1 schedule",
    )
    parser.add_argument(
        "--dependency-mode",
        choices=("dependent", "independent"),
        default="dependent",
        help="keep the same stages while toggling only the producer-consumer edge",
    )
    parser.add_argument(
        "--transport",
        choices=("registered-direct", "pinned-bounce", "pageable-bounce"),
        default="registered-direct",
        help="host/device handoff path used by the production pipeline",
    )
    parser.add_argument(
        "--deadline-mode",
        choices=("wall", "validation-excluded"),
        default="wall",
        help=(
            "latency contract used for deadline classification; wall is the "
            "production-path default"
        ),
    )
    parser.add_argument(
        "--checksum-mode",
        choices=("inline", "sampled", "off"),
        default="inline",
        help="inline correctness, sampled audit, or off for production timing",
    )
    parser.add_argument("--checksum-sample-period", type=int, default=10)
    parser.add_argument(
        "--validation-delay-us", type=float, default=0.0,
        help="inject post-boundary correctness delay for protection validation tests",
    )
    parser.add_argument(
        "--application-output-trace-dir",
        type=pathlib.Path,
        help=(
            "optional directory for per-scenario raw TensorRT output traces; "
            "capture is post-completion and never part of the production wall"
        ),
    )
    parser.add_argument(
        "--quiet-gate-scope",
        choices=("producer", "consumer", "pipeline"),
        default=None,
        help=(
            "optional compatibility assertion for the selected QUIET plan; "
            "the plan's protection_scope is authoritative"
        ),
    )
    parser.add_argument(
        "--quiet-plan",
        type=pathlib.Path,
        help="selected stage-DAG/slack plan enforced for the QUIET scenario",
    )
    parser.add_argument(
        "--allow-plan-diagnostic",
        action="store_true",
        help=(
            "record a QUIET plan/slack violation as diagnostic-only evidence; "
            "never promotes the run to a formal result"
        ),
    )
    args = parser.parse_args()
    if args.quiet_gate_scope == "consumer" and args.dependency_mode != "dependent":
        raise ValueError("consumer protection scope requires dependent mode")
    if args.workload == "resnet-detection-head" and args.consumer_input_tensor == "features":
        args.consumer_input_tensor = "Layer6_relu_Y"
    if args.workload == "resnet50-classification" and args.consumer_input_tensor == "features":
        args.consumer_input_tensor = RESNET50_SPLIT_TENSOR
    if args.require_common_workload and args.common_workload_contract is None:
        raise ValueError("--require-common-workload requires --common-workload-contract")
    if args.require_operational_arrival_trace and args.operational_arrival_trace is None:
        raise ValueError(
            "--require-operational-arrival-trace requires --operational-arrival-trace"
        )
    if args.background_period_ms < 0.0:
        raise ValueError("background period must be nonnegative")
    if args.checksum_sample_period <= 0:
        raise ValueError("checksum sample period must be positive")
    if not math.isfinite(args.validation_delay_us) or args.validation_delay_us < 0.0:
        raise ValueError("validation delay must be nonnegative and finite")
    if args.workload == "resnet50-classification" and args.placement_variant != "fixed-1g-producer-2g-consumer":
        raise ValueError(
            "resnet50-classification currently requires the profiled "
            "1g-producer/2g-consumer split"
        )
    for label, quota in (
        ("producer", args.producer_quota),
        ("background", args.background_quota),
    ):
        if quota not in {10, 25, 50, 75, 90, 100}:
            raise ValueError(f"{label} quota lacks a profiled engine")
    repo = args.repo.resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from runtime.quiet_stage_dag import actuate_selected_plan

    deadline_lock_provenance: dict[str, str] | None = None
    if args.deadline_lock is not None:
        lock_path = args.deadline_lock.resolve()
        lock_header = json.loads(lock_path.read_text(encoding="utf-8"))
        verifier = (
            repo / "analysis" / "freeze_p9_common_placement_deadline.py"
            if lock_header.get("kind") == "p9-common-placement-deadline-lock"
            else repo / "analysis" / "freeze_p9_pipeline_deadline.py"
        )
        subprocess.run(
            [
                "python3",
                str(verifier),
                "--verify",
                str(lock_path),
            ],
            cwd=repo,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if (
            lock.get("kind") not in {
                "p9-dependent-pipeline-deadline-lock",
                "p9-common-placement-deadline-lock",
            }
            or lock.get("contract", {}).get("workload") != args.workload
        ):
            raise ValueError("deadline lock differs from requested workload")
        args.deadline_us = float(lock["deadline_us"])
        deadline_lock_provenance = {
            "path": str(lock_path),
            "sha256": sha256(lock_path),
        }
    quiet_plan: dict[str, Any] | None = None
    quiet_plan_provenance: dict[str, str] | None = None
    if args.quiet_plan is not None:
        plan_scope = quiet_plan_protection_scope(args.quiet_plan)
        if args.quiet_gate_scope is not None and args.quiet_gate_scope != plan_scope:
            raise ValueError("--quiet-gate-scope differs from the selected QUIET plan")
        args.quiet_gate_scope = plan_scope
        if args.quiet_gate_scope == "consumer" and args.dependency_mode != "dependent":
            raise ValueError("consumer protection scope requires dependent mode")
        quiet_plan, quiet_plan_provenance = validate_quiet_plan(
            args.quiet_plan,
            args.deadline_us,
            deadline_lock_provenance,
            args.producer_quota,
            args.background_quota,
            args.quiet_gate_scope,
            args.workload,
            args.placement_variant,
            args.transport,
        )
    elif args.quiet_gate_scope is None:
        args.quiet_gate_scope = "producer"
    quiet_plan_violation: dict[str, float] | None = None
    mig = load_env(args.mig_env)
    if args.deadline_lock is not None:
        validate_deadline_placement(lock, mig, args.placement_variant)
    quiet_execution: dict[str, Any] | None = None
    if quiet_plan is not None:
        quiet_execution = actuate_selected_plan(
            quiet_plan,
            mig,
            args.placement_variant,
            args.transport,
            args.producer_quota,
            args.background_quota,
            args.quiet_gate_scope,
            args.workload,
            args.dependency_mode,
        )
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if quiet_execution is not None:
        (args.result_dir / "quiet-execution.json").write_text(
            json.dumps(quiet_execution, indent=2) + "\n", encoding="utf-8"
        )
    application_output_trace_dir = (
        args.application_output_trace_dir.resolve()
        if args.application_output_trace_dir is not None
        else None
    )
    if application_output_trace_dir is not None:
        application_output_trace_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.mig_env, args.result_dir / "mig.env")
    inventory = subprocess.run(
        ["nvidia-smi", "-L"], check=True, capture_output=True, text=True
    ).stdout
    (args.result_dir / "gpu-inventory.txt").write_text(
        inventory, encoding="utf-8"
    )
    producer_model = (
        "resnet10-detection"
        if args.workload == "resnet-control"
        else "resnet10-backbone"
        if args.workload == "resnet-detection-head"
        else "resnet50-backbone"
        if args.workload == "resnet50-classification"
        else "whisper-tiny-encoder"
    )
    placement = PLACEMENT_VARIANTS[args.placement_variant]
    common_workload: dict[str, Any] | None = None
    if args.common_workload_contract is not None:
        common_workload = load_common_workload_contract(
            args.common_workload_contract,
            workload=args.workload,
            placement=args.placement_variant,
            input_tensor=args.consumer_input_tensor,
            payload_bytes=WORKLOAD_PAYLOAD_BYTES[args.workload],
        )
    producer_engine = (
        args.producer_engine.resolve()
        if args.producer_engine is not None
        else repo / (
            f"models/engines/mig-{placement['producer_slice']}-q{args.producer_quota}"
            f"/{producer_model}.engine"
        )
    )
    if not producer_engine.is_file():
        raise ValueError(
            f"{args.placement_variant} lacks a profiled producer engine for "
            f"q{args.producer_quota}: {producer_engine}"
        )
    consumer_engine = args.consumer_engine.resolve() if args.consumer_engine else None
    if consumer_engine is not None and not consumer_engine.is_file():
        raise ValueError(f"--consumer-engine is not a regular file: {consumer_engine}")
    if args.deadline_lock is not None:
        from analysis.verify_p9_runtime_lock import verify_runtime_binding

        verify_runtime_binding(
            args.deadline_lock,
            repo,
            producer_engine,
            consumer_engine,
        )
    producer_input_trace = (
        args.producer_input_trace.resolve()
        if args.producer_input_trace is not None else None
    )
    if producer_input_trace is not None and not producer_input_trace.is_file():
        raise ValueError("--producer-input-trace is not a regular file")
    activation_replay_trace = (
        args.activation_replay_trace.resolve()
        if args.activation_replay_trace is not None else None
    )
    if args.dependency_mode == "independent" and activation_replay_trace is None:
        raise ValueError("independent mode requires --activation-replay-trace")
    if activation_replay_trace is not None and not activation_replay_trace.is_file():
        raise ValueError("--activation-replay-trace is not a regular file")
    operational_arrival_trace = (
        args.operational_arrival_trace.resolve()
        if args.operational_arrival_trace is not None else None
    )
    if operational_arrival_trace is not None and not operational_arrival_trace.is_file():
        raise ValueError("--operational-arrival-trace is not a regular file")
    if args.require_operational_arrival_trace and operational_arrival_trace is None:
        raise ValueError("operational arrival trace is required")
    if common_workload is not None and "operational_arrival_trace_path" in common_workload:
        if (
            operational_arrival_trace is None
            or str(operational_arrival_trace)
            != common_workload["operational_arrival_trace_path"]
            or sha256(operational_arrival_trace)
            != common_workload["operational_arrival_trace_sha256"]
        ):
            raise ValueError("operational arrival trace differs from common workload contract")
    if common_workload is not None and "producer_input_trace_path" in common_workload:
        if (
            producer_input_trace is None
            or str(producer_input_trace) != common_workload["producer_input_trace_path"]
            or sha256(producer_input_trace) != common_workload["producer_input_trace_sha256"]
        ):
            raise ValueError("producer input trace differs from common workload contract")
    background_engine = (
        repo
        / f"models/engines/mig-1g-q{args.background_quota}/distilbert-sst2.engine"
    )
    provenance_paths = (
        repo / "build-r39/jdg-mig-trt-pipeline",
        repo / "build-r39/jdg-trt-bench",
        repo / "benchmarks/mig_trt_pipeline.cpp",
        repo / "benchmarks/trt_inference.cpp",
        producer_engine,
        background_engine,
    )
    if consumer_engine is not None:
        provenance_paths = provenance_paths + (consumer_engine,)
    if producer_input_trace is not None:
        provenance_paths = provenance_paths + (producer_input_trace,)
    if activation_replay_trace is not None:
        provenance_paths = provenance_paths + (activation_replay_trace,)
    if operational_arrival_trace is not None:
        provenance_paths = provenance_paths + (operational_arrival_trace,)
    if args.quiet_plan is not None:
        provenance_paths = provenance_paths + (args.quiet_plan.resolve(),)
    provenance: dict[str, str] = {}
    for path in provenance_paths:
        try:
            key = str(path.relative_to(repo))
        except ValueError:
            # External immutable traces are allowed for directional probes;
            # retain their absolute path so provenance is not silently lost.
            key = str(path)
        provenance[key] = sha256(path)
    (args.result_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    rows: list[dict[str, Any]] = []

    by_name = {item.name: item for item in SCENARIOS}
    if args.scenario:
        if len(set(args.scenario)) != len(args.scenario):
            raise ValueError("scenario order contains duplicates")
        selected_scenarios = tuple(by_name[name] for name in args.scenario)
    else:
        selected_scenarios = DEFAULT_SCENARIOS
    for scenario in selected_scenarios:
        effective_scenario = Scenario(
            scenario.name,
            scenario.same_instance,
            scenario.gate_mode,
            scenario.gate_scope,
            scenario.best_effort_admitted and not args.no_background,
        )
        scenario_dir = args.result_dir / scenario.name
        scenario_dir.mkdir()
        trace_csv = scenario_dir / "pipeline.csv"
        event_trace_csv = scenario_dir / "events.csv"
        application_output_trace = None
        if application_output_trace_dir is not None:
            application_output_trace = (
                application_output_trace_dir / scenario.name / "outputs.bin"
            )
            application_output_trace.parent.mkdir(parents=True, exist_ok=True)
        scenario_execution = (
            quiet_execution if scenario.name == "quiet" else None
        )
        effective_producer_quota = (
            int(scenario_execution["producer_quota_percent"])
            if scenario_execution is not None
            else args.producer_quota
        )
        effective_background_quota = (
            int(scenario_execution["background_quota_percent"])
            if scenario_execution is not None
            else args.background_quota
        )
        effective_transport = (
            str(scenario_execution["transport"])
            if scenario_execution is not None
            else args.transport
        )
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": mig["JDG_MIG_SMALL_UUID"],
                "CUDA_MPS_PIPE_DIRECTORY": mig["JDG_MPS_PIPE_DIRECTORY"],
                "CUDA_MPS_LOG_DIRECTORY": mig["JDG_MPS_LOG_DIRECTORY"],
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(
                    effective_background_quota
                ),
            }
        )
        background: subprocess.Popen[str] | None = None
        if effective_scenario.best_effort_admitted:
            background = subprocess.Popen(
                [
                "taskset",
                "--cpu-list",
                "0",
                str(repo / "build-r39" / "jdg-trt-bench"),
                "--engine",
                str(background_engine),
                "--model-name",
                "distilbert-sst2",
                "--role",
                "pressure",
                "--duration-seconds",
                "3600",
                "--burst-size",
                "1",
                "--period-ms",
                str(args.background_period_ms),
                "--warmup",
                str(args.warmup),
                "--include-transfers",
                "true",
                "--priority",
                "default",
                "--start-paused",
                "true",
                ],
                cwd=repo,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        try:
            if background is not None:
                wait_paused(background)
                effective_scope = (
                    args.quiet_gate_scope
                    if scenario.name == "quiet"
                    else scenario.gate_scope
                )
                if scenario.gate_mode is None or effective_scope == "consumer":
                    os.kill(background.pid, signal.SIGCONT)
            producer_uuid = mig[placement["producer_uuid_key"]]
            consumer_uuid = mig[placement["consumer_uuid_key"]]
            if scenario.same_instance:
                producer_uuid = consumer_uuid = mig["JDG_MIG_SMALL_UUID"]
            if scenario_execution is not None:
                producer_uuid = str(scenario_execution["producer_uuid"])
                consumer_uuid = str(scenario_execution["consumer_uuid"])
            command = [
                "taskset",
                "--cpu-list",
                "13",
                str(repo / "build-r39" / "jdg-mig-trt-pipeline"),
                "--producer-engine",
                str(producer_engine),
                *(["--consumer-engine", str(consumer_engine)] if consumer_engine else []),
                "--consumer-input-tensor",
                args.consumer_input_tensor,
                "--producer",
                producer_uuid,
                "--consumer",
                consumer_uuid,
                "--transport",
                effective_transport,
                "--warmup",
                str(args.warmup),
                "--iterations",
                str(args.iterations),
                "--deadline-us",
                str(args.deadline_us),
                "--workload",
                args.workload,
                "--dependency-mode",
                args.dependency_mode,
                "--deadline-mode",
                args.deadline_mode,
                "--checksum-mode",
                args.checksum_mode,
                "--checksum-sample-period",
                str(args.checksum_sample_period),
                "--validation-delay-us",
                str(args.validation_delay_us),
                "--producer-quota",
                str(effective_producer_quota),
                "--consumer-quota",
                str(
                    scenario_execution["consumer_quota_percent"]
                    if scenario_execution is not None
                    else 100
                ),
                "--trace-csv",
                str(trace_csv),
                "--event-trace-csv",
                str(event_trace_csv),
            ]
            if producer_input_trace is not None:
                command.extend(("--producer-input-trace", str(producer_input_trace)))
            if activation_replay_trace is not None:
                command.extend(("--activation-replay-trace", str(activation_replay_trace)))
            if operational_arrival_trace is not None:
                command.extend(("--arrival-trace", str(operational_arrival_trace)))
            if application_output_trace is not None:
                command.extend((
                    "--application-output-trace",
                    str(application_output_trace),
                ))
            if producer_uuid == mig["JDG_MIG_SMALL_UUID"]:
                command.extend(("--producer-mps-pipe", mig["JDG_MPS_PIPE_DIRECTORY"]))
            if consumer_uuid == mig["JDG_MIG_SMALL_UUID"]:
                command.extend(
                    ("--consumer-mps-pipe", mig["JDG_MPS_PIPE_DIRECTORY"])
                )
            if scenario.gate_mode is not None and background is not None:
                command.extend(
                    (
                        "--gate-pids",
                        str(background.pid),
                        "--gate-mode",
                        scenario.gate_mode,
                    )
                )
                command.extend(
                    (
                        "--gate-scope",
                        effective_scope,
                    )
                )
            completed = subprocess.run(
                command,
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
            )
            pipeline = json.loads(completed.stdout)
            if args.require_operational_arrival_trace:
                if pipeline.get("arrival_schedule_mode") != "operational-trace":
                    raise RuntimeError("pipeline did not consume the operational arrival trace")
                if not event_trace_csv.is_file():
                    raise RuntimeError("pipeline did not produce its event trace")
            if application_output_trace is not None and not application_output_trace.is_file():
                raise RuntimeError("benchmark did not produce application output trace")
            (scenario_dir / "pipeline.json").write_text(
                json.dumps(pipeline, indent=2) + "\n", encoding="utf-8"
            )
            (scenario_dir / "pipeline.stderr").write_text(
                completed.stderr, encoding="utf-8"
            )
            if scenario.name == "quiet" and quiet_plan is not None:
                selected = quiet_plan["selected_plan"]
                actual_p99 = float(pipeline["end_to_end_us"]["p99"])
                actual_gate_p99 = float(pipeline["gate_us"]["p99"])
                if (
                    actual_p99 > args.deadline_us
                    or actual_gate_p99 > float(selected["critical_lookahead_us"])
                ):
                    quiet_plan_violation = {
                        "observed_end_to_end_p99_us": actual_p99,
                        "deadline_us": args.deadline_us,
                        "observed_gate_p99_us": actual_gate_p99,
                        "critical_lookahead_us": float(
                            selected["critical_lookahead_us"]
                        ),
                    }
                    if not args.allow_plan_diagnostic:
                        raise RuntimeError(
                            "held-out QUIET execution violates its slack plan"
                        )
        finally:
            if background is not None:
                background_stdout, background_stderr = stop(background)
            else:
                background_stdout, background_stderr = "", ""
            (scenario_dir / "background.stderr").write_text(
                background_stderr, encoding="utf-8"
            )
        if background is not None and background.returncode != 0:
            raise RuntimeError(
                f"{scenario.name} background failed ({background.returncode})"
            )
        background_result = (
            json.loads(background_stdout)
            if background is not None
            else {
                "status": "rejected-no-best-effort-slice",
                "throughput_per_second": 0.0,
                "completed": 0,
            }
        )
        (scenario_dir / "background.json").write_text(
            json.dumps(background_result, indent=2) + "\n", encoding="utf-8"
        )
        rows.append(
            summarize(
                effective_scenario,
                pipeline,
                background_result,
                args.producer_quota,
                args.background_quota,
                trace_csv,
                args.placement_variant,
                application_output_trace,
                event_trace_csv,
            )
        )

    output = {
        "schema_version": 1,
        "kind": "p9-dependent-small-stress-smoke",
        "deadline_us": args.deadline_us,
        "iterations": args.iterations,
        "background_period_ms": args.background_period_ms,
        "background_disabled": args.no_background,
        "background_offered_rps": (
            None
            if args.background_period_ms == 0.0
            else 1000.0 / args.background_period_ms
        ),
        "producer_quota_percent": args.producer_quota,
        "background_quota_percent": args.background_quota,
        "best_effort_admitted": {
            item.name: item.best_effort_admitted and not args.no_background
            for item in selected_scenarios
        },
        "workload": args.workload,
        "common_workload": common_workload,
        "dependency_mode": args.dependency_mode,
        "operational_arrival_trace": (
            {"path": str(operational_arrival_trace), "sha256": sha256(operational_arrival_trace)}
            if operational_arrival_trace is not None else None
        ),
        "transport": args.transport,
        "latency_contract": (
            "production-wall-arrival-to-completion"
            if args.deadline_mode == "wall"
            else "audit-validation-excluded"
        ),
        "deadline_mode": args.deadline_mode,
        "production_wall_definition": (
            "arrival-to-consumer-completion-excludes-correctness-validation"
            if args.deadline_mode == "wall"
            else "legacy-validation-excluded"
        ),
        "correctness_validation_placement": "post-completion"
        if args.checksum_mode != "off"
        else "disabled",
        "checksum_mode": args.checksum_mode,
        "checksum_sample_period": args.checksum_sample_period,
        "quiet_gate_scope": args.quiet_gate_scope,
        "placement_variant": args.placement_variant,
        "consumer_engine_mode": (
            "external-trained-engine" if consumer_engine is not None
            else "generated-control-policy"
        ),
        "consumer_input_tensor": args.consumer_input_tensor,
        "consumer_engine": (
            {"path": str(consumer_engine), "sha256": sha256(consumer_engine)}
            if consumer_engine is not None else None
        ),
        "execution_order": [PUBLIC_SYSTEM_NAMES[item.name] for item in selected_scenarios],
        "deadline_source": (
            "frozen-independent-pipeline-p99-factor"
            if deadline_lock_provenance is not None
            else "exploratory-command-line"
        ),
        "deadline_lock": deadline_lock_provenance,
        "quiet_plan": quiet_plan_provenance,
        "quiet_execution": quiet_execution,
        "quiet_plan_violation": quiet_plan_violation,
        "claim_status": (
            "diagnostic-only-plan-violation"
            if quiet_plan_violation is not None
            else "exploratory-contract-smoke"
        ),
        "application_output_trace_dir": (
            str(application_output_trace_dir)
            if application_output_trace_dir is not None else None
        ),
        "results": rows,
    }
    (args.result_dir / "summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
