#!/usr/bin/env python3
"""Run one frozen Williams sequence of common-workload P9 systems.

Use ``--active-only`` for the current exploratory MPS/XSched/QUIET matrix;
without it, the legacy six-row sequence is retained for historical replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path
from typing import Any, Iterable


SYSTEMS = ("NVIDIA MIG", "NVIDIA MPS", "Orion", "XSched", "gpulet", "QUIET")
ACTIVE_SYSTEMS = ("NVIDIA MPS", "XSched", "QUIET")
NUMERIC_FRONTIER_SYSTEMS = ("NVIDIA MPS", "QUIET")
PRODUCTION_WALL_DEFINITION = (
    "arrival-to-consumer-completion-excludes-correctness-validation"
)
CORRECTNESS_PLACEMENT = "post-completion"
JDGINT_MAGIC = b"JDGINT1\x00"
JDGINT_HEADER = struct.Struct("<IIQ")
SCENARIOS = {
    "NVIDIA MIG": "nvidia-mig-isolation",
    "NVIDIA MPS": "nvidia-mps-spatial-sharing",
    "QUIET": "quiet",
}

# Explicit topology/fairness metadata prevents a functional or infeasible
# control from silently becoming a ranked same-SLO competitor.
COMPARISON_CONTRACTS = {
    "NVIDIA MIG": {
        "topology": "fixed-2g+1g-physical-isolation",
        "numeric_comparison_allowed": False,
        "comparison_status": "capacity-control-no-best-effort-slice",
    },
    "NVIDIA MPS": {
        "topology": "1g-shared-MPS-with-2g-reserved",
        "numeric_comparison_allowed": True,
        "comparison_status": "executable-baseline",
    },
    "Orion": {
        "topology": "fixed-2g+1g-native-interposition-port",
        "numeric_comparison_allowed": False,
        "comparison_status": "faithful-port-pending-differential-gate",
    },
    "XSched": {
        "topology": "fixed-2g+1g-native-xqueue-port",
        "numeric_comparison_allowed": False,
        "comparison_status": "native-runtime-verified-common-workload-accuracy-pending",
    },
    "gpulet": {
        "topology": "fixed-2g+1g-local-planner",
        "numeric_comparison_allowed": False,
        "comparison_status": "structural-only-until-spatial-search-feasible",
    },
    "QUIET": {
        "topology": "fixed-2g+1g-dependent-dag-quiescence",
        "numeric_comparison_allowed": True,
        "comparison_status": "proposed-system",
    },
}


def decorate_row(row: dict[str, Any], system: str) -> dict[str, Any]:
    row.update(COMPARISON_CONTRACTS[system])
    row["comparison_contract_version"] = 1
    return row


def require_active_production_wall(row: dict[str, Any], system: str) -> None:
    """Reject legacy timing contracts before they enter the active matrix.

    The active matrix is the only path that can later feed a same-SLO
    frontier.  A validation-excluded or pre-v2 row is useful historical
    evidence, but mixing it with production-wall rows would make the
    comparison silently measure different intervals.
    """
    if system not in ACTIVE_SYSTEMS:
        return
    expected = {
        "deadline_mode": "wall",
        "latency_contract": "production-wall-arrival-to-completion",
        "production_wall_definition": PRODUCTION_WALL_DEFINITION,
        "correctness_validation_placement": CORRECTNESS_PLACEMENT,
    }
    mismatches = [
        f"{key}={row.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if row.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"{system} evidence is not the active production-wall contract: "
            + ", ".join(mismatches)
        )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def common_operational_arrival_trace(path: Path) -> Path | None:
    """Resolve the operational release trace bound by a common contract."""
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    trace_path = value.get("operational_arrival_trace_path")
    expected = value.get("operational_arrival_trace_sha256")
    if trace_path is None and expected is None:
        return None
    if not isinstance(trace_path, str) or not trace_path:
        raise ValueError("common workload operational arrival path is invalid")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("common workload operational arrival SHA is invalid")
    resolved = Path(trace_path).resolve()
    if not resolved.is_file() or sha256(resolved) != expected:
        raise ValueError("common workload operational arrival trace SHA differs")
    return resolved


def producer_input_trace_count(path: Path) -> int:
    """Read the fixed JDGINT1 record count before launching a comparator."""
    raw = path.read_bytes()
    header_offset = len(JDGINT_MAGIC)
    if len(raw) < header_offset + JDGINT_HEADER.size:
        raise ValueError("producer input trace is truncated")
    if raw[:header_offset] != JDGINT_MAGIC:
        raise ValueError("producer input trace magic differs")
    schema, count, sample_bytes = JDGINT_HEADER.unpack_from(raw, header_offset)
    if schema != 1 or count <= 0 or sample_bytes <= 0:
        raise ValueError("producer input trace header is invalid")
    return count


def williams_orders() -> tuple[tuple[str, ...], ...]:
    return _williams_orders(SYSTEMS)


def active_williams_orders() -> tuple[tuple[str, ...], ...]:
    """Return the frozen three-row exploratory design.

    Historical six-row orders remain available for raw replay, but they mix
    capacity/functional controls with numeric rows. The active design is
    intentionally limited to the manifest's MPS, XSched, and QUIET rows;
    only MPS and QUIET are numeric-frontier eligible.
    """
    return _williams_orders(ACTIVE_SYSTEMS)


def _williams_orders(systems: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    count = len(systems)
    base = [0]
    for offset in range(1, count // 2 + 1):
        base.append(offset)
        if len(base) < count:
            base.append(count - offset)
    return tuple(
        tuple(systems[(index + rotation) % count] for index in base[:count])
        for rotation in range(count)
    )


def run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(
        command, cwd=cwd, env=env, check=True,
        stdout=subprocess.DEVNULL,
    )


def base_row(
    repo: Path, output: Path, system: str, lock: Path, plan: Path, requests: int,
    workload: str, background_period_ms: float, placement_variant: str,
    common_workload_contract: Path | None = None,
    consumer_engine: Path | None = None,
    producer_input_trace: Path | None = None,
    producer_engine: Path | None = None,
    warmup: int = 10,
) -> tuple[dict[str, Any], Path]:
    command = [
        "python3", str(repo / "scripts/run_p9_dependent_stress_smoke.py"),
        "--repo", str(repo), "--result-dir", str(output),
        "--iterations", str(requests), "--warmup", str(warmup),
        "--deadline-lock", str(lock), "--background-period-ms", str(background_period_ms),
        "--workload", workload, "--scenario", SCENARIOS[system],
        "--placement-variant", placement_variant,
        "--application-output-trace-dir", str(output / "application-outputs"),
    ]
    if system == "QUIET":
        command.extend(("--quiet-plan", str(plan)))
    if consumer_engine is not None:
        command.extend(("--consumer-engine", str(consumer_engine)))
    if producer_engine is not None:
        command.extend(("--producer-engine", str(producer_engine)))
    if producer_input_trace is not None:
        command.extend(("--producer-input-trace", str(producer_input_trace)))
    if common_workload_contract is not None:
        operational_arrival_trace = common_operational_arrival_trace(
            common_workload_contract
        )
        command.extend((
            "--common-workload-contract", str(common_workload_contract),
            "--require-common-workload",
        ))
        if operational_arrival_trace is not None:
            command.extend((
                "--operational-arrival-trace", str(operational_arrival_trace),
                "--require-operational-arrival-trace",
            ))
    run(command, cwd=repo)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = summary.get("results")
    if not isinstance(rows, list) or len(rows) != 1 or rows[0].get("system") != system:
        raise ValueError(f"{system} result contract differs")
    row = rows[0]
    decorated = decorate_row({
        "system": system,
        "requests": row["pipeline_requests"],
        "misses": row["deadline_misses"],
        "p99_us": row["pipeline_p99_us"],
        "background_goodput_rps": row["background_goodput_rps"],
        "deadline_mode": row.get("deadline_mode"),
        "correctness_validated": row.get("correctness_validated", True),
        "latency_contract": row.get("latency_contract"),
        "production_wall_definition": row.get("production_wall_definition"),
        "correctness_validation_placement": row.get("correctness_validation_placement"),
    }, system)
    require_active_production_wall(decorated, system)
    return decorated, summary_path


def orion_row(
    repo: Path, output: Path, lock: Path, requests: int, workload: str,
) -> tuple[dict[str, Any], Path]:
    env = dict(
        os.environ, RESULT_DIR=str(output), DEADLINE_LOCK=str(lock),
        CRITICAL_REQUESTS=str(requests),
    )
    script = (
        "run_p9_orion_resnet_control_smoke.sh"
        if workload == "resnet-control"
        else "run_p9_orion_dependent_smoke.sh"
    )
    run(["bash", str(repo / f"scripts/{script}")], cwd=repo, env=env)
    path = output / "verification.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    return decorate_row({
        "system": "Orion", "requests": value["requests"], "misses": value["misses"],
        "p99_us": value["p99_us"], "background_goodput_rps": value["background_goodput_rps"],
        "deadline_mode": value.get("deadline_mode"),
        "correctness_validated": value.get("correctness_validated", True),
    }, "Orion"), path


def xsched_row(
    repo: Path, output: Path, lock: Path, requests: int, workload: str,
    common_workload_contract: Path | None = None,
    consumer_engine: Path | None = None,
    producer_input_trace: Path | None = None,
    producer_engine: Path | None = None,
    warmup: int = 20,
) -> tuple[dict[str, Any], Path]:
    env = dict(
        os.environ, RESULT_DIR=str(output), DEADLINE_LOCK=str(lock),
        CRITICAL_REQUESTS=str(requests), BE_REQUESTS="5000",
        APPLICATION_OUTPUT_TRACE=str(output / "application-outputs.bin"),
    )
    if common_workload_contract is not None:
        env["COMMON_WORKLOAD_CONTRACT"] = str(common_workload_contract)
        env["REQUIRE_COMMON_WORKLOAD"] = "1"
        operational_arrival_trace = common_operational_arrival_trace(
            common_workload_contract
        )
        if operational_arrival_trace is not None:
            env["OPERATIONAL_ARRIVAL_TRACE"] = str(operational_arrival_trace)
    if consumer_engine is not None:
        env["CONSUMER_ENGINE"] = str(consumer_engine)
    if producer_engine is not None:
        env["PRODUCER_ENGINE"] = str(producer_engine)
    if producer_input_trace is not None:
        env["PRODUCER_INPUT_TRACE"] = str(producer_input_trace)
    env["WARMUP"] = str(warmup)
    script = (
        "run_p9_xsched_resnet_control_smoke.sh"
        if workload == "resnet-control"
        else "run_p9_xsched_dependent_smoke.sh"
    )
    run([str(repo / f"scripts/{script}")], cwd=repo, env=env)
    path = output / "verification.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    background_goodput = value.get("background_goodput_rps")
    if background_goodput is None:
        background_goodput = value["background_window"]["completion_goodput_rps"]
    decorated = decorate_row({
        "system": "XSched", "requests": value["requests"], "misses": value["misses"],
        "p99_us": value["p99_us"], "background_goodput_rps": background_goodput,
        "deadline_mode": value.get("deadline_mode"),
        "correctness_validated": value.get("correctness_validated", True),
        "latency_contract": value.get("latency_contract"),
        "production_wall_definition": value.get("production_wall_definition"),
        "correctness_validation_placement": value.get("correctness_validation_placement"),
    }, "XSched")
    require_active_production_wall(decorated, "XSched")
    return decorated, path


def gpulet_row(
    repo: Path, source: Path, output: Path, lock: Path,
    profile_requests: int, evaluation_requests: int, workload: str,
    background_period_ms: float,
) -> tuple[dict[str, Any], Path]:
    run([
        "python3", str(repo / "baselines/gpulet/run_thor.py"),
        "--repo", str(repo), "--source", str(source), "--result-dir", str(output),
        "--deadline-lock", str(lock), "--background-period-ms", str(background_period_ms),
        "--profile-iterations", str(profile_requests),
        "--evaluation-iterations", str(evaluation_requests),
        "--workload", workload,
    ], cwd=repo)
    path = output / "verification.json"
    run([
        "python3", str(repo / "baselines/gpulet/verify_resnet_control_smoke.py"),
        "--result-dir", str(output), "--output", str(path),
        "--expected-requests", str(evaluation_requests),
    ], cwd=repo)
    value = json.loads(path.read_text(encoding="utf-8"))
    return decorate_row({
        "system": "gpulet", "requests": value["requests"], "misses": value["misses"],
        "p99_us": value["p99_us"], "background_goodput_rps": value["background_goodput_rps"],
        "deadline_mode": value.get("deadline_mode", "wall"),
        "correctness_validated": value.get("correctness_validated", True),
        "spatial_schedule_feasible": value["spatial_schedule_feasible"],
    }, "gpulet"), path


def existing_row(output: Path, system: str) -> tuple[dict[str, Any], Path]:
    path = output / ("summary.json" if system in SCENARIOS else "verification.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    if system in SCENARIOS:
        rows = value.get("results")
        if not isinstance(rows, list) or len(rows) != 1 or rows[0].get("system") != system:
            raise ValueError(f"{system} existing result contract differs")
        source = rows[0]
        row = {
            "system": system,
            "requests": source["pipeline_requests"],
            "misses": source["deadline_misses"],
            "p99_us": source["pipeline_p99_us"],
            "background_goodput_rps": source["background_goodput_rps"],
            "deadline_mode": source.get("deadline_mode"),
            "correctness_validated": source.get("correctness_validated", True),
            "latency_contract": source.get("latency_contract"),
            "production_wall_definition": source.get("production_wall_definition"),
            "correctness_validation_placement": source.get("correctness_validation_placement"),
        }
    else:
        goodput = value.get("background_goodput_rps")
        if goodput is None:
            goodput = value["background_window"]["completion_goodput_rps"]
        row = {
            "system": system,
            "requests": value["requests"],
            "misses": value["misses"],
            "p99_us": value["p99_us"],
            "background_goodput_rps": goodput,
            "deadline_mode": value.get("deadline_mode"),
            "correctness_validated": value.get("correctness_validated", True),
            "latency_contract": value.get("latency_contract"),
            "production_wall_definition": value.get("production_wall_definition"),
            "correctness_validation_placement": value.get("correctness_validation_placement"),
        }
        if "spatial_schedule_feasible" in value:
            row["spatial_schedule_feasible"] = value["spatial_schedule_feasible"]
    if row["requests"] <= 0:
        raise ValueError(f"{system} existing evidence has no requests")
    decorated = decorate_row(row, system)
    require_active_production_wall(decorated, system)
    return decorated, path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--deadline-lock", type=Path, required=True)
    parser.add_argument("--quiet-plan", type=Path, required=True)
    parser.add_argument("--gpulet-source", type=Path)
    parser.add_argument("--sequence-index", type=int, required=True)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--gpulet-profile-requests", type=int, default=100)
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="warmup requests; learned JDGINT1 traces must contain warmup+requests records",
    )
    parser.add_argument(
        "--workload", choices=("resnet-control", "resnet-detection-head", "resnet50-classification", "whisper-projection"),
        default="resnet-control",
    )
    parser.add_argument("--background-period-ms", type=float, default=4.0)
    parser.add_argument(
        "--placement-variant",
        choices=("fixed-1g-producer-2g-consumer", "fixed-2g-producer-1g-consumer"),
        default="fixed-1g-producer-2g-consumer",
        help="placement to bind for every active system in this sequence",
    )
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="run only the exploratory MPS/XSched/QUIET matrix",
    )
    parser.add_argument("--reuse-existing-evidence", action="store_true")
    parser.add_argument("--common-workload-contract", type=Path)
    parser.add_argument("--consumer-engine", type=Path)
    parser.add_argument("--producer-input-trace", type=Path)
    parser.add_argument("--producer-engine", type=Path)
    args = parser.parse_args(argv)
    repo, root = args.repo.resolve(), args.result_dir.resolve()
    lock, plan = args.deadline_lock.resolve(), args.quiet_plan.resolve()
    common_workload_contract = (
        args.common_workload_contract.resolve()
        if args.common_workload_contract is not None else None
    )
    consumer_engine = args.consumer_engine.resolve() if args.consumer_engine else None
    if consumer_engine is not None and not consumer_engine.is_file():
        raise ValueError("consumer engine is missing")
    producer_engine = args.producer_engine.resolve() if args.producer_engine else None
    if producer_engine is not None and not producer_engine.is_file():
        raise ValueError("producer engine is missing")
    producer_input_trace = (
        args.producer_input_trace.resolve()
        if args.producer_input_trace is not None else None
    )
    if producer_input_trace is not None and not producer_input_trace.is_file():
        raise ValueError("producer input trace is missing")
    if args.active_only and args.workload in {"resnet-detection-head", "resnet50-classification"} and consumer_engine is None:
        raise ValueError("learned ResNet workloads require --consumer-engine")
    if args.active_only and args.workload == "resnet50-classification" and args.placement_variant != "fixed-1g-producer-2g-consumer":
        raise ValueError("resnet50-classification currently requires the profiled 1g-producer/2g-consumer split")
    if common_workload_contract is not None and not common_workload_contract.is_file():
        raise ValueError("common workload contract is missing")
    if common_workload_contract is not None and not args.active_only:
        raise ValueError(
            "common workload contract is supported only by the active MPS/XSched/QUIET sequence"
        )
    if args.active_only and args.workload in {"resnet-detection-head", "resnet50-classification"}:
        if common_workload_contract is None:
            raise ValueError(
                "active learned workload requires --common-workload-contract"
            )
        if producer_input_trace is None:
            raise ValueError(
                "active learned workload requires --producer-input-trace"
            )
        contract = json.loads(common_workload_contract.read_text(encoding="utf-8"))
        if (
            contract.get("producer_input_trace_path") != str(producer_input_trace)
            or contract.get("producer_input_trace_sha256") != sha256(producer_input_trace)
        ):
            raise ValueError("producer input trace differs from common workload contract")
        if contract.get("request_count") != args.requests:
            raise ValueError("request count differs from common workload contract")
        if producer_input_trace_count(producer_input_trace) != args.warmup + args.requests:
            raise ValueError(
                "producer input trace count differs from warmup plus requests"
            )
    gpulet_source = args.gpulet_source.resolve() if args.gpulet_source else None
    if not args.active_only and gpulet_source is None:
        raise ValueError("--gpulet-source is required for the historical six-row mode")
    orders = active_williams_orders() if args.active_only else williams_orders()
    if (args.requests <= 0 or args.warmup < 0 or args.gpulet_profile_requests <= 0 or
            args.background_period_ms <= 0):
        raise ValueError("request counts and warmup are invalid")
    if args.active_only and args.placement_variant != "fixed-1g-producer-2g-consumer":
        raise ValueError(
            "active XSched contract currently supports only the forward "
            "1g-producer/2g-consumer placement"
        )
    if not 0 <= args.sequence_index < len(orders):
        raise ValueError("sequence index is outside the Williams design")
    if root.exists() and not args.reuse_existing_evidence:
        raise ValueError("result directory already exists")
    root.mkdir(parents=True, exist_ok=args.reuse_existing_evidence)
    lock_value = json.loads(lock.read_text(encoding="utf-8"))
    if lock_value.get("contract", {}).get("workload") != args.workload:
        raise ValueError("deadline lock workload differs")
    order = orders[args.sequence_index]
    rows, inputs = [], []
    for position, system in enumerate(order, 1):
        output = root / f"{position:02d}-{system.lower().replace(' ', '-')}"
        if args.reuse_existing_evidence:
            row, evidence = existing_row(output, system)
        elif system in SCENARIOS:
            row, evidence = base_row(
                repo, output, system, lock, plan, args.requests,
                args.workload, args.background_period_ms, args.placement_variant,
                common_workload_contract,
                consumer_engine,
                producer_input_trace,
                producer_engine,
                args.warmup,
            )
        elif system == "Orion":
            row, evidence = orion_row(repo, output, lock, args.requests, args.workload)
        elif system == "XSched":
            row, evidence = xsched_row(
                repo, output, lock, args.requests, args.workload,
                common_workload_contract,
                consumer_engine,
                producer_input_trace,
                producer_engine,
                args.warmup,
            )
        else:
            if gpulet_source is None:
                raise ValueError("gpulet source is required for this order")
            row, evidence = gpulet_row(
                repo, gpulet_source, output, lock,
                args.gpulet_profile_requests, args.requests, args.workload,
                args.background_period_ms,
            )
        rows.append(row)
        inputs.append({"system": system, "path": str(evidence), "sha256": sha256(evidence)})
    summary = {
        "schema_version": 1,
        "kind": "p9-common-sota-williams-sequence",
        "scope": (
            "active-numeric-frontier-smoke-not-thermal-normalized-formal"
            if args.active_only
            else "balanced-sequence-smoke-not-thermal-normalized-formal"
        ),
        "proposed_system": "QUIET",
        "sequence_index": args.sequence_index,
        "execution_order": list(order),
        "deadline_lock": {"path": str(lock), "sha256": sha256(lock)},
        "quiet_plan": {"path": str(plan), "sha256": sha256(plan)},
        "requests_per_system": args.requests,
        "gpulet_profile_requests_per_partition": (
            args.gpulet_profile_requests if not args.active_only else None
        ),
        "active_exploratory_systems": list(ACTIVE_SYSTEMS),
        "numeric_frontier_systems": list(NUMERIC_FRONTIER_SYSTEMS),
        "active_only": args.active_only,
        "workload": args.workload,
        "common_workload": (
            json.loads(common_workload_contract.read_text(encoding="utf-8"))
            if common_workload_contract is not None else None
        ),
        "placement_variant": args.placement_variant,
        "deadline_mode": "wall",
        "comparison_contract_version": 1,
        "background_offered_rps": 1000.0 / args.background_period_ms,
        "reused_existing_evidence": args.reuse_existing_evidence,
        "results": rows,
        "inputs": inputs,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
