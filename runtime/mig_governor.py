#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import subprocess
import time


QUOTAS = (25, 50, 75, 100)
TRACE = (
    ("compute", 1),
    ("memory", 2),
    ("memory", 4),
    ("compute", 4),
    ("memory", 6),
    ("compute", 2),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a deadline-aware MPS quota governor across Thor MIG instances"
    )
    parser.add_argument("--bench", type=pathlib.Path, required=True)
    parser.add_argument("--mig-env", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--pressure-seconds", type=float, default=2.0)
    parser.add_argument("--slo-factor", type=float, default=1.05)
    parser.add_argument("--critical-cpus", default="12")
    parser.add_argument("--pressure-cpus", default="0-11")
    parser.add_argument(
        "--policy-order",
        default="static-q25,static-q100,jdg-governor",
        help="Comma-separated policy order for counterbalanced repetitions",
    )
    return parser.parse_args()


def load_env(path: pathlib.Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    return values


def run_json(command: list[str], env: dict[str, str], timeout: float) -> dict:
    completed = subprocess.run(
        command,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return json.loads(completed.stdout)


def expand_cpu_list(specification: str) -> list[int]:
    cpus = []
    for item in specification.split(","):
        bounds = item.split("-", 1)
        if len(bounds) == 1:
            cpus.append(int(bounds[0]))
        else:
            first, last = (int(value) for value in bounds)
            if first > last:
                raise ValueError(f"invalid CPU range: {item}")
            cpus.extend(range(first, last + 1))
    if not cpus or len(cpus) != len(set(cpus)):
        raise ValueError("CPU list must be non-empty and contain no duplicates")
    return cpus


def pinned(command: list[str], cpu_specification: str) -> list[str]:
    return ["taskset", "--cpu-list", cpu_specification, *command]


def critical_command(args: argparse.Namespace, deadline_ms: float | None) -> list[str]:
    command = [
        str(args.bench),
        "--background",
        "none",
        "--samples",
        str(args.samples),
        "--warmup",
        str(args.warmup),
    ]
    if deadline_ms is not None:
        command.extend(("--deadline-ms", str(deadline_ms)))
    return command


def pressure_command(args: argparse.Namespace, mode: str) -> list[str]:
    return [
        str(args.bench),
        "--role",
        "pressure",
        "--background",
        mode,
        "--duration-seconds",
        str(args.pressure_seconds),
    ]


def profiled_action(
    mode: str, offered_tenants: int, memory_admission_limit: int
) -> tuple[int, int]:
    if mode == "memory":
        return QUOTAS[0], min(offered_tenants, memory_admission_limit)
    fair_share = max(QUOTAS[0], 100 // offered_tenants)
    quota = max(candidate for candidate in QUOTAS if candidate <= fair_share)
    return quota, offered_tenants


def run_policy(
    name: str,
    initial_quota: int,
    adaptive: bool,
    args: argparse.Namespace,
    base_env: dict[str, str],
    mig: dict[str, str],
    deadline_ms: float,
) -> dict:
    quota = initial_quota
    memory_admission_limit = 1
    memory_safe_epochs = 0
    epochs = []
    start = time.monotonic()
    for epoch_index in range(args.epochs):
        mode, offered_tenants = TRACE[epoch_index % len(TRACE)]
        if adaptive:
            quota, admitted_tenants = profiled_action(
                mode, offered_tenants, memory_admission_limit
            )
        else:
            admitted_tenants = offered_tenants
        epoch_start = time.monotonic()
        pressures = []
        pressure_processes = []
        if admitted_tenants > 0:
            pressure_env = base_env | {
                "CUDA_VISIBLE_DEVICES": mig["JDG_MIG_SMALL_UUID"],
                "CUDA_MPS_PIPE_DIRECTORY": mig["JDG_MPS_PIPE_DIRECTORY"],
                "CUDA_MPS_LOG_DIRECTORY": mig["JDG_MPS_LOG_DIRECTORY"],
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(quota),
            }
            pressure_cpus = expand_cpu_list(args.pressure_cpus)
            for tenant in range(admitted_tenants):
                pressure_processes.append(
                    subprocess.Popen(
                        pinned(
                            pressure_command(args, mode),
                            str(pressure_cpus[tenant % len(pressure_cpus)]),
                        ),
                        env=pressure_env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )
            time.sleep(0.25)

        critical_env = base_env | {
            "CUDA_VISIBLE_DEVICES": mig["JDG_MIG_BIG_UUID"]
        }
        try:
            critical = run_json(
                pinned(
                    critical_command(args, deadline_ms), args.critical_cpus
                ),
                critical_env,
                timeout=60.0,
            )

            if pressure_processes:
                for pressure_process in pressure_processes:
                    stdout, stderr = pressure_process.communicate(
                        timeout=args.pressure_seconds + 30.0
                    )
                    if pressure_process.returncode != 0:
                        raise RuntimeError(
                            "pressure process failed "
                            f"({pressure_process.returncode}): {stderr}"
                        )
                    pressures.append(json.loads(stdout))
            else:
                remaining = args.pressure_seconds - (time.monotonic() - epoch_start)
                if remaining > 0:
                    time.sleep(remaining)
        except Exception:
            for pressure_process in pressure_processes:
                if pressure_process.poll() is not None:
                    continue
                pressure_process.terminate()
                try:
                    pressure_process.communicate(timeout=5.0)
                except subprocess.TimeoutExpired:
                    pressure_process.kill()
                    pressure_process.communicate()
            raise

        latency = critical["release_to_completion"]
        miss_rate = critical["deadline_miss_rate"]
        violated = latency["p99_ms"] > deadline_ms or miss_rate > 0.01
        launches = sum(pressure["completed_launches"] for pressure in pressures)
        epoch = {
            "epoch": epoch_index,
            "mode": mode,
            "offered_tenants": offered_tenants,
            "admitted_tenants": admitted_tenants,
            "quota_percent": quota,
            "p50_ms": latency["p50_ms"],
            "p99_ms": latency["p99_ms"],
            "p999_ms": latency["p999_ms"],
            "max_ms": latency["max_ms"],
            "deadline_misses": critical["deadline_misses"],
            "deadline_miss_rate": miss_rate,
            "violated": violated,
            "pressure_launches": launches,
            "elapsed_seconds": time.monotonic() - epoch_start,
        }
        epochs.append(epoch)
        if adaptive and mode == "memory":
            if violated:
                memory_admission_limit = max(0, memory_admission_limit - 1)
                memory_safe_epochs = 0
            else:
                memory_safe_epochs += 1
                if memory_safe_epochs >= 2 and memory_admission_limit < 1:
                    memory_admission_limit += 1
                    memory_safe_epochs = 0

    elapsed = time.monotonic() - start
    total_requests = args.epochs * args.samples
    total_misses = sum(epoch["deadline_misses"] for epoch in epochs)
    violation_epochs = sum(epoch["violated"] for epoch in epochs)
    by_mode = {}
    for mode in sorted({mode for mode, _ in TRACE}):
        selected = [epoch for epoch in epochs if epoch["mode"] == mode]
        if selected:
            by_mode[mode] = {
                "pressure_launches": sum(e["pressure_launches"] for e in selected),
                "p99_ms_max": max(e["p99_ms"] for e in selected),
            }
    return {
        "name": name,
        "adaptive": adaptive,
        "initial_quota_percent": initial_quota,
        "deadline_ms": deadline_ms,
        "deadline_misses": total_misses,
        "deadline_miss_rate": total_misses / total_requests,
        "violation_epochs": violation_epochs,
        "violation_epoch_rate": violation_epochs / args.epochs,
        "pressure_launches": sum(epoch["pressure_launches"] for epoch in epochs),
        "pressure_goodput_per_second": sum(
            epoch["pressure_launches"] for epoch in epochs
        )
        / elapsed,
        "elapsed_seconds": elapsed,
        "by_mode": by_mode,
        "epochs": epochs,
    }


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.samples <= 0 or args.warmup < 0:
        raise SystemExit("epochs and samples must be positive")
    mig = load_env(args.mig_env)
    base_env = os.environ.copy()
    base_env["LD_LIBRARY_PATH"] = "/usr/local/cuda-13.2/lib64:" + base_env.get(
        "LD_LIBRARY_PATH", ""
    )
    critical_env = base_env | {"CUDA_VISIBLE_DEVICES": mig["JDG_MIG_BIG_UUID"]}
    calibration = run_json(
        pinned(critical_command(args, None), args.critical_cpus),
        critical_env,
        timeout=60.0,
    )
    isolated_p99 = calibration["release_to_completion"]["p99_ms"]
    deadline_ms = isolated_p99 * args.slo_factor
    policy_definitions = {
        "static-q25": (25, False),
        "static-q100": (100, False),
        "jdg-governor": (100, True),
    }
    policy_order = args.policy_order.split(",")
    if len(policy_order) != len(policy_definitions) or set(policy_order) != set(
        policy_definitions
    ):
        raise SystemExit("policy-order must contain each policy exactly once")
    policies = []
    for name in policy_order:
        initial_quota, adaptive = policy_definitions[name]
        policies.append(
            run_policy(
                name,
                initial_quota,
                adaptive,
                args,
                base_env,
                mig,
                deadline_ms,
            )
        )
    output = {
        "schema_version": 1,
        "config": {
            "epochs": args.epochs,
            "samples_per_epoch": args.samples,
            "warmup": args.warmup,
            "pressure_seconds": args.pressure_seconds,
            "slo_factor": args.slo_factor,
            "critical_cpus": args.critical_cpus,
            "pressure_cpus": args.pressure_cpus,
            "policy_order": policy_order,
            "trace": [
                {"mode": mode, "offered_tenants": tenants}
                for mode, tenants in TRACE
            ],
        },
        "calibration": calibration,
        "deadline_ms": deadline_ms,
        "policies": policies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
