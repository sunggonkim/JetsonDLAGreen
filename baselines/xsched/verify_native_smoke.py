#!/usr/bin/env python3
import argparse
import json
import pathlib
import re
from typing import Any


UPSTREAM_COMMIT = "bd494cb7a72958cd11900243a0798df00d856c6e"
TRANSITION = re.compile(
    r"schedule transition pid (?P<pid>\d+) operation (?P<operation>\d+) "
    r"running (?P<running>\d+) suspended (?P<suspended>\d+)"
)


def load_result(path: pathlib.Path, role: str) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("schema_version") != 1 or result.get("role") != role:
        raise ValueError(f"{role} result schema/role mismatch")
    if result.get("completed_requests", 0) <= 0:
        raise ValueError(f"{role} completed no requests")
    start = result.get("measurement_start_monotonic_ns")
    end = result.get("measurement_end_monotonic_ns")
    if not isinstance(start, int) or not isinstance(end, int) or start >= end:
        raise ValueError(f"{role} measurement clocks are invalid")
    return result


def verify(
    be_path: pathlib.Path,
    hp_path: pathlib.Path,
    server_log_path: pathlib.Path,
    be_log_path: pathlib.Path,
    hp_log_path: pathlib.Path,
) -> dict[str, Any]:
    be = load_result(be_path, "pressure")
    hp = load_result(hp_path, "benchmark")
    if be.get("gpu") != hp.get("gpu"):
        raise ValueError("clients ran on different GPU instances")
    be_env = be.get("execution_environment", {})
    hp_env = hp.get("execution_environment", {})
    if be_env.get("cuda_visible_devices") != hp_env.get("cuda_visible_devices"):
        raise ValueError("clients used different CUDA_VISIBLE_DEVICES")
    if be["measurement_start_monotonic_ns"] >= hp["measurement_end_monotonic_ns"] or \
            hp["measurement_start_monotonic_ns"] >= be["measurement_end_monotonic_ns"]:
        raise ValueError("client measurement intervals did not overlap")
    if be["measurement_end_monotonic_ns"] <= hp["measurement_end_monotonic_ns"]:
        raise ValueError("BE measurement did not continue after HP completion")

    be_pid = be_env.get("pid")
    hp_pid = hp_env.get("pid")
    if not isinstance(be_pid, int) or not isinstance(hp_pid, int) or be_pid == hp_pid:
        raise ValueError("client PIDs are invalid")

    server_log = server_log_path.read_text(encoding="utf-8", errors="replace")
    transitions = [
        {key: int(value) for key, value in match.groupdict().items()}
        for match in TRANSITION.finditer(server_log)
    ]
    be_suspended = [
        index for index, row in enumerate(transitions)
        if row["pid"] == be_pid and row["running"] == 0 and row["suspended"] == 1
    ]
    hp_running = [
        index for index, row in enumerate(transitions)
        if row["pid"] == hp_pid and row["running"] == 1 and row["suspended"] == 0
    ]
    be_resumed = [
        index for index, row in enumerate(transitions)
        if row["pid"] == be_pid and row["running"] == 1 and row["suspended"] == 0
    ]
    if not be_suspended or not hp_running:
        raise ValueError("HPF did not run HP while suspending BE")
    first_suspend = be_suspended[0]
    if not any(index <= first_suspend for index in hp_running):
        raise ValueError("HP queue was not running before BE suspension")
    if not any(index > first_suspend for index in be_resumed):
        raise ValueError("BE queue was not resumed after suspension")

    for path in (be_log_path, hp_log_path):
        log = path.read_text(encoding="utf-8", errors="replace")
        if "cuda error" in log.lower():
            raise ValueError(f"CUDA error in {path}")
        if "using global scheduler" not in log:
            raise ValueError(f"global scheduler missing from {path}")

    return {
        "schema_version": 1,
        "kind": "xsched-thor-native-positive-control",
        "upstream_commit": UPSTREAM_COMMIT,
        "numeric_comparison_allowed": False,
        "gpu": hp["gpu"],
        "cuda_visible_devices": hp_env["cuda_visible_devices"],
        "be_pid": be_pid,
        "hp_pid": hp_pid,
        "be_completed_requests": be["completed_requests"],
        "hp_completed_requests": hp["completed_requests"],
        "be_suspend_transitions": len(be_suspended),
        "hp_running_transitions": len(hp_running),
        "be_resume_transitions": len(be_resumed),
        "measurement_overlap_ns": min(
            be["measurement_end_monotonic_ns"], hp["measurement_end_monotonic_ns"]
        ) - max(
            be["measurement_start_monotonic_ns"], hp["measurement_start_monotonic_ns"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--be", type=pathlib.Path, required=True)
    parser.add_argument("--hp", type=pathlib.Path, required=True)
    parser.add_argument("--server-log", type=pathlib.Path, required=True)
    parser.add_argument("--be-log", type=pathlib.Path, required=True)
    parser.add_argument("--hp-log", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = verify(args.be, args.hp, args.server_log, args.be_log, args.hp_log)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
