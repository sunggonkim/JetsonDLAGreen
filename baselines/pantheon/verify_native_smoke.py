#!/usr/bin/env python3
import argparse
import json
import pathlib
import re
from typing import Any


UPSTREAM_COMMIT = "1caa4321fe9f9902ffacb78978f11a32a7a62f64"
EXIT = re.compile(
    r"\[EXEC:EXIT\] HIGH_PRIORITY (?P<release>\d+) (?P<end>\d+) "
    r"(?P<latency>\d+) (?P<service>\d+) (?P<job>\d+) "
    r"(?P<last_block>\d+) (?P<accuracy>[0-9.]+)"
)
BLOCK = re.compile(
    r"\[EXEC:BLOCK\] HIGH_PRIORITY \d+ \d+ (?P<job>\d+) (?P<block>\d+)"
)


def verify(log_path: pathlib.Path, environment_path: pathlib.Path) -> dict[str, Any]:
    log = log_path.read_text(encoding="utf-8", errors="replace")
    if "[EXEC:START] HIGH_PRIORITY" not in log or "[EXEC:STOP] HIGH_PRIORITY" not in log:
        raise ValueError("Pantheon worker did not start and stop cleanly")
    if log.count("[SCHE]") != 2:
        raise ValueError("Pantheon did not schedule exactly two jobs")
    exits = [
        {
            "job": int(match.group("job")),
            "last_block": int(match.group("last_block")),
            "accuracy": float(match.group("accuracy")),
            "latency_us": int(match.group("latency")),
        }
        for match in EXIT.finditer(log)
    ]
    if len(exits) != 2 or {row["job"] for row in exits} != {0, 1}:
        raise ValueError("Pantheon did not complete both jobs exactly once")
    by_job = {row["job"]: row for row in exits}
    if by_job[0]["last_block"] != 1 or abs(by_job[0]["accuracy"] - 0.9) > 1e-6:
        raise ValueError("relaxed-deadline job did not use the full exit")
    if by_job[1]["last_block"] != 0 or abs(by_job[1]["accuracy"] - 0.7) > 1e-6:
        raise ValueError("tight-deadline job did not use Pantheon early exit")

    blocks = [(int(m.group("job")), int(m.group("block"))) for m in BLOCK.finditer(log)]
    if blocks != [(0, 0), (0, 1), (1, 0)]:
        raise ValueError("executed block sequence does not match selected exits")

    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    if environment.get("schema_version") != 1 or not environment.get("cuda_available"):
        raise ValueError("Pantheon CUDA environment was not verified")
    if environment.get("gpu") != {
        "name": "NVIDIA Thor MIG 2g.0gb",
        "multiprocessors": 12,
    }:
        raise ValueError("Pantheon used the wrong GPU instance")
    mig_uuid = environment.get("mig_uuid")
    if not isinstance(mig_uuid, str) or not mig_uuid.startswith("MIG-"):
        raise ValueError("Pantheon MIG UUID is missing")
    if not isinstance(environment.get("gemm_checksum"), (int, float)):
        raise ValueError("Pantheon CUDA GEMM gate is missing")

    return {
        "schema_version": 1,
        "kind": "pantheon-thor-native-positive-control",
        "upstream_commit": UPSTREAM_COMMIT,
        "numeric_comparison_allowed": False,
        "mig_uuid": mig_uuid,
        "gpu": environment["gpu"],
        "torch_version": environment["torch_version"],
        "full_exit_job": by_job[0],
        "early_exit_job": by_job[1],
        "executed_blocks": blocks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=pathlib.Path, required=True)
    parser.add_argument("--environment", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = verify(args.log, args.environment)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
