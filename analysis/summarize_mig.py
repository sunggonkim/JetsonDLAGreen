#!/usr/bin/env python3
import json
import pathlib
import sys


def load(path: pathlib.Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_mig.py RESULT_DIRECTORY")
    result_dir = pathlib.Path(sys.argv[1])
    isolated = load(result_dir / "isolated.json")
    isolated_p99 = isolated["release_to_completion"]["p99_ms"]
    cases = {}
    for name in ("isolated", "compute", "memory"):
        result = isolated if name == "isolated" else load(result_dir / f"{name}.json")
        latency = result["release_to_completion"]
        cases[name] = {
            "p50_ms": latency["p50_ms"],
            "p99_ms": latency["p99_ms"],
            "p999_ms": latency["p999_ms"],
            "max_ms": latency["max_ms"],
            "p99_slowdown": latency["p99_ms"] / isolated_p99,
        }
        if name != "isolated":
            pressure = load(result_dir / f"{name}-pressure.json")
            cases[name]["pressure_launches"] = pressure["completed_launches"]
    print(json.dumps({"schema_version": 1, "cases": cases}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
