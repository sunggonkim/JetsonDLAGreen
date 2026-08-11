#!/usr/bin/env python3
import json
import pathlib
import sys


def load(path: pathlib.Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_full_gpu.py RESULT_DIRECTORY")
    result_dir = pathlib.Path(sys.argv[1])
    isolated = load(result_dir / "isolated.json")
    isolated_p99 = isolated["release_to_completion"]["p99_ms"]
    green_isolated_path = result_dir / "green-isolated.json"
    green_isolated_p99 = (
        load(green_isolated_path)["release_to_completion"]["p99_ms"]
        if green_isolated_path.exists()
        else isolated_p99
    )
    cases = {}
    for path in sorted(result_dir.glob("*.json")):
        if path.name.endswith("-pressure.json") or path.name == "summary.json":
            continue
        result = load(path)
        latency = result["release_to_completion"]
        name = path.stem
        reference_name = "green-isolated" if name.startswith("green-") else "isolated"
        reference_p99 = green_isolated_p99 if name.startswith("green-") else isolated_p99
        case = {
            "p50_ms": latency["p50_ms"],
            "p99_ms": latency["p99_ms"],
            "p999_ms": latency["p999_ms"],
            "max_ms": latency["max_ms"],
            "p99_slowdown": latency["p99_ms"] / isolated_p99,
            "interference_reference": reference_name,
            "interference_p99_slowdown": latency["p99_ms"] / reference_p99,
        }
        pressure_path = result_dir / f"{name}-pressure.json"
        if pressure_path.exists():
            pressure = load(pressure_path)
            case["pressure_launches"] = pressure["completed_launches"]
        cases[name] = case
    output = {
        "schema_version": 1,
        "reference": "isolated",
        "samples_per_case": isolated["release_to_completion"]["count"],
        "cases": cases,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
