#!/usr/bin/env python3
import json
import pathlib
import sys


def load(path: pathlib.Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_mig_matrix.py RESULT_DIRECTORY")
    result_dir = pathlib.Path(sys.argv[1])
    reference = load(result_dir / "mig2g-isolated.json")
    reference_p99 = reference["release_to_completion"]["p99_ms"]
    cases = {}
    for path in sorted(result_dir.glob("*.json")):
        if path.name.endswith("-pressure.json") or path.name == "summary.json":
            continue
        result = load(path)
        latency = result["release_to_completion"]
        name = path.stem
        case = {
            "p50_ms": latency["p50_ms"],
            "p99_ms": latency["p99_ms"],
            "p999_ms": latency["p999_ms"],
            "max_ms": latency["max_ms"],
            "p99_slowdown": latency["p99_ms"] / reference_p99,
        }
        pressure_path = result_dir / f"{name}-pressure.json"
        if pressure_path.exists():
            pressure = load(pressure_path)
            duration = pressure["actual_duration_seconds"]
            case["pressure_launches"] = pressure["completed_launches"]
            case["pressure_goodput_per_second"] = (
                pressure["completed_launches"] / duration
            )
        cases[name] = case
    print(
        json.dumps(
            {
                "schema_version": 1,
                "reference": "mig2g-isolated",
                "samples_per_case": reference["release_to_completion"]["count"],
                "cases": cases,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
