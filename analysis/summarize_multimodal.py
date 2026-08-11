#!/usr/bin/env python3
"""Summarize repeated critical-plus-multimodal TensorRT experiments."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import statistics


T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
CRITICAL_PATTERN = re.compile(r"^(?P<case>.+)-r\d+-critical\.json$")


def confidence(values: list[float]) -> dict[str, float | int]:
    count = len(values)
    mean = statistics.fmean(values)
    if count == 1:
        return {"n": 1, "mean": mean, "stdev": 0.0, "ci95": 0.0}
    stdev = statistics.stdev(values)
    return {
        "n": count,
        "mean": mean,
        "stdev": stdev,
        "ci95": T95.get(count - 1, 1.96) * stdev / math.sqrt(count),
    }


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=pathlib.Path)
    args = parser.parse_args()

    grouped: dict[str, list[pathlib.Path]] = {}
    for path in args.result_dir.glob("*-r*-critical.json"):
        match = CRITICAL_PATTERN.match(path.name)
        if match:
            grouped.setdefault(match.group("case"), []).append(path)
    if "isolated" not in grouped:
        raise SystemExit("isolated repetitions are missing")

    isolated_p99 = statistics.fmean(
        load(path)["release_to_completion"]["p99_ms"]
        for path in grouped["isolated"]
    )
    cases = {}
    for name, paths in sorted(grouped.items()):
        critical = [load(path) for path in sorted(paths)]
        p99 = [item["release_to_completion"]["p99_ms"] for item in critical]
        case = {
            "critical_p50_ms": confidence(
                [item["release_to_completion"]["p50_ms"] for item in critical]
            ),
            "critical_p99_ms": confidence(p99),
            "critical_p999_ms": confidence(
                [item["release_to_completion"]["p999_ms"] for item in critical]
            ),
            "critical_deadline_miss_rate": confidence(
                [
                    float(item["deadline_miss_rate"] or 0.0)
                    for item in critical
                ]
            ),
            "critical_p99_slowdown": statistics.fmean(p99) / isolated_p99,
        }
        for modality in ("language", "audio"):
            pressure_paths = [
                path.with_name(path.name.replace("-critical.json", f"-{modality}.json"))
                for path in paths
            ]
            if all(path.exists() for path in pressure_paths):
                pressure = [load(path) for path in pressure_paths]
                case[f"{modality}_goodput_per_second"] = confidence(
                    [item["throughput_per_second"] for item in pressure]
                )
        cases[name] = case

    output = {
        "schema_version": 1,
        "result_dir": str(args.result_dir),
        "deadline_ms": float(
            (args.result_dir / "deadline-ms.txt").read_text(encoding="utf-8")
        ),
        "isolated_p99_ms": isolated_p99,
        "cases": cases,
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
