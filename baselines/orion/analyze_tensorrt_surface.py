#!/usr/bin/env python3
"""Classify Orion's compute interception coverage for an Nsight API trace."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import re
import subprocess
from typing import Iterable


UPSTREAM_COMMIT = "20f9469764fb96d94ce23a8e70615196e9ce4ba1"
ORION_COMPUTE_APIS = {
    "cudaLaunchKernel",
    "cublasSgemm",
    "cublasSgemm_v2",
    "cublasSgemmStridedBatched",
    "cublasLtMatmul",
}
DRIVER_LAUNCH_APIS = {"cuLaunchKernel", "cuLaunchKernelEx"}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_orion(source: pathlib.Path) -> pathlib.Path:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError("Orion source is not the pinned upstream commit")
    interceptor = source / "src" / "cuda_capture" / "intercept_temp.cpp"
    text = interceptor.read_text(encoding="utf-8")
    if not re.search(r"cudaError_t\s+cudaLaunchKernel\s*\(", text):
        raise ValueError("Orion runtime launch interceptor is missing")
    if re.search(r"CUresult\s+cuLaunchKernel(?:Ex)?\s*\(", text):
        raise ValueError("pinned Orion unexpectedly intercepts driver launches")
    return interceptor


def api_counts(stats_csv: pathlib.Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with stats_csv.open(newline="", encoding="utf-8") as source:
        for row in csv.reader(source):
            if len(row) != 9 or row[0] == "Time (%)":
                continue
            try:
                calls = int(row[2])
            except ValueError:
                continue
            name = row[8]
            counts[name] = counts.get(name, 0) + calls
    if not counts:
        raise ValueError("Nsight stats contain no CUDA API rows")
    return counts


def analyze(
    source: pathlib.Path, stats_csv: pathlib.Path, report: pathlib.Path
) -> dict[str, object]:
    interceptor = verify_orion(source)
    counts = api_counts(stats_csv)
    intercepted = {
        name: counts.get(name, 0) for name in sorted(ORION_COMPUTE_APIS)
    }
    driver = {name: counts.get(name, 0) for name in sorted(DRIVER_LAUNCH_APIS)}
    intercepted_total = sum(intercepted.values())
    driver_total = sum(driver.values())
    if driver_total == 0:
        status = "trace-does-not-prove-tensorrt-compute"
    elif intercepted_total == 0:
        status = "unsupported-tensorrt-driver-launch-surface"
    else:
        status = "interceptable-compute-observed"
    return {
        "schema_version": 1,
        "system": "Orion",
        "provenance": {
            "upstream_commit": UPSTREAM_COMMIT,
            "interceptor_path": str(interceptor.resolve()),
            "interceptor_sha256": sha256(interceptor),
            "nsys_stats_path": str(stats_csv.resolve()),
            "nsys_stats_sha256": sha256(stats_csv),
            "nsys_report_path": str(report.resolve()),
            "nsys_report_sha256": sha256(report),
        },
        "orion_compute_api_calls": intercepted,
        "unsupported_driver_launch_calls": driver,
        "orion_compute_api_call_total": intercepted_total,
        "unsupported_driver_launch_call_total": driver_total,
        "status": status,
        "numeric_comparison_allowed": status == "interceptable-compute-observed",
        "reason": (
            "TensorRT launches compute through CUDA driver APIs that the pinned "
            "Orion interceptor does not wrap"
            if status == "unsupported-tensorrt-driver-launch-surface"
            else "see status"
        ),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--stats", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(
        args.source.resolve(), args.stats.resolve(), args.report.resolve()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
