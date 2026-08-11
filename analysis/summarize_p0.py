#!/usr/bin/env python3
"""Validate a P0 result directory and summarize p99 interference."""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any


class ResultError(RuntimeError):
    """Raised when a result file violates the P0 schema."""


def load_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as source:
            value = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise ResultError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise ResultError(f"{path} must contain a JSON object")
    if value.get("schema_version") != 1:
        raise ResultError(f"{path} has an unsupported schema version")
    return value


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ResultError(f"{label} must be an object")
    return value


def require_positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ResultError(f"{label} must be a positive number")
    return float(value)


def benchmark_p99(path: pathlib.Path, expected_mode: str) -> tuple[dict[str, Any], float]:
    result = load_json(path)
    config = require_mapping(result.get("config"), f"{path}: config")
    if config.get("background") != expected_mode:
        raise ResultError(f"{path}: expected background mode {expected_mode!r}")
    latency = require_mapping(
        result.get("release_to_completion"), f"{path}: release_to_completion"
    )
    return result, require_positive_number(latency.get("p99_ms"), f"{path}: p99_ms")


def summarize(directory: pathlib.Path) -> dict[str, Any]:
    capabilities = load_json(directory / "capabilities.json")
    cuda = require_mapping(capabilities.get("cuda"), "capabilities: cuda")
    devices = cuda.get("devices")
    if not isinstance(devices, list) or not devices:
        raise ResultError("capabilities: at least one CUDA device is required")
    first_device = require_mapping(devices[0], "capabilities: cuda.devices[0]")
    tensorrt = require_mapping(
        capabilities.get("tensorrt"), "capabilities: tensorrt"
    )

    _, isolated_p99 = benchmark_p99(directory / "none.json", "none")
    modes: dict[str, Any] = {}
    for mode in ("none", "compute", "memory"):
        result, p99 = benchmark_p99(directory / f"{mode}.json", mode)
        modes[mode] = {
            "p99_ms": p99,
            "p99_inflation": p99 / isolated_p99,
            "p50_ms": result["release_to_completion"]["p50_ms"],
            "background_completed_launches": result[
                "background_completed_launches"
            ],
        }

    green_context = require_mapping(
        first_device.get("green_context"), "capabilities: green_context"
    )
    return {
        "schema_version": 1,
        "platform": capabilities.get("board", {}).get("model", ""),
        "gpu": first_device.get("name", ""),
        "compute_capability": first_device.get("compute_capability", ""),
        "tensorrt_dla_cores": tensorrt.get("dla_cores"),
        "green_context_resource_query_supported": green_context.get(
            "resource_query_supported"
        ),
        "green_context_creation_supported": green_context.get(
            "context_creation_supported"
        ),
        "green_context_minimum_partition_size": green_context.get(
            "minimum_partition_size"
        ),
        "isolated_p99_ms": isolated_p99,
        "modes": modes,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize_p0.py RESULT_DIRECTORY", file=sys.stderr)
        return 2
    try:
        summary = summarize(pathlib.Path(sys.argv[1]))
    except ResultError as error:
        print(f"summarize_p0.py: {error}", file=sys.stderr)
        return 1
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
