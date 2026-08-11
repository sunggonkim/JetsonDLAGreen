#!/usr/bin/env python3
"""Source-faithful DeepPlan plan selection over Thor profile records."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Iterable


UPSTREAM_COMMIT = "ceb324428184bb46987fba235c7c893a0e6a48f1"
LOAD_THEN_EXECUTE = 0
DIRECT_HOST_ACCESS = 1
REQUIRED_KEYS = (
    "index", "layer_type", "size_bytes", "load_time_us",
    "cuda_exec_time_us", "cuda_host_exec_time_us",
)


def validate_layers(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("layers must be a nonempty list")
    layers: list[dict[str, Any]] = []
    for expected_index, source in enumerate(value):
        if not isinstance(source, dict) or set(source) != set(REQUIRED_KEYS):
            raise ValueError("each layer must have the exact DeepPlan profile schema")
        layer = dict(source)
        if type(layer["index"]) is not int or layer["index"] != expected_index:
            raise ValueError("layer indices must be contiguous and ordered")
        if not isinstance(layer["layer_type"], str) or not layer["layer_type"]:
            raise ValueError("layer_type must be nonempty")
        if type(layer["size_bytes"]) is not int or layer["size_bytes"] < 0:
            raise ValueError("size_bytes must be a nonnegative integer")
        for key in ("load_time_us", "cuda_exec_time_us", "cuda_host_exec_time_us"):
            number = layer[key]
            if type(number) not in (int, float) or not math.isfinite(float(number)) or number < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
            layer[key] = float(number)
        layers.append(layer)
    return layers


def naive_plan(layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = copy.deepcopy(layers)
    for layer in result:
        layer["exec_type"] = LOAD_THEN_EXECUTE if layer["size_bytes"] > 0 else DIRECT_HOST_ACCESS
    return result


def execution_trace(layers: list[dict[str, Any]]) -> list[tuple[float, float, float]]:
    trace = [(0.0, 0.0, 0.0)]
    for layer in layers:
        ready, runtime, _ = trace[-1]
        stall = 0.0
        if layer["exec_type"] == LOAD_THEN_EXECUTE:
            ready -= layer["load_time_us"]
            if ready < 0:
                stall = -ready
                runtime += stall
                ready = 0.0
            ready += layer["cuda_exec_time_us"]
            runtime += layer["cuda_exec_time_us"]
        else:
            ready += layer["cuda_host_exec_time_us"]
            runtime += layer["cuda_host_exec_time_us"]
        trace.append((ready, runtime, stall))
    return trace


def static_plan(layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = naive_plan(layers)
    for layer in result:
        if layer["exec_type"] == DIRECT_HOST_ACCESS:
            continue
        if "BatchNorm" in layer["layer_type"] or "Embedding" in layer["layer_type"]:
            layer["exec_type"] = DIRECT_HOST_ACCESS
        elif layer["cuda_host_exec_time_us"] < (
            layer["cuda_exec_time_us"] + layer["load_time_us"]
        ):
            layer["exec_type"] = DIRECT_HOST_ACCESS
    return result


def dynamic_plan(layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = naive_plan(layers)
    trace = execution_trace(result)
    for boundary in range(len(trace)):
        stall = trace[boundary][2]
        if stall <= 0:
            continue
        candidates = sorted(
            result[:boundary],
            key=lambda layer: (
                layer["cuda_host_exec_time_us"] - layer["cuda_exec_time_us"],
                -layer["load_time_us"],
            ),
        )
        for layer in candidates[:boundary]:
            if layer["exec_type"] == DIRECT_HOST_ACCESS:
                continue
            performance_gap = layer["cuda_host_exec_time_us"] - layer["cuda_exec_time_us"]
            overloaded = layer["cuda_host_exec_time_us"] > (
                layer["cuda_exec_time_us"] + 1.5 * layer["load_time_us"]
            )
            if overloaded or stall < performance_gap:
                break
            result[layer["index"]]["exec_type"] = DIRECT_HOST_ACCESS
            stall -= layer["load_time_us"] + performance_gap
            if stall <= 0:
                trace = execution_trace(result)
                break
    return result


def summarize(plan: list[dict[str, Any]]) -> dict[str, Any]:
    trace = execution_trace(plan)
    return {
        "direct_host_layers": [
            layer["index"] for layer in plan if layer["exec_type"] == DIRECT_HOST_ACCESS
        ],
        "load_then_execute_layers": [
            layer["index"] for layer in plan if layer["exec_type"] == LOAD_THEN_EXECUTE
        ],
        "predicted_runtime_us": trace[-1][1],
        "total_stall_us": sum(row[2] for row in trace),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    if profile.get("kind") != "deepplan-thor-layer-profile":
        raise ValueError("unexpected profile kind")
    layers = validate_layers(profile.get("layers"))
    plans = {
        "naive": summarize(naive_plan(layers)),
        "static": summarize(static_plan(layers)),
        "dynamic": summarize(dynamic_plan(layers)),
    }
    output = {
        "schema_version": 1,
        "kind": "deepplan-thor-plan",
        "system": "DeepPlan",
        "upstream_commit": UPSTREAM_COMMIT,
        "algorithm": "upstream-plan.py-naive-static-dynamic",
        "scope": "source-faithful-planner-not-yet-common-runtime",
        "plans": plans,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
