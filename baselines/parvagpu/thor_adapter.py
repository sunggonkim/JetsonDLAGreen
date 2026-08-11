#!/usr/bin/env python3
"""ParvaGPU segment configurator and allocator for Thor's fixed 2g+1g layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any


UPSTREAM_COMMIT = "5f3de1e18582b4c81896a1c3eb0e2915238dfee6"
FIDELITY = "algorithm-preserving-thor-port"
SUPPORTED_SEGMENTS = (1, 2)
SLO_HEADROOM = 0.9
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ProfilePoint:
    model: str
    segment_gpc: int
    batch_size: int
    processes: int
    throughput: float
    latency_ms: float


@dataclass(frozen=True)
class Service:
    model: str
    request_rate: float
    slo_ms: float


@dataclass(frozen=True)
class SegmentRequest:
    model: str
    point: ProfilePoint


def finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def load_profiles(path: pathlib.Path) -> list[ProfilePoint]:
    points: list[ProfilePoint] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {"model", "segment_gpc", "batch_size", "processes", "throughput", "latency_ms"}
        if set(reader.fieldnames or ()) != expected:
            raise ValueError("ParvaGPU profile CSV schema differs from the Thor adapter contract")
        for row in reader:
            segment = int(row["segment_gpc"])
            batch = int(row["batch_size"])
            processes = int(row["processes"])
            if segment not in SUPPORTED_SEGMENTS:
                raise ValueError("Thor ParvaGPU profiles may contain only 1g or 2g segments")
            if batch <= 0 or processes < 1 or processes > 3:
                raise ValueError("ParvaGPU batch/process profile is invalid")
            points.append(
                ProfilePoint(
                    row["model"], segment, batch, processes,
                    finite_positive(float(row["throughput"]), "throughput"),
                    finite_positive(float(row["latency_ms"]), "latency_ms"),
                )
            )
    if not points:
        raise ValueError("ParvaGPU profile is empty")
    return points


def load_spec(path: pathlib.Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or spec.get("schema_version") != 1:
        raise ValueError("ParvaGPU spec must use schema_version 1")
    if spec.get("system") != "ParvaGPU" or spec.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("ParvaGPU spec does not bind the pinned artifact")
    contract = spec.get("contract")
    if not isinstance(contract, dict) or contract.get("pressure_layout") != "1g+2g":
        raise ValueError("ParvaGPU Thor port requires the fixed 1g+2g layout")
    available = spec.get("available_segments_gpc")
    scenario = contract.get("scenario")
    expected = [1, 2] if scenario == "independent-payload-services" else [1]
    if available != expected:
        raise ValueError("available segments differ from the fixed-layout contract")
    lock_value = contract.get("deadline_lock_path")
    if lock_value is not None:
        repo = REPO_ROOT
        lock_path = (repo / lock_value).resolve()
        if sha256(lock_path) != contract.get("deadline_lock_sha256"):
            raise ValueError("ParvaGPU deadline lock hash differs")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if (
            lock.get("kind") != "p9-dependent-pipeline-deadline-lock"
            or finite_positive(lock.get("deadline_us"), "locked deadline")
            != finite_positive(contract.get("pipeline_deadline_us"), "pipeline deadline")
        ):
            raise ValueError("ParvaGPU contract differs from deadline lock")
    return spec


def load_profile_manifest(
    path: pathlib.Path, profile_path: pathlib.Path
) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("platform") != "NVIDIA Thor"
        or manifest.get("mig_profile") not in {"1g.0gb", "fixed-2g+1g"}
        or manifest.get("mps_quota_percent") != 100
    ):
        raise ValueError("invalid ParvaGPU Thor profile manifest")
    expected = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    if manifest.get("profile_sha256") != expected:
        raise ValueError("ParvaGPU profile hash differs from its manifest")
    inputs = manifest.get("input_sha256")
    if (
        not isinstance(inputs, dict)
        or len(inputs) < 2
        or not all(isinstance(value, str) and len(value) == 64 for value in inputs.values())
    ):
        raise ValueError("ParvaGPU input profile provenance is incomplete")
    return manifest


def services_from_spec(spec: dict[str, Any]) -> list[Service]:
    services: list[Service] = []
    seen: set[str] = set()
    for raw in spec.get("services", []):
        if not isinstance(raw, dict) or not isinstance(raw.get("model"), str):
            raise ValueError("ParvaGPU service must name a model")
        model = raw["model"]
        if not model or model in seen:
            raise ValueError("ParvaGPU service model is empty or duplicated")
        seen.add(model)
        services.append(
            Service(
                model,
                finite_positive(raw.get("request_rate"), "request_rate"),
                finite_positive(raw.get("slo_ms"), "slo_ms"),
            )
        )
    if not services:
        raise ValueError("ParvaGPU needs at least one service")
    return services


def configure_service(service: Service, profiles: list[ProfilePoint]) -> list[SegmentRequest]:
    eligible = [
        point
        for point in profiles
        if point.model == service.model
        and point.latency_ms < service.slo_ms / 2.0 * SLO_HEADROOM
    ]
    if not eligible:
        raise RuntimeError(f"no SLO-feasible ParvaGPU profile for {service.model}")

    # Upstream optimal_triplet_decision maximizes throughput per GPC.
    optimal = max(eligible, key=lambda point: point.throughput / point.segment_gpc)
    complete = int(math.floor(service.request_rate / optimal.throughput))
    if math.fmod(service.request_rate, optimal.throughput) == 0.0:
        complete -= 1
    complete = max(0, complete)
    segments = [SegmentRequest(service.model, optimal) for _ in range(complete)]
    covered = complete * optimal.throughput
    remainder = service.request_rate - covered

    # Upstream demand_matching chooses the smallest profiled segment whose
    # selected point can cover the last-instance demand.
    by_size: dict[int, ProfilePoint] = {}
    for size in SUPPORTED_SEGMENTS:
        candidates = [point for point in eligible if point.segment_gpc == size]
        if candidates:
            by_size[size] = max(candidates, key=lambda point: point.throughput / size)
    for size in SUPPORTED_SEGMENTS:
        point = by_size.get(size)
        if point is not None and remainder <= point.throughput:
            segments.append(SegmentRequest(service.model, point))
            return segments
    raise RuntimeError(f"ParvaGPU cannot cover the remaining demand for {service.model}")


def allocate_fixed(
    requests: list[SegmentRequest], available_segments: list[int]
) -> tuple[bool, list[dict[str, Any]], str | None]:
    bins = sorted(available_segments, reverse=True)
    allocation: list[dict[str, Any]] = []
    # Different models cannot share one ParvaGPU segment; same-model MPS
    # concurrency is already represented by ProfilePoint.processes.
    for request in sorted(requests, key=lambda item: item.point.segment_gpc, reverse=True):
        try:
            index = next(i for i, size in enumerate(bins) if size >= request.point.segment_gpc)
        except StopIteration:
            return False, allocation, "insufficient fixed MIG segments"
        physical = bins.pop(index)
        allocation.append(
            {
                "model": request.model,
                "physical_segment_gpc": physical,
                "profile_segment_gpc": request.point.segment_gpc,
                "batch_size": request.point.batch_size,
                "processes": request.point.processes,
                "profile_throughput": request.point.throughput,
                "profile_latency_ms": request.point.latency_ms,
            }
        )
    return True, allocation, None


def run(
    spec: dict[str, Any],
    profiles: list[ProfilePoint],
    profile_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    requests: list[SegmentRequest] = []
    configuration_error: str | None = None
    try:
        for service in services_from_spec(spec):
            requests.extend(configure_service(service, profiles))
    except RuntimeError as error:
        configuration_error = str(error)
    if configuration_error is None:
        feasible, allocation, reason = allocate_fixed(requests, spec["available_segments_gpc"])
    else:
        feasible, allocation, reason = False, [], configuration_error
    result: dict[str, Any] = {
        "schema_version": 1,
        "system": "ParvaGPU",
        "provenance": {"upstream_commit": UPSTREAM_COMMIT, "fidelity": FIDELITY},
        "contract": spec["contract"],
        "feasible": feasible,
        "reason": reason,
        "segment_requests": [
            {
                "model": request.model,
                "segment_gpc": request.point.segment_gpc,
                "batch_size": request.point.batch_size,
                "processes": request.point.processes,
            }
            for request in requests
        ],
        "allocation": allocation,
    }
    if profile_provenance is not None:
        result["provenance"]["thor_profile"] = profile_provenance
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=pathlib.Path, required=True)
    parser.add_argument("--profile", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    profile_path = args.profile.resolve()
    manifest_path = args.manifest.resolve()
    manifest = load_profile_manifest(manifest_path, profile_path)
    result = run(
        load_spec(args.spec),
        load_profiles(profile_path),
        {
            "profile_path": str(profile_path),
            "manifest_path": str(manifest_path),
            "profile_sha256": manifest["profile_sha256"],
            "input_sha256": manifest["input_sha256"],
        },
    )
    result["provenance"]["spec_path"] = str(args.spec.resolve())
    result["provenance"]["spec_sha256"] = sha256(args.spec.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
