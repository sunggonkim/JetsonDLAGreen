#!/usr/bin/env python3
"""Build the immutable common-workload contract shared by every comparator arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DATASET_KEYS = {"schema_version", "sample_id", "input_sha256", "expected_label"}
REQUEST_KEYS = {
    "schema_version", "iteration", "request_id", "arrival_sequence",
    "input_sha256", "expected_label",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def _hex(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _read_jsonl(path: Path, expected_keys: set[str], label: str) -> list[dict[str, Any]]:
    raw = path.resolve().read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{label} is not newline-complete")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label}:{line_number} is invalid JSON") from error
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise ValueError(f"{label}:{line_number} schema differs")
        if value.get("schema_version") != 1:
            raise ValueError(f"{label}:{line_number} schema version differs")
        rows.append(value)
    if not rows:
        raise ValueError(f"{label} is empty")
    return rows


def build(
    *,
    workload_id: str,
    topology: str,
    placement: str,
    input_tensor: str,
    payload_bytes: int,
    arrival_trace: Path,
    dataset_manifest: Path,
    producer_input_trace: Path | None = None,
    operational_arrival_trace: Path | None = None,
) -> dict[str, Any]:
    if not workload_id or not topology or not placement or not input_tensor:
        raise ValueError("workload, topology, placement, and input tensor are required")
    if isinstance(payload_bytes, bool) or not isinstance(payload_bytes, int) or payload_bytes <= 0:
        raise ValueError("payload_bytes must be a positive integer")
    arrival = _read_jsonl(arrival_trace, REQUEST_KEYS, "arrival trace")
    dataset = _read_jsonl(dataset_manifest, DATASET_KEYS, "dataset manifest")
    labels: dict[str, str] = {}
    sample_ids: set[str] = set()
    for row in dataset:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise ValueError("dataset sample IDs must be unique and nonempty")
        sample_ids.add(sample_id)
        digest = _hex(row.get("input_sha256"), "dataset input_sha256")
        label = row.get("expected_label")
        if not isinstance(label, str) or not label:
            raise ValueError("dataset expected_label must be nonempty")
        if digest in labels and labels[digest] != label:
            raise ValueError("dataset contains conflicting labels for one input")
        labels[digest] = label

    arrivals: list[int] = []
    request_ids: set[str] = set()
    for row in arrival:
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id or request_id in request_ids:
            raise ValueError("arrival request IDs must be unique and nonempty")
        request_ids.add(request_id)
        sequence = row.get("arrival_sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise ValueError("arrival_sequence must be an integer")
        arrivals.append(sequence)
        digest = _hex(row.get("input_sha256"), "arrival input_sha256")
        if digest not in labels or labels[digest] != row.get("expected_label"):
            raise ValueError("arrival input/label is not bound to the dataset manifest")
        iteration = row.get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise ValueError("arrival iteration is invalid")
    if sorted(arrivals) != list(range(len(arrival))):
        raise ValueError("arrival_sequence must be dense and zero-based")

    arrival_path = arrival_trace.resolve()
    dataset_path = dataset_manifest.resolve()
    result = {
        "schema_version": 1,
        "workload_id": workload_id,
        "topology": topology,
        "placement": placement,
        "input_tensor": input_tensor,
        "payload_bytes": payload_bytes,
        "request_count": len(arrival),
        "arrival_trace_path": str(arrival_path),
        "arrival_trace_sha256": sha256(arrival_path),
        "dataset_manifest_path": str(dataset_path),
        "dataset_manifest_sha256": sha256(dataset_path),
        "binding": "request-input-hashes-and-external-labels",
    }
    if producer_input_trace is not None:
        producer_path = producer_input_trace.resolve()
        if not producer_path.is_file() or producer_path.stat().st_size <= 0:
            raise ValueError("producer_input_trace must be a non-empty file")
        result["producer_input_trace_path"] = str(producer_path)
        result["producer_input_trace_sha256"] = sha256(producer_path)
        result["producer_input_binding"] = "JDGINT1-bytes-bound-to-arrival-contract"
    if operational_arrival_trace is not None:
        operational_path = operational_arrival_trace.resolve()
        if not operational_path.is_file() or operational_path.stat().st_size <= 0:
            raise ValueError("operational_arrival_trace must be a non-empty file")
        result["operational_arrival_trace_path"] = str(operational_path)
        result["operational_arrival_trace_sha256"] = sha256(operational_path)
        result["operational_arrival_binding"] = (
            "JDGARR1-release-offsets-consumed-by-pipeline"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--topology", default="fixed-2g+1g")
    parser.add_argument("--placement", required=True)
    parser.add_argument("--input-tensor", required=True)
    parser.add_argument("--payload-bytes", type=int, required=True)
    parser.add_argument("--arrival-trace", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument(
        "--producer-input-trace", type=Path,
        help="optional JDGINT1 tensor trace whose bytes are frozen with this contract",
    )
    parser.add_argument(
        "--operational-arrival-trace", type=Path,
        help="optional JDGARR1 release schedule consumed by the pipeline",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        workload_id=args.workload_id,
        topology=args.topology,
        placement=args.placement,
        input_tensor=args.input_tensor,
        payload_bytes=args.payload_bytes,
        arrival_trace=args.arrival_trace,
        dataset_manifest=args.dataset_manifest,
        producer_input_trace=args.producer_input_trace,
        operational_arrival_trace=args.operational_arrival_trace,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
