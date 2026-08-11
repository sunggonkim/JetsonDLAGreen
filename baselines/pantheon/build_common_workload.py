#!/usr/bin/env python3
"""Encode the current operational arrival trace for Pantheon's protobuf API."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path
from typing import Any, Iterable


MAGIC = b"JDGARR1\x00"
HEADER = struct.Struct("<IIQ")
RECORD = struct.Struct("<IIQ")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_arrivals(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if len(raw) < 8 + HEADER.size or raw[:8] != MAGIC:
        raise ValueError("operational arrival trace magic differs")
    schema, count, record_bytes = HEADER.unpack_from(raw, 8)
    if schema != 1 or record_bytes != 144 or count <= 0:
        raise ValueError("operational arrival trace header differs")
    offset = 8 + HEADER.size
    rows: list[dict[str, Any]] = []
    first_iteration: int | None = None
    for expected in range(count):
        if offset + record_bytes > len(raw):
            raise ValueError("operational arrival trace is truncated")
        iteration, arrival, release_ns = RECORD.unpack_from(raw, offset)
        offset += RECORD.size
        input_sha256 = raw[offset:offset + 64].decode("ascii")
        offset += 64
        request_id = raw[offset:offset + 64].split(b"\0", 1)[0].decode("ascii")
        offset += 64
        if arrival != expected:
            raise ValueError("operational arrivals are not dense")
        if first_iteration is None:
            first_iteration = iteration
        if iteration != first_iteration + expected:
            raise ValueError("operational arrival iterations are not contiguous")
        if len(input_sha256) != 64 or len(request_id) == 0:
            raise ValueError("operational arrival identity is invalid")
        rows.append({
            "iteration": iteration,
            "arrival_sequence": arrival,
            "release_us": release_ns // 1000,
            "input_sha256": input_sha256,
            "request_id": request_id,
        })
    if offset != len(raw):
        raise ValueError("operational arrival trace has trailing bytes")
    return rows


def build(
    arrival_trace: Path,
    output: Path,
    *,
    proto_dir: Path,
    model_name: str,
    deadline_us: int,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"refusing existing Pantheon workload: {output}")
    if deadline_us <= 0 or not model_name:
        raise ValueError("Pantheon workload contract is invalid")
    rows = read_arrivals(arrival_trace)
    text = "".join(
        f'workload {{ model_name: "{model_name}" release: {row["release_us"]} '
        f'deadline: {row["release_us"] + deadline_us} id: {row["arrival_sequence"]} }}\n'
        for row in rows
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    text_path = output.with_suffix(".textproto")
    text_path.write_text(text, encoding="ascii")
    with output.open("wb") as stream:
        subprocess.run(
            [
                "protoc", f"--proto_path={proto_dir}", "--encode=Workloads",
                str(proto_dir / "workload.proto"),
            ],
            input=text.encode("ascii"), stdout=stream, check=True,
        )
    result = {
        "schema_version": 1,
        "kind": "pantheon-common-workload-protobuf",
        "model_name": model_name,
        "deadline_us": deadline_us,
        "request_count": len(rows),
        "input_trace_iteration_offset": rows[0]["iteration"],
        "arrival_trace_path": str(arrival_trace.resolve()),
        "arrival_trace_sha256": sha256(arrival_trace),
        "workload_textproto_path": str(text_path.resolve()),
        "workload_textproto_sha256": sha256(text_path),
        "workload_path": str(output.resolve()),
        "workload_sha256": sha256(output),
        "requests": rows,
        "numeric_comparison_allowed": False,
    }
    output.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrival-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proto-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="resnet50-imagenette")
    parser.add_argument("--deadline-us", type=int, required=True)
    args = parser.parse_args(argv)
    result = build(
        args.arrival_trace.resolve(), args.output.resolve(),
        proto_dir=args.proto_dir.resolve(), model_name=args.model_name,
        deadline_us=args.deadline_us,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
