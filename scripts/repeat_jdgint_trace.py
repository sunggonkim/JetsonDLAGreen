#!/usr/bin/env python3
"""Repeat a validated JDGINT1 trace for directional performance runs.

The output is a cyclic replay of real, already-preprocessed inputs.  It is
appropriate for load/crossover characterization, but it does not increase
dataset coverage and must not be reported as an accuracy sample expansion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


MAGIC = b"JDGINT1\x00"
HEADER = struct.Struct("<IIQ")
ITERATION = struct.Struct("<I")
HEX = set(b"0123456789abcdef")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> tuple[int, list[tuple[bytes, bytes]]]:
    raw = path.read_bytes()
    header_bytes = len(MAGIC) + HEADER.size
    if len(raw) < header_bytes or raw[: len(MAGIC)] != MAGIC:
        raise ValueError("JDGINT1 trace magic or header differs")
    schema, count, sample_bytes = HEADER.unpack_from(raw, len(MAGIC))
    if schema != 1 or count <= 0 or sample_bytes <= 0:
        raise ValueError("JDGINT1 trace header is invalid")
    offset = header_bytes
    records: list[tuple[bytes, bytes]] = []
    for expected in range(count):
        end = offset + ITERATION.size + 64 + sample_bytes
        if end > len(raw):
            raise ValueError("JDGINT1 record is truncated")
        (iteration,) = ITERATION.unpack_from(raw, offset)
        offset += ITERATION.size
        if iteration != expected:
            raise ValueError("JDGINT1 iterations are not dense")
        digest = raw[offset : offset + 64]
        offset += 64
        payload = raw[offset : offset + sample_bytes]
        offset += sample_bytes
        if len(digest) != 64 or any(value not in HEX for value in digest):
            raise ValueError("JDGINT1 input hash is invalid")
        if hashlib.sha256(payload).hexdigest().encode("ascii") != digest:
            raise ValueError("JDGINT1 payload hash differs")
        records.append((digest, payload))
    if offset != len(raw):
        raise ValueError("JDGINT1 trace has trailing bytes")
    return sample_bytes, records


def repeat(source: Path, output: Path, count: int) -> dict[str, object]:
    if isinstance(count, bool) or count <= 0:
        raise ValueError("output record count must be positive")
    source = source.resolve()
    output = output.resolve()
    sample_bytes, records = _read(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(HEADER.pack(1, count, sample_bytes))
        for iteration in range(count):
            digest, payload = records[iteration % len(records)]
            stream.write(ITERATION.pack(iteration))
            stream.write(digest)
            stream.write(payload)
    return {
        "schema_version": 1,
        "kind": "jdgint1-cyclic-performance-replay",
        "coverage_policy": "cyclic-performance-replay-not-accuracy-expansion",
        "source_trace": str(source),
        "source_trace_sha256": _sha(source),
        "source_records": len(records),
        "output_trace": str(output),
        "output_trace_sha256": _sha(output),
        "output_records": count,
        "sample_bytes": sample_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args()
    result = repeat(args.source, args.output, args.count)
    if args.provenance is not None:
        args.provenance.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.provenance.resolve().write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
