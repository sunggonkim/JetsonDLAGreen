#!/usr/bin/env python3
"""Validate the request-indexed ``JDGACT1`` producer activation trace.

The C++ pipeline consumes this exact little-endian format.  Validation is
deliberately independent of TensorRT so schema and raw-replay checks can run
before any GPU smoke.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


MAGIC = b"JDGACT1\x00"
HEADER = struct.Struct("<IIQ")
PREFIX = struct.Struct("<I64sQ")
HEX = set("0123456789abcdef")


def _checksum(payload: bytes) -> int:
    value = 1469598103934665603
    for byte in payload:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def load(path: Path) -> dict[str, Any]:
    raw = path.resolve().read_bytes()
    if len(raw) < len(MAGIC) + HEADER.size or raw[: len(MAGIC)] != MAGIC:
        raise ValueError("activation replay trace magic or header differs")
    schema, count, sample_bytes = HEADER.unpack_from(raw, len(MAGIC))
    if schema != 1 or count <= 0 or count > 1_000_000:
        raise ValueError("activation replay trace header differs")
    if sample_bytes <= 0 or sample_bytes > (1 << 34):
        raise ValueError("activation replay trace sample size is invalid")
    offset = len(MAGIC) + HEADER.size
    rows: list[dict[str, Any]] = []
    for expected in range(count):
        end = offset + PREFIX.size + sample_bytes
        if end > len(raw):
            raise ValueError("activation replay trace record is truncated")
        iteration, input_sha256, activation_checksum = PREFIX.unpack_from(raw, offset)
        offset += PREFIX.size
        if iteration != expected:
            raise ValueError("activation replay trace iterations are not dense")
        digest = input_sha256.decode("ascii")
        if len(digest) != 64 or any(char not in HEX for char in digest):
            raise ValueError("activation replay trace input SHA-256 is invalid")
        payload = raw[offset : offset + sample_bytes]
        offset += sample_bytes
        if _checksum(payload) != activation_checksum:
            raise ValueError("activation replay trace checksum differs from payload")
        rows.append(
            {
                "iteration": iteration,
                "input_sha256": digest,
                "activation_checksum": activation_checksum,
            }
        )
    if offset != len(raw):
        raise ValueError("activation replay trace has trailing bytes")
    return {
        "schema_version": 1,
        "format": "JDGACT1",
        "path": str(path.resolve()),
        "record_count": count,
        "sample_bytes": sample_bytes,
        "records": rows,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    print(json.dumps(load(args.trace), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
