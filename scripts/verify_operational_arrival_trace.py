#!/usr/bin/env python3
"""Validate the operational ``JDGARR1`` release schedule consumed by QUIET."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


MAGIC = b"JDGARR1\x00"
HEADER = struct.Struct("<IIQ")
RECORD = struct.Struct("<IIQ64s64s")
HEX = set("0123456789abcdef")


def _text(field: bytes) -> str:
    value = field.split(b"\x00", 1)[0].decode("ascii")
    if not value:
        raise ValueError("operational arrival trace request_id is empty")
    return value


def load(path: Path) -> dict[str, Any]:
    raw = path.resolve().read_bytes()
    if len(raw) < len(MAGIC) + HEADER.size or raw[: len(MAGIC)] != MAGIC:
        raise ValueError("operational arrival trace magic or header differs")
    schema, count, record_bytes = HEADER.unpack_from(raw, len(MAGIC))
    if schema != 1 or count <= 0 or count > 1_000_000:
        raise ValueError("operational arrival trace header differs")
    if record_bytes != RECORD.size:
        raise ValueError("operational arrival trace record size differs")
    offset = len(MAGIC) + HEADER.size
    rows: list[dict[str, Any]] = []
    for expected in range(count):
        end = offset + RECORD.size
        if end > len(raw):
            raise ValueError("operational arrival trace record is truncated")
        iteration, sequence, release_offset_ns, digest, request_id = RECORD.unpack_from(
            raw, offset
        )
        offset = end
        if sequence != expected:
            raise ValueError("operational arrival trace sequences are not dense")
        input_sha256 = digest.decode("ascii")
        if len(input_sha256) != 64 or any(char not in HEX for char in input_sha256):
            raise ValueError("operational arrival trace input SHA-256 is invalid")
        request = _text(request_id)
        if len(request.encode("ascii")) > 64:
            raise ValueError("operational arrival trace request_id is too long")
        rows.append(
            {
                "iteration": iteration,
                "arrival_sequence": sequence,
                "release_offset_ns": release_offset_ns,
                "input_sha256": input_sha256,
                "request_id": request,
            }
        )
    if offset != len(raw):
        raise ValueError("operational arrival trace has trailing bytes")
    return {
        "schema_version": 1,
        "format": "JDGARR1",
        "path": str(path.resolve()),
        "record_count": count,
        "record_bytes": RECORD.size,
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
