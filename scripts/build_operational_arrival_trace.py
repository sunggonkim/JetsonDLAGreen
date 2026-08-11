#!/usr/bin/env python3
"""Build a deterministic, request-bound ``JDGARR1`` operational schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

from verify_operational_arrival_trace import HEADER, MAGIC, RECORD, load


REQUEST_KEYS = {
    "schema_version", "iteration", "request_id", "arrival_sequence",
    "input_sha256", "expected_label",
}
INPUT_MAGIC = b"JDGINT1\x00"
INPUT_HEADER = struct.Struct("<IIQ")
INPUT_PREFIX = struct.Struct("<I64s")


def _sha(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError("input_sha256 must be lowercase SHA-256")
    return value


def _load_input_trace(path: Path) -> list[str]:
    raw = path.resolve().read_bytes()
    if len(raw) < len(INPUT_MAGIC) + INPUT_HEADER.size or raw[: len(INPUT_MAGIC)] != INPUT_MAGIC:
        raise ValueError("producer input trace magic or header differs")
    _schema, count, sample_bytes = INPUT_HEADER.unpack_from(raw, len(INPUT_MAGIC))
    if _schema != 1 or count <= 0 or sample_bytes <= 0:
        raise ValueError("producer input trace header differs")
    offset = len(INPUT_MAGIC) + INPUT_HEADER.size
    hashes: list[str] = []
    for expected in range(count):
        end = offset + INPUT_PREFIX.size + sample_bytes
        if end > len(raw):
            raise ValueError("producer input trace record is truncated")
        iteration, digest = INPUT_PREFIX.unpack_from(raw, offset)
        if iteration != expected:
            raise ValueError("producer input trace iterations are not dense")
        hashes.append(_sha(digest.decode("ascii")))
        offset = end
    if offset != len(raw):
        raise ValueError("producer input trace has trailing bytes")
    return hashes


def _load_requests(path: Path) -> list[dict[str, Any]]:
    raw = path.resolve().read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("request trace is not newline-complete")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(raw.splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"request trace:{number} is invalid JSON") from error
        if not isinstance(row, dict) or set(row) != REQUEST_KEYS:
            raise ValueError(f"request trace:{number} schema differs")
        if row.get("schema_version") != 1:
            raise ValueError(f"request trace:{number} schema version differs")
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id or len(request_id.encode("ascii")) > 64:
            raise ValueError("request_id must be nonempty ASCII of at most 64 bytes")
        if not isinstance(row.get("iteration"), int) or isinstance(row.get("iteration"), bool):
            raise ValueError("request iteration must be an integer")
        if not isinstance(row.get("arrival_sequence"), int) or isinstance(row.get("arrival_sequence"), bool):
            raise ValueError("arrival_sequence must be an integer")
        _sha(row.get("input_sha256"))
        rows.append(row)
    if not rows:
        raise ValueError("request trace is empty")
    if [row["arrival_sequence"] for row in rows] != list(range(len(rows))):
        raise ValueError("arrival_sequence must be dense and ordered")
    return rows


def build(
    *, output: Path, period_us: int, warmup: int = 0,
    request_trace: Path | None = None, producer_input_trace: Path | None = None,
    requests: int | None = None,
) -> dict[str, Any]:
    if period_us <= 0:
        raise ValueError("period_us must be positive")
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if (request_trace is None) == (producer_input_trace is None):
        raise ValueError("choose exactly one source trace")
    if request_trace is not None:
        rows = _load_requests(request_trace)
        records = [
            (int(row["iteration"]), int(row["arrival_sequence"]), _sha(row["input_sha256"]), str(row["request_id"]))
            for row in rows
        ]
    else:
        hashes = _load_input_trace(producer_input_trace)  # type: ignore[arg-type]
        measured = len(hashes) - warmup if requests is None else requests
        if measured <= 0 or warmup + measured > len(hashes):
            raise ValueError("warmup/requests exceed producer input trace")
        records = [
            (warmup + index, index, hashes[warmup + index], f"request-{index:06d}")
            for index in range(measured)
        ]
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(HEADER.pack(1, len(records), RECORD.size))
        for iteration, sequence, digest, request_id in records:
            encoded = request_id.encode("ascii")
            stream.write(RECORD.pack(
                iteration, sequence, period_us * 1000 * sequence,
                digest.encode("ascii"), encoded.ljust(64, b"\x00"),
            ))
    result = load(output)
    result["release_period_us"] = period_us
    result["warmup"] = warmup
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--request-trace", type=Path)
    source.add_argument("--producer-input-trace", type=Path)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--requests", type=int)
    parser.add_argument("--period-us", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        output=args.output, period_us=args.period_us, warmup=args.warmup,
        request_trace=args.request_trace,
        producer_input_trace=args.producer_input_trace,
        requests=args.requests,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
