#!/usr/bin/env python3
"""Parse the optional post-completion TensorRT output trace.

The producer benchmark writes a small binary container after each consumer
completion.  This parser validates truncation and duplicate iterations before
deriving an output checksum/argmax.  It does not attach labels; that remains
the responsibility of the external dataset manifest and accuracy gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any


MAGIC = b"JDGOUT1\x00"
ASR_MAGIC = b"JDGASR1\x00"
HEADER = struct.Struct("<I")
SIZE = struct.Struct("<Q")
ITERATION = struct.Struct("<I")
ASR_HEADER = struct.Struct("<II")
ASR_RECORD = struct.Struct("<II")
ASR_TOKEN = struct.Struct("<I")
WHISPER_VOCAB_SIZE = 51865
WHISPER_MAX_TOKENS = 128


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _argmax_float32(data: bytes) -> int:
    if len(data) == 0 or len(data) % 4:
        raise ValueError("float32 output is empty or not 4-byte aligned")
    values = struct.unpack("<" + "f" * (len(data) // 4), data)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("float32 output contains a non-finite value")
    return max(range(len(values)), key=values.__getitem__)


def _parse_asr(raw: bytes, path: Path) -> dict[str, Any]:
    """Parse the variable-length token container emitted by the ASR runner."""
    offset = len(ASR_MAGIC)
    if len(raw) < offset + ASR_HEADER.size:
        raise ValueError("JDGASR1 trace is truncated")
    schema, record_count = ASR_HEADER.unpack_from(raw, offset)
    offset += ASR_HEADER.size
    if schema != 1:
        raise ValueError("JDGASR1 schema differs")
    if record_count <= 0 or record_count > 1_000_000:
        raise ValueError("JDGASR1 record count is invalid")
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    output_sizes: set[int] = set()
    for _ in range(record_count):
        if offset + ASR_RECORD.size > len(raw):
            raise ValueError("JDGASR1 record header is truncated")
        iteration, token_count = ASR_RECORD.unpack_from(raw, offset)
        offset += ASR_RECORD.size
        if iteration in seen:
            raise ValueError("JDGASR1 repeats an iteration")
        if token_count > WHISPER_MAX_TOKENS:
            raise ValueError("JDGASR1 token count is invalid")
        token_bytes = token_count * ASR_TOKEN.size
        if offset + token_bytes > len(raw):
            raise ValueError("JDGASR1 record is truncated")
        tokens = list(struct.unpack_from("<" + "I" * token_count, raw, offset))
        if any(token >= WHISPER_VOCAB_SIZE for token in tokens):
            raise ValueError("JDGASR1 token id is outside Whisper vocabulary")
        offset += token_bytes
        seen.add(iteration)
        output_sizes.add(token_bytes)
        records.append({
            "iteration": iteration,
            "outputs": [{
                "bytes": token_bytes,
                "sha256": _sha(raw[offset - token_bytes:offset]),
                "tokens": tokens,
            }],
        })
    if offset != len(raw):
        raise ValueError("JDGASR1 has trailing bytes")
    records.sort(key=lambda record: record["iteration"])
    return {
        "schema_version": schema,
        "kind": "p9-application-output-trace",
        "format": "JDGASR1",
        "task": "asr",
        "path": str(path.resolve()),
        "sha256": _sha(raw),
        "output_count": 1,
        "output_sizes": sorted(output_sizes),
        "variable_output_sizes": True,
        "records": records,
        "record_count": len(records),
        "float32_argmax": False,
        "float32_values": False,
    }


def parse(
    path: Path, *, float32_output: bool = False, float32_values: bool = False,
) -> dict[str, Any]:
    raw = path.resolve().read_bytes()
    if len(raw) >= len(ASR_MAGIC) and raw[:len(ASR_MAGIC)] == ASR_MAGIC:
        return _parse_asr(raw, path)
    if len(raw) < len(MAGIC) + HEADER.size:
        raise ValueError("application output trace is truncated")
    if raw[:len(MAGIC)] != MAGIC:
        raise ValueError("application output trace magic differs")
    offset = len(MAGIC)
    (output_count,) = HEADER.unpack_from(raw, offset)
    offset += HEADER.size
    if output_count <= 0 or output_count > 64:
        raise ValueError("application output trace output count is invalid")
    output_sizes: list[int] = []
    for _ in range(output_count):
        if offset + SIZE.size > len(raw):
            raise ValueError("application output trace header is truncated")
        (size,) = SIZE.unpack_from(raw, offset)
        offset += SIZE.size
        if size <= 0 or size > (1 << 34):
            raise ValueError("application output trace tensor size is invalid")
        output_sizes.append(size)
    record_size = ITERATION.size + sum(output_sizes)
    if record_size <= ITERATION.size or (len(raw) - offset) % record_size:
        raise ValueError("application output trace has a partial record")
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    while offset < len(raw):
        (iteration,) = ITERATION.unpack_from(raw, offset)
        offset += ITERATION.size
        if iteration in seen:
            raise ValueError("application output trace repeats an iteration")
        seen.add(iteration)
        outputs: list[dict[str, Any]] = []
        for size in output_sizes:
            value = raw[offset:offset + size]
            offset += size
            item: dict[str, Any] = {"bytes": size, "sha256": _sha(value)}
            if float32_output or float32_values:
                values = struct.unpack("<" + "f" * (len(value) // 4), value)
                if any(not math.isfinite(number) for number in values):
                    raise ValueError("float32 output contains a non-finite value")
                if float32_output:
                    item["argmax"] = max(range(len(values)), key=values.__getitem__)
                if float32_values:
                    item["values"] = list(values)
            outputs.append(item)
        records.append({"iteration": iteration, "outputs": outputs})
    records.sort(key=lambda record: record["iteration"])
    return {
        "schema_version": 1,
        "kind": "p9-application-output-trace",
        "path": str(path.resolve()),
        "sha256": _sha(raw),
        "output_count": output_count,
        "output_sizes": output_sizes,
        "records": records,
        "record_count": len(records),
        "float32_argmax": float32_output,
        "float32_values": float32_values,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--float32-output", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = parse(args.trace, float32_output=args.float32_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
