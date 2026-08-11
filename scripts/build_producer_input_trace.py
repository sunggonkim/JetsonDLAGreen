#!/usr/bin/env python3
"""Pack preprocessed TensorRT input tensors into the ``JDGINT1`` trace.

The benchmark consumes fixed-size tensor bytes, not image files.  The sample
list therefore names already-preprocessed tensors and carries their external
SHA-256.  This tool never infers labels or silently resizes samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


MAGIC = b"JDGINT1\x00"
HEADER = struct.Struct("<IIQ")
RECORD = struct.Struct("<I")
HEX = set("0123456789abcdef")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_samples(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("sample list is not newline-complete")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_inodes: set[tuple[int, int]] = set()
    for line_number, line in enumerate(raw.splitlines(), 1):
        value = json.loads(line)
        required = {"iteration", "sample_id", "path", "input_sha256"}
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError(f"sample list:{line_number} schema differs")
        iteration = value["iteration"]
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise ValueError(f"sample list:{line_number} iteration is invalid")
        sample_id = value["sample_id"]
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen_ids:
            raise ValueError(f"sample list:{line_number} sample_id is invalid or duplicated")
        digest = value["input_sha256"]
        if (
            not isinstance(digest, str) or len(digest) != 64
            or any(char not in HEX for char in digest)
        ):
            raise ValueError(f"sample list:{line_number} input_sha256 is invalid")
        sample = Path(value["path"]).resolve()
        stat = sample.stat()
        inode = (stat.st_dev, stat.st_ino)
        if inode in seen_inodes:
            raise ValueError(f"sample list:{line_number} reuses an input inode")
        if not sample.is_file() or _sha(sample) != digest:
            raise ValueError(f"sample list:{line_number} input SHA differs")
        value = dict(value)
        value["path"] = str(sample)
        value["bytes"] = sample.stat().st_size
        rows.append(value)
        seen_ids.add(sample_id)
        seen_inodes.add(inode)
    if not rows:
        raise ValueError("sample list is empty")
    rows.sort(key=lambda row: row["iteration"])
    if [row["iteration"] for row in rows] != list(range(len(rows))):
        raise ValueError("sample iterations must be dense and zero-based")
    sample_bytes = rows[0]["bytes"]
    if sample_bytes <= 0 or any(row["bytes"] != sample_bytes for row in rows):
        raise ValueError("all producer input tensors must have the same nonzero size")
    return rows


def build(sample_list: Path, output: Path, provenance: Path | None = None) -> dict[str, Any]:
    rows = _load_samples(sample_list.resolve())
    sample_bytes = rows[0]["bytes"]
    with output.resolve().open("wb") as stream:
        stream.write(MAGIC)
        stream.write(HEADER.pack(1, len(rows), sample_bytes))
        for row in rows:
            path = Path(row["path"])
            stream.write(RECORD.pack(row["iteration"]))
            stream.write(row["input_sha256"].encode("ascii"))
            stream.write(path.read_bytes())
    result = {
        "schema_version": 1,
        "kind": "p9-producer-input-trace",
        "format": "JDGINT1",
        "path": str(output.resolve()),
        "sample_list_path": str(sample_list.resolve()),
        "sample_list_sha256": _sha(sample_list.resolve()),
        "record_count": len(rows),
        "sample_bytes": sample_bytes,
        "input_sha256": [row["input_sha256"] for row in rows],
        "sample_ids": [row["sample_id"] for row in rows],
    }
    result["sha256"] = _sha(output.resolve())
    if provenance is not None:
        provenance.resolve().parent.mkdir(parents=True, exist_ok=True)
        provenance.resolve().write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args()
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    result = build(args.sample_list, args.output, args.provenance)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
