#!/usr/bin/env python3
"""Recompute deadline flags from an immutable application timing trace.

This is useful when the application reference predictions are unchanged but a
new frozen common deadline is selected.  Latencies, inputs, and row order are
copied byte-for-byte; only the derived ``deadline_miss`` field is recomputed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = ("request", "input_sha256", "wall_end_to_end_us", "deadline_miss")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def reclassify(source: Path, output: Path, deadline_us: float) -> dict[str, Any]:
    if not math.isfinite(deadline_us) or deadline_us <= 0.0:
        raise ValueError("deadline_us must be finite and positive")
    source = source.resolve()
    output = output.resolve()
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise ValueError("application timing trace schema differs")
        rows: list[dict[str, str]] = []
        for line_number, row in enumerate(reader, 2):
            try:
                request = int(row["request"])
                latency = float(row["wall_end_to_end_us"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"timing trace:{line_number} has invalid values") from error
            if request < 0 or not math.isfinite(latency) or latency < 0.0:
                raise ValueError(f"timing trace:{line_number} has invalid timing")
            copied = dict(row)
            copied["deadline_miss"] = "1" if latency > deadline_us else "0"
            rows.append(copied)
    if not rows:
        raise ValueError("application timing trace is empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "kind": "p9-reclassified-application-deadline-trace",
        "source_path": str(source),
        "source_sha256": sha256(source),
        "output_path": str(output),
        "output_sha256": sha256(output),
        "deadline_us": deadline_us,
        "records": len(rows),
        "deadline_misses": sum(row["deadline_miss"] == "1" for row in rows),
        "latencies_unchanged": True,
        "labels_unchanged": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-us", type=float, required=True)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    result = reclassify(args.source, args.output, args.deadline_us)
    if args.metadata is not None:
        args.metadata.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.metadata.resolve().write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
