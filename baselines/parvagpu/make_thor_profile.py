#!/usr/bin/env python3
"""Convert isolated jdg-trt-bench results to ParvaGPU profile rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="MODEL=benchmark-result.json",
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    hashes: dict[str, str] = {}
    for item in args.input:
        identity, separator, raw_path = item.partition("=")
        model, segment_separator, segment_text = identity.partition("@")
        segment = int(segment_text) if segment_separator else 1
        path = pathlib.Path(raw_path).resolve()
        hash_key = f"{model}-{segment}g"
        if (
            not separator
            or not model
            or segment not in {1, 2}
            or not path.is_file()
            or hash_key in hashes
        ):
            raise ValueError("invalid or duplicate ParvaGPU profile input")
        raw = json.loads(path.read_text(encoding="utf-8"))
        expected_gpu = {
            1: {"name": "NVIDIA Thor MIG 1g.0gb", "multiprocessors": 8},
            2: {"name": "NVIDIA Thor MIG 2g.0gb", "multiprocessors": 12},
        }[segment]
        if (
            raw.get("role") != "pressure"
            or raw.get("gpu") != expected_gpu
            or raw.get("execution_environment", {}).get(
                "mps_active_thread_percentage"
            )
            != 100
        ):
            raise ValueError("ParvaGPU input is not an isolated Thor MIG/q100 run")
        rows.append(
            {
                "model": model,
                "segment_gpc": segment,
                "batch_size": 1,
                "processes": 1,
                "throughput": raw["throughput_per_second"],
                "latency_ms": raw["release_to_completion"]["p99_ms"],
            }
        )
        hashes[hash_key] = hashlib.sha256(path.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "model", "segment_gpc", "batch_size", "processes",
                "throughput", "latency_ms",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 1,
        "platform": "NVIDIA Thor",
        "mig_profile": (
            "fixed-2g+1g"
            if {int(row["segment_gpc"]) for row in rows} == {1, 2}
            else "1g.0gb"
        ),
        "mps_quota_percent": 100,
        "input_sha256": hashes,
        "profile_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
