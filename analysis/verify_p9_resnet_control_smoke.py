#!/usr/bin/env python3
"""Verify the real ResNet10 Layer7_cov to TensorRT control-MLP smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


EXPECTED_COLUMNS = [
    "request",
    "producer_compute_us",
    "producer_copy_us",
    "producer_validation_us",
    "notification_us",
    "consumer_validation_us",
    "consumer_copy_us",
    "edge_transport_us",
    "consumer_compute_us",
    "output_verification_us",
    "validation_excluded_end_to_end_us",
    "wall_end_to_end_us",
    "deadline_miss",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"{path.name} is empty or newline incomplete")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise ValueError("MIG environment differs")
        values[key] = value
    return values


def verify_hashes(
    path: Path, *, allow_historical_mismatch: bool = False
) -> tuple[dict[str, str], bool]:
    hashes: dict[str, str] = {}
    stale = False
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError("SHA256SUMS record differs")
        expected, raw_path = match.groups()
        artifact = Path(raw_path)
        if not artifact.is_file():
            raise ValueError("artifact SHA-256 path is missing")
        if digest(artifact) != expected:
            if not allow_historical_mismatch:
                raise ValueError("artifact SHA-256 differs")
            stale = True
        hashes[str(artifact)] = expected
    if len(hashes) != 3:
        raise ValueError("artifact manifest must bind binary, source, and engine")
    return hashes, stale


def finite_nonnegative(text: str, name: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"{name} is not numeric") from error
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} is not finite nonnegative")
    return value


def verify(directory: Path) -> dict[str, Any]:
    result = load_object(directory / "result.json")
    env = parse_env(directory / "mig.env")
    inventory = (directory / "gpu-inventory.txt").read_text(encoding="utf-8")
    # Older preserved hardware evidence points at absolute source/binary paths
    # that may legitimately have changed after a new measurement contract was
    # introduced. Keep that evidence replayable but label it stale; new runs
    # must use a copied, current manifest and therefore remain strict.
    historical = not bool(result.get("artifact_binding_mode"))
    hashes, stale_manifest = verify_hashes(
        directory / "SHA256SUMS", allow_historical_mismatch=historical
    )
    small = env.get("JDG_MIG_SMALL_UUID")
    big = env.get("JDG_MIG_BIG_UUID")
    if (
        result.get("schema_version") != 1
        or result.get("status") != "ok"
        or result.get("pipeline") != "resnet10-layer7-cov-to-control-mlp"
        or result.get("transport")
        != "registered-shared-sysmem-direct-binding"
        or result.get("producer_uuid") != small
        or result.get("consumer_uuid") != big
        or result.get("producer_sms") != 8
        or result.get("consumer_sms") != 12
        or result.get("producer_quota") != 100
        or result.get("consumer_quota") != 100
        or result.get("payload_bytes") != 14_720
        or result.get("payload_shape") != [1, 4, 23, 40]
        or result.get("producer_output_tensor") != "Layer7_cov"
        or result.get("consumer_input_tensor") != "features"
        or result.get("consumer_output_tensor") != "policy_output"
        or result.get("iterations") != 100
        or result.get("checksum_failures") != 0
        or result.get("unique_payload_checksums", 0) < 2
        or result.get("unique_policy_output_checksums", 0) < 2
        or result.get("orion", {}).get("enabled") is not False
    ):
        raise ValueError("ResNet control pipeline result differs")
    if (
        not isinstance(small, str)
        or not isinstance(big, str)
        or small == big
        or f"MIG 1g.0gb      Device  1: (UUID: {small})" not in inventory
        or f"MIG 2g.0gb      Device  0: (UUID: {big})" not in inventory
    ):
        raise ValueError("active fixed 2g+1g MIG inventory differs")

    with (directory / "trace.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError("pipeline trace schema differs")
        rows = list(reader)
    if len(rows) != 100:
        raise ValueError("pipeline trace must contain 100 measured requests")
    expected_first = int(result.get("warmup", -1))
    for offset, row in enumerate(rows):
        if int(row["request"]) != expected_first + offset:
            raise ValueError("pipeline request sequence differs")
        if row["deadline_miss"] not in {"0", "1"}:
            raise ValueError("pipeline checksum trace differs")
        for name in EXPECTED_COLUMNS[1:12]:
            finite_nonnegative(row[name], name)
    with (directory / "checksums.csv").open(newline="", encoding="utf-8") as stream:
        checksum_reader = csv.DictReader(stream)
        if checksum_reader.fieldnames != [
            "request", "payload_checksum", "output_checksum"
        ]:
            raise ValueError("checksum trace schema differs")
        checksum_rows = list(checksum_reader)
    if len(checksum_rows) != 100:
        raise ValueError("checksum trace must contain 100 measured requests")
    payloads: set[int] = set()
    outputs: set[int] = set()
    for offset, row in enumerate(checksum_rows):
        if int(row["request"]) != expected_first + offset:
            raise ValueError("checksum request sequence differs")
        payload = int(row["payload_checksum"])
        output = int(row["output_checksum"])
        if payload <= 0 or output <= 0:
            raise ValueError("pipeline checksum trace differs")
        payloads.add(payload)
        outputs.add(output)
    if len(payloads) != result["unique_payload_checksums"] or len(outputs) != result[
        "unique_policy_output_checksums"
    ]:
        raise ValueError("pipeline checksum cardinality differs")
    return {
        "schema_version": 1,
        "kind": "p9-resnet-layer7-control-mlp-cross-mig-smoke",
        "status": "passed",
        "artifact_provenance_status": (
            "historical-manifest-stale" if stale_manifest else "current-manifest"
        ),
        "requests": len(rows),
        "producer": {"mig_profile": "1g.0gb", "sms": 8, "uuid": small},
        "consumer": {"mig_profile": "2g.0gb", "sms": 12, "uuid": big},
        "edge": {
            "producer_tensor": "Layer7_cov",
            "consumer_tensor": "features",
            "shape": [1, 4, 23, 40],
            "bytes": 14_720,
            "transport": result["transport"],
        },
        "checksum_failures": 0,
        "unique_payload_checksums": len(payloads),
        "unique_policy_output_checksums": len(outputs),
        "p99_us": result["end_to_end_us"]["p99"],
        "edge_p99_us": result["stage_latency_us"]["edge_transport_p99"],
        "artifacts_sha256": hashes,
        "result_sha256": digest(directory / "result.json"),
        "trace_sha256": digest(directory / "trace.csv"),
        "checksum_trace_sha256": digest(directory / "checksums.csv"),
        "token_only": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = verify(args.result_dir.resolve())
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
