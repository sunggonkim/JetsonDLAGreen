#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_SMS = [2, 4, 6, 8]
DRIVER_KEYS = {
    "schema_version", "sequence", "api", "tid", "start_monotonic_ns",
    "end_monotonic_ns", "function", "stream", "grid", "block",
    "shared_mem_bytes", "attributes", "result",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def verify(directory: Path, engine: Path) -> dict[str, Any]:
    result_path = directory / "result.json"
    stderr_path = directory / "stderr.txt"
    trace_path = directory / "driver-launches.jsonl"
    result = load(result_path)
    outputs = result.get("output_checksums")
    source = result.get("activation_source_checksum")
    destination = result.get("activation_destination_checksum")
    post_copy = result.get("post_copy_output_checksum")
    if (
        result.get("schema_version") != 1
        or result.get("kind") != "bless-thor-trt-activation-replica-smoke"
        or result.get("status") != "passed"
        or result.get("affinity_domain_sms") != EXPECTED_SMS
        or isinstance(result.get("activation_bytes"), bool)
        or not isinstance(result.get("activation_bytes"), int)
        or result["activation_bytes"] <= 0
        or not isinstance(outputs, list)
        or len(outputs) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in outputs)
        or len(set(outputs)) != 1
        or isinstance(source, bool)
        or not isinstance(source, int)
        or source <= 0
        or destination != source
        or post_copy != outputs[0]
        or result.get("restricted_to_unrestricted_copy") is not True
    ):
        raise ValueError("BLESS TensorRT activation result differs")

    raw = trace_path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("BLESS activation driver trace is empty or truncated")
    records = [json.loads(line) for line in raw.splitlines()]
    previous_end = -1
    signatures: list[tuple[Any, ...]] = []
    for sequence, record in enumerate(records):
        if (
            not isinstance(record, dict)
            or set(record) != DRIVER_KEYS
            or record.get("schema_version") != 1
            or record.get("sequence") != sequence
            or record.get("api") not in {
                "cuLaunchKernel", "cuLaunchKernel_ptsz",
                "cuLaunchKernelEx", "cuLaunchKernelEx_ptsz",
            }
            or record.get("result") != 0
        ):
            raise ValueError("BLESS activation driver launch differs")
        start = record.get("start_monotonic_ns")
        end = record.get("end_monotonic_ns")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or end < start
            or start < previous_end
        ):
            raise ValueError("BLESS activation driver clocks differ")
        previous_end = end
        signatures.append(
            (
                record["api"], tuple(record["grid"]), tuple(record["block"]),
                record["shared_mem_bytes"], record["attributes"],
            )
        )
    inference_count = 5
    if not records or len(records) % inference_count != 0:
        raise ValueError("BLESS activation launch count differs")
    launches_per_inference = len(records) // inference_count
    canonical = signatures[:launches_per_inference]
    if any(
        signatures[index * launches_per_inference : (index + 1) * launches_per_inference]
        != canonical
        for index in range(1, inference_count)
    ):
        raise ValueError("BLESS activation replica launch sequences differ")
    stderr = stderr_path.read_text(encoding="utf-8").lower()
    if "error:" in stderr or "launchfailure" in stderr or "launch failure" in stderr:
        raise ValueError("BLESS activation stderr reports a failure")
    return {
        "schema_version": 1,
        "kind": "bless-thor-trt-activation-replica-functional-gate",
        "status": "passed",
        "numeric_comparison_allowed": False,
        "engine": {"path": str(engine), "sha256": sha256(engine)},
        "affinity_domain_sms": EXPECTED_SMS,
        "activation_bytes": result["activation_bytes"],
        "output_checksum": outputs[0],
        "activation_checksum": source,
        "restricted_to_unrestricted_peer_copy": True,
        "post_copy_inference_passed": True,
        "driver_launch_records": len(records),
        "launches_per_inference": launches_per_inference,
        "replica_launch_sequences_identical": True,
        "evidence_sha256": {
            "result.json": sha256(result_path),
            "stderr.txt": sha256(stderr_path),
            "driver-launches.jsonl": sha256(trace_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = verify(args.result_dir.resolve(), args.engine.resolve())
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
