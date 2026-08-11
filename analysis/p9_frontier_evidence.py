#!/usr/bin/env python3
"""Validate correctness evidence consumed by production-wall frontiers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _check_verification_file(
    value: dict[str, Any], row: dict[str, Any], path: Path, system: str
) -> dict[str, Any]:
    evidence = row.get("sota_verification")
    if not isinstance(evidence, dict):
        raise ValueError(f"{path}: {system} lacks checksum verification provenance")
    evidence_path = evidence.get("path")
    evidence_sha = evidence.get("sha256")
    if not isinstance(evidence_path, str) or not evidence_path:
        raise ValueError(f"{path}: {system} verification path is missing")
    if (
        not isinstance(evidence_sha, str)
        or len(evidence_sha) != 64
        or any(char not in "0123456789abcdef" for char in evidence_sha)
    ):
        raise ValueError(f"{path}: {system} verification SHA is invalid")
    verification_path = Path(evidence_path).resolve()
    if not verification_path.is_file() or _sha256(verification_path) != evidence_sha:
        raise ValueError(f"{path}: {system} verification evidence changed or is missing")
    try:
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: {system} verification evidence is invalid JSON") from error
    if not isinstance(verification, dict):
        raise ValueError(f"{path}: {system} verification evidence is not an object")
    if (
        verification.get("status") not in {"passed", "passed-smoke"}
        or verification.get("token_only") is True
        or verification.get("checksum_failures") != 0
        or _nonnegative_int(verification.get("unique_payload_checksums"), "payload checksums") <= 0
        or _nonnegative_int(verification.get("unique_policy_output_checksums"), "output checksums") <= 0
    ):
        raise ValueError(f"{path}: {system} verification lacks valid checksum evidence")
    if verification.get("workload") not in {None, value.get("workload"), "resnet10-layer7-cov-to-control-mlp"}:
        raise ValueError(f"{path}: {system} verification workload differs")
    if not math.isclose(
        _finite(verification.get("deadline_us"), "verification deadline"),
        _finite(value.get("deadline_us"), "row deadline"),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{path}: {system} verification deadline differs")
    if verification.get("requests") != row.get("pipeline_requests"):
        raise ValueError(f"{path}: {system} verification request count differs")
    if verification.get("misses") != row.get("deadline_misses"):
        raise ValueError(f"{path}: {system} verification miss count differs")
    verification_p99 = _finite(verification.get("p99_us"), "verification p99")
    row_p99 = _finite(row.get("wall_pipeline_p99_us", row.get("pipeline_p99_us")), "row p99")
    if not math.isclose(verification_p99, row_p99, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{path}: {system} verification p99 differs")
    return {
        "kind": verification.get("kind"),
        "path": str(verification_path),
        "sha256": evidence_sha,
        "checksum_failures": 0,
        "unique_payload_checksums": verification["unique_payload_checksums"],
        "unique_policy_output_checksums": verification["unique_policy_output_checksums"],
    }


def validate_correctness(value: dict[str, Any], row: dict[str, Any], path: Path) -> dict[str, Any]:
    if value.get("checksum_mode") != "inline" or row.get("correctness_validated") is not True:
        raise ValueError(f"{path}: inline correctness validation is required")
    failures = row.get("checksum_failures")
    if failures is not None and failures != 0:
        raise ValueError(f"{path}: checksum failures are nonzero")
    payload = row.get("unique_payload_checksums")
    output = row.get("unique_policy_output_checksums")
    if payload is not None and output is not None:
        payload = _nonnegative_int(payload, "payload checksums")
        output = _nonnegative_int(output, "output checksums")
        if payload <= 0 or output <= 0:
            raise ValueError(f"{path}: checksum diversity is empty")
        return {
            "mode": "inline",
            "checksum_failures": 0,
            "unique_payload_checksums": payload,
            "unique_policy_output_checksums": output,
            "source": "row",
        }
    return {"mode": "inline", "source": "sota_verification", **_check_verification_file(value, row, path, row.get("system", "unknown"))}
