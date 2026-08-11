#!/usr/bin/env python3
"""Compare checksum modes without promoting timing-only runs to a frontier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


MODES = ("inline", "sampled", "off")
SYSTEMS = ("QUIET", "NVIDIA MPS")


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _read(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.resolve().read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{path} is not newline-complete")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value, hashlib.sha256(raw).hexdigest()


def _validate_row(mode: str, row: dict[str, Any], path: Path) -> dict[str, Any]:
    system = row.get("system")
    if system not in SYSTEMS:
        raise ValueError(f"{path}: unsupported system {system!r}")
    requests = row.get("pipeline_requests")
    misses = row.get("deadline_misses")
    if isinstance(requests, bool) or not isinstance(requests, int) or requests <= 0:
        raise ValueError(f"{path}: invalid request count")
    if isinstance(misses, bool) or not isinstance(misses, int) or not 0 <= misses <= requests:
        raise ValueError(f"{path}: invalid deadline misses")
    p99 = _finite(row.get("wall_pipeline_p99_us", row.get("pipeline_p99_us")), f"{path} p99")
    validated = row.get("correctness_validated")
    if mode == "inline":
        if validated is not True or row.get("checksum_failures", 0) not in (0, None):
            raise ValueError(f"{path}: inline mode lacks correctness validation")
        if min(row.get("unique_payload_checksums", 0), row.get("unique_policy_output_checksums", 0)) < 2:
            raise ValueError(f"{path}: inline mode lacks checksum diversity")
    elif validated is not False:
        raise ValueError(f"{path}: timing-only mode must report correctness_validated=false")
    return {
        "system": system,
        "p99_us": p99,
        "requests": requests,
        "deadline_misses": misses,
        "dmr": misses / requests,
        "background_goodput_rps": _finite(row.get("background_goodput_rps"), f"{path} goodput"),
        "correctness_validated": validated,
    }


def summarize(paths: Iterable[Path]) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    contract: tuple[Any, ...] | None = None
    for path in paths:
        value, digest = _read(path)
        mode = value.get("checksum_mode")
        if mode not in MODES:
            raise ValueError(f"{path}: unsupported checksum mode")
        if (
            value.get("kind") != "p9-dependent-small-stress-smoke"
            or value.get("workload") != "resnet-control"
            or value.get("latency_contract") != "production-wall-arrival-to-completion"
            or value.get("deadline_mode") != "wall"
        ):
            raise ValueError(f"{path}: outside production-wall probe contract")
        lock = value.get("deadline_lock")
        if not isinstance(lock, dict) or not isinstance(lock.get("sha256"), str):
            raise ValueError(f"{path}: missing deadline lock")
        current = (
            value.get("iterations"), value.get("warmup", 10), value.get("workload"),
            value.get("placement_variant"), value.get("deadline_us"), lock["sha256"],
        )
        if contract is None:
            contract = current
        elif current != contract:
            raise ValueError(f"{path}: checksum modes differ in execution contract")
        rows = value.get("results")
        if not isinstance(rows, list):
            raise ValueError(f"{path}: results must be a list")
        parsed: dict[str, dict[str, Any]] = {}
        for row in rows:
            if row.get("system") not in SYSTEMS:
                continue
            checked = _validate_row(mode, row, path)
            if checked["system"] in parsed:
                raise ValueError(f"{path}: duplicate probe system {checked['system']}")
            parsed[checked["system"]] = checked
        record = records.setdefault(mode, {"artifacts": [], "rows": {}})
        for system, row in parsed.items():
            if system in record["rows"]:
                raise ValueError(f"{path}: duplicate probe system {system} for mode {mode}")
            record["rows"][system] = row
        record["artifacts"].append({"path": str(path.resolve()), "sha256": digest})
    if set(records) != set(MODES):
        raise ValueError("probe requires exactly inline, sampled, and off artifacts")
    for mode, record in records.items():
        if set(record["rows"]) != set(SYSTEMS):
            raise ValueError(f"mode {mode} lacks exactly QUIET and NVIDIA MPS")
    output: dict[str, Any] = {
        "schema_version": 1,
        "kind": "p9-checksum-mode-probe",
        "proposed_system": "QUIET",
        "contract": {"latency": "production-wall-arrival-to-completion", "deadline_lock_sha256": contract[-1]},
        "artifacts": {mode: data["artifacts"] for mode, data in records.items()},
        "systems": {},
        "claim_guard": "timing-mode diagnostic only; not a numeric SLO frontier",
    }
    for system in SYSTEMS:
        inline = records["inline"]["rows"][system]
        modes: dict[str, Any] = {}
        for mode in MODES:
            row = records[mode]["rows"][system]
            modes[mode] = row
            modes[mode]["delta_p99_vs_inline_us"] = row["p99_us"] - inline["p99_us"]
        output["systems"][system] = {"modes": modes}
    return output


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inline", type=Path, nargs="+", required=True)
    parser.add_argument("--sampled", type=Path, nargs="+", required=True)
    parser.add_argument("--off", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = summarize((*args.inline, *args.sampled, *args.off))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
