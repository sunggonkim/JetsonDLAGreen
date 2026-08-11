#!/usr/bin/env python3
"""Freeze a common deadline from two placement-specific pipeline locks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from freeze_p9_pipeline_deadline import build_lock
except ModuleNotFoundError:  # imported as analysis.freeze_p9_common_placement_deadline
    from analysis.freeze_p9_pipeline_deadline import build_lock


COMMON_KIND = "p9-common-placement-deadline-lock"
EXPECTED = {"fixed-1g-producer-2g-consumer", "fixed-2g-producer-1g-consumer"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.resolve().read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{path} is not newline-complete")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not an object")
    return value, hashlib.sha256(raw).hexdigest()


def build_common_lock(paths: list[Path]) -> dict[str, Any]:
    if len(paths) != 2:
        raise ValueError("exactly two placement locks are required")
    locks: list[dict[str, Any]] = []
    provenance: list[dict[str, str]] = []
    for path in paths:
        lock, digest = _load(path)
        rebuilt = build_lock(Path(lock["source_summary"]))
        if rebuilt != lock:
            raise ValueError(f"placement lock does not replay: {path}")
        if lock.get("kind") != "p9-dependent-pipeline-deadline-lock":
            raise ValueError("unexpected placement lock kind")
        locks.append(lock)
        provenance.append({"path": str(path.resolve()), "sha256": digest})
    contracts = [lock["contract"] for lock in locks]
    placements = {contract.get("placement_variant") for contract in contracts}
    if placements != EXPECTED:
        raise ValueError("both fixed placement variants are required")
    base = contracts[0]
    for contract in contracts[1:]:
        for key in ("workload", "payload_bytes", "transport", "deadline_mode", "warmup",
                    "samples_per_block", "slo_factor", "producer_quota_percent", "cpu"):
            if contract.get(key) != base.get(key):
                raise ValueError(f"placement locks differ in {key}")
        for artifact_name in ("binary", "source"):
            if locks[0]["artifacts"].get(artifact_name) != locks[1]["artifacts"].get(artifact_name):
                raise ValueError("placement locks use different binary/source artifacts")
    common_contract = dict(base)
    common_contract.update({
        "placement_variant": "common-fixed-2g+1g",
        "allowed_placements": sorted(EXPECTED),
        "placement_locks": provenance,
    })
    return {
        "schema_version": 1,
        "kind": COMMON_KIND,
        "proposed_system": "QUIET",
        "source_locks": provenance,
        "contract": common_contract,
        "allowed_placements": sorted(EXPECTED),
        "deadline_us": max(float(lock["deadline_us"]) for lock in locks),
        "slo_factor": base["slo_factor"],
        "artifacts": {
            "binary": locks[0]["artifacts"]["binary"],
            "source": locks[0]["artifacts"]["source"],
            "engines": {
                contract["placement_variant"]: lock["artifacts"]["engine"]
                for contract, lock in zip(contracts, locks)
            },
        },
        "placement_deadlines": {
            contract["placement_variant"]: lock["deadline_us"]
            for contract, lock in zip(contracts, locks)
        },
    }


def verify(path: Path) -> dict[str, Any]:
    lock, _ = _load(path)
    rebuilt = build_common_lock([Path(item["path"]) for item in lock["source_locks"]])
    if rebuilt != lock:
        raise ValueError("common placement deadline lock differs from source locks")
    return lock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", action="append", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        print(json.dumps(verify(args.verify.resolve()), indent=2))
        return 0
    if not args.lock or args.output is None:
        parser.error("--lock twice and --output are required")
    lock = build_common_lock([path.resolve() for path in args.lock])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
