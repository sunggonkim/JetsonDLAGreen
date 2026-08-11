#!/usr/bin/env python3
"""Verify that a frozen deadline lock binds the runtime artifacts in use.

Deadline calibration is only meaningful for the exact executable, source,
producer engine, and consumer engine used by a later comparator run.  This
module is shared by the Python launcher and the XSched shell wrapper so a
stale installed engine cannot silently enter a current workload row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_artifact(artifacts: dict[str, Any], name: str) -> tuple[Path, str]:
    record = artifacts.get(name)
    if not isinstance(record, dict):
        raise ValueError(f"deadline lock lacks artifacts.{name}")
    path_value, expected = record.get("path"), record.get("sha256")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"deadline lock artifacts.{name}.path is invalid")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"deadline lock artifacts.{name}.sha256 is invalid")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise ValueError(f"deadline lock artifact is missing: {path}")
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"deadline lock artifact SHA differs: {name}")
    return path, observed


def verify_runtime_binding(
    lock_path: Path,
    repo: Path,
    producer_engine: Path,
    consumer_engine: Path | None = None,
) -> dict[str, Any]:
    """Require the frozen lock to name and hash every launched artifact."""
    lock_path = lock_path.resolve()
    repo = repo.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(lock, dict):
        raise ValueError("deadline lock root is not an object")
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("deadline lock lacks artifact provenance")

    expected_paths = {
        "binary": (repo / "build-r39/jdg-mig-trt-pipeline").resolve(),
        "source": (repo / "benchmarks/mig_trt_pipeline.cpp").resolve(),
        "engine": producer_engine.resolve(),
    }
    verified: dict[str, dict[str, str]] = {}
    for name, expected_path in expected_paths.items():
        actual_path, digest = _bound_artifact(artifacts, name)
        if actual_path != expected_path:
            raise ValueError(
                f"deadline lock artifacts.{name}.path differs from launched runtime: "
                f"{actual_path} != {expected_path}"
            )
        verified[name] = {"path": str(actual_path), "sha256": digest}

    consumer_record = artifacts.get("consumer_engine")
    if consumer_engine is not None:
        actual_path, digest = _bound_artifact(artifacts, "consumer_engine")
        expected_path = consumer_engine.resolve()
        if actual_path != expected_path:
            raise ValueError(
                "deadline lock consumer engine differs from launched runtime: "
                f"{actual_path} != {expected_path}"
            )
        verified["consumer_engine"] = {"path": str(actual_path), "sha256": digest}
    elif consumer_record is not None:
        raise ValueError(
            "deadline lock binds a consumer engine but the launched runtime omits it"
        )

    return {
        "lock_path": str(lock_path),
        "lock_sha256": sha256(lock_path),
        "artifacts": verified,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--producer-engine", type=Path, required=True)
    parser.add_argument("--consumer-engine", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_runtime_binding(
        args.lock, args.repo, args.producer_engine, args.consumer_engine,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
