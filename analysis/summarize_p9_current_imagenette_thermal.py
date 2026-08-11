#!/usr/bin/env python3
"""Aggregate six current production-wall sessions with a frozen thermal lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

try:
    from analysis.summarize_p9_current_imagenette_formal import summarize as summarize_formal
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.summarize_p9_current_imagenette_formal import summarize as summarize_formal  # type: ignore


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def summarize_thermal(
    sessions: Iterable[Path], thermal_verifications: Iterable[Path], thermal_lock: Path
) -> dict[str, Any]:
    session_paths = [Path(path).resolve() for path in sessions]
    thermal_paths = [Path(path).resolve() for path in thermal_verifications]
    if len(session_paths) != 6 or len(thermal_paths) != 6:
        raise ValueError("thermal aggregate requires six sessions and six thermal gates")
    formal = summarize_formal(session_paths)
    lock = json.loads(thermal_lock.read_bytes())
    if lock.get("kind") != "p9-current-quiet-thermal-lock" or lock.get("status") != "frozen":
        raise ValueError("thermal lock is not frozen")
    if lock.get("thermal_normalized") is not True:
        raise ValueError("thermal lock is not normalized")
    seen: set[tuple[int, str]] = set()
    thermal_rows: list[dict[str, Any]] = []
    for path in thermal_paths:
        value = json.loads(path.read_bytes())
        if (
            value.get("kind") != "p9-current-quiet-thermal-session-verification"
            or value.get("status") != "passed"
            or value.get("deadline_lock_sha256") != formal.get("deadline_lock_sha256")
            or value.get("quiet_plan_sha256") != formal.get("quiet_plan_sha256")
        ):
            raise ValueError(f"thermal gate {path} does not match formal contract")
        sequence = value.get("sequence_index")
        evidence = value.get("evidence")
        telemetry = evidence.get("telemetry") if isinstance(evidence, dict) else None
        if not isinstance(sequence, int) or not isinstance(telemetry, dict):
            raise ValueError("thermal gate sequence/telemetry provenance is missing")
        key = (sequence, telemetry.get("sha256", ""))
        if key in seen:
            raise ValueError("thermal gate evidence is duplicated")
        seen.add(key)
        thermal_rows.append({
            "path": str(path),
            "sha256": sha256(path),
            "sequence_index": sequence,
            "telemetry": telemetry,
            "metrics": value["metrics"],
            "thermal_condition": value["thermal_condition"],
        })
    if len(seen) != 6 or {row["sequence_index"] for row in thermal_rows} != {0, 1, 2}:
        raise ValueError("thermal gate sequence set differs")
    formal["kind"] = "p9-current-imagenette-thermal-formal-production-wall-aggregate"
    formal["thermal_normalized"] = True
    formal["thermal_claim_allowed"] = True
    formal["ranking_allowed"] = False
    formal["thermal_lock"] = {
        "path": str(thermal_lock.resolve()),
        "sha256": sha256(thermal_lock),
    }
    formal["thermal_sessions"] = thermal_rows
    formal["claim_guard"] = (
        "Thermal headline is limited to the frozen soc012/tj/VDD_GPU envelope; "
        "same-SLO ranking remains a separate frontier gate."
    )
    return formal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--thermal", action="append", type=Path, required=True)
    parser.add_argument("--thermal-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = summarize_thermal(args.input, args.thermal, args.thermal_lock)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
