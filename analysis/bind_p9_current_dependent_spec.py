#!/usr/bin/env python3
"""Bind a BOER or ParvaGPU dependent template to a frozen pipeline lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def replace_option(command: list[str], option: str, value: str) -> None:
    try:
        index = command.index(option)
    except ValueError as error:
        raise ValueError(f"template evaluator omits {option}") from error
    if index + 1 >= len(command):
        raise ValueError(f"template evaluator has no value for {option}")
    command[index + 1] = value


def bind(
    template: dict[str, Any], lock: dict[str, Any], lock_path: Path,
    repo: Path, result_root: Path | None,
) -> dict[str, Any]:
    if (
        lock.get("kind") != "p9-dependent-pipeline-deadline-lock"
        or lock.get("contract", {}).get("workload") != "whisper-projection"
    ):
        raise ValueError("deadline lock is not the Whisper dependent contract")
    deadline = finite_positive(lock.get("deadline_us"), "deadline")
    system = template.get("system")
    contract = template.get("contract")
    if not isinstance(contract, dict) or contract.get("scenario") != "dependent-large-payload":
        raise ValueError("template is not the large dependent workload")
    relative_lock = str(lock_path.resolve().relative_to(repo.resolve()))
    contract["deadline_lock_path"] = relative_lock
    contract["deadline_lock_sha256"] = sha256(lock_path)
    if system == "BOER":
        contract["deadline_us"] = deadline
        command = template.get("evaluator_command")
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise ValueError("BOER evaluator command is invalid")
        replace_option(command, "--deadline-us", str(deadline))
        if result_root is None:
            raise ValueError("BOER binding requires a result root")
        replace_option(command, "--result-root", str(result_root.resolve()))
    elif system == "ParvaGPU":
        contract["pipeline_deadline_us"] = deadline
        services = template.get("services")
        if not isinstance(services, list):
            raise ValueError("ParvaGPU services are invalid")
        producer = next(
            (item for item in services if item.get("model") == "whisper-producer"), None
        )
        if producer is None:
            raise ValueError("ParvaGPU template omits the Whisper producer")
        producer["slo_ms"] = deadline / 1000.0
    else:
        raise ValueError("unsupported published system template")
    return template


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--deadline-lock", type=Path, required=True)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    result = bind(
        json.loads(args.template.read_text(encoding="utf-8")),
        json.loads(args.deadline_lock.read_text(encoding="utf-8")),
        args.deadline_lock, repo, args.result_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
