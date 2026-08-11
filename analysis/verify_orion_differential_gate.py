#!/usr/bin/env python3
"""Verify canonical Orion upstream-vs-Thor scheduler decisions.

The two traces are deliberately canonical adapter outputs.  Raw CUDA timing
and pointer values are not comparable across processes; admission, ordering,
and operation identity are.  A passed gate is required before an Orion row can
be promoted from functional evidence to a numeric comparator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable


UPSTREAM_COMMIT = "20f9469764fb96d94ce23a8e70615196e9ce4ba1"
TRACE_KEYS = {
    "schema_version", "case_id", "arrival_sequence", "decision_sequence",
    "client_id", "priority", "api", "profile_position", "sm_used",
    "duration_us", "admitted", "reordered", "admission_reason",
}
SIGNATURE_KEYS = (
    "case_id", "arrival_sequence", "client_id", "priority", "api",
    "profile_position", "sm_used", "duration_us", "admitted", "reordered",
    "admission_reason",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_pinned_checkout(source_path: Path) -> dict[str, str]:
    """Require the reference source to be a tracked file at the pinned HEAD."""
    source_path = source_path.resolve()
    try:
        root = Path(subprocess.check_output(
            ["git", "-C", str(source_path.parent), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()).resolve()
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        relative = source_path.relative_to(root)
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", str(relative)],
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise ValueError("reference source is not inside a Git checkout") from error
    if head != UPSTREAM_COMMIT:
        raise ValueError("reference source checkout HEAD is not the pinned Orion commit")
    if not tracked:
        raise ValueError("reference source is not a tracked file in the pinned checkout")
    return {
        "reference_git_root": str(root),
        "reference_git_head": head,
        "reference_source_relative_path": str(relative),
    }


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _integer(value: Any, label: str, *, nonnegative: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if nonnegative and value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _hex(value: Any, label: str, length: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase hexadecimal")
    return value


def _load_common_workload_contract(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    raw = resolved.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("common workload contract is not newline-complete")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("common workload contract is invalid JSON") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("common workload contract schema differs")
    for key in (
        "workload_id", "topology", "placement", "input_tensor", "payload_bytes",
        "arrival_trace_path", "arrival_trace_sha256",
        "dataset_manifest_path", "dataset_manifest_sha256",
    ):
        if key not in value:
            raise ValueError(f"common workload contract lacks {key}")
    for path_key, digest_key in (
        ("arrival_trace_path", "arrival_trace_sha256"),
        ("dataset_manifest_path", "dataset_manifest_sha256"),
    ):
        evidence = Path(value[path_key]).resolve()
        digest = _hex(value[digest_key], digest_key, 64)
        if not evidence.is_file() or sha256(evidence) != digest:
            raise ValueError(f"common workload evidence SHA mismatches: {path_key}")
        value[path_key] = str(evidence)
    value["contract_path"] = str(resolved)
    value["contract_sha256"] = hashlib.sha256(raw).hexdigest()
    return value


def _load_trace_provenance(
    path: Path,
    *,
    reference_trace: Path,
    reference_source: Path,
    upstream_runtime_binary: Path,
    common_workload: dict[str, Any],
) -> dict[str, Any]:
    """Bind the reference trace to one pinned-upstream execution record."""
    raw = path.resolve().read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("Orion trace provenance is not newline-complete")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("Orion trace provenance is invalid JSON") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Orion trace provenance schema differs")
    if value.get("kind") != "orion-upstream-trace-provenance":
        raise ValueError("Orion trace provenance kind differs")
    if value.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("Orion trace provenance commit differs")
    if value.get("generator") != "pinned-upstream-orion-runtime":
        raise ValueError("Orion trace provenance generator is not upstream")
    trace_value = value.get("reference_trace_path")
    source_value = value.get("reference_source_path")
    binary_value = value.get("upstream_runtime_binary_path")
    contract_value = value.get("common_workload_path")
    if not all(
        isinstance(item, str) and item
        for item in (trace_value, source_value, binary_value, contract_value)
    ):
        raise ValueError("Orion trace provenance paths are missing")
    trace_path = Path(trace_value).resolve()
    source_path = Path(source_value).resolve()
    binary_path = Path(binary_value).resolve()
    if (
        trace_path != reference_trace.resolve()
        or source_path != reference_source.resolve()
        or binary_path != upstream_runtime_binary.resolve()
    ):
        raise ValueError("Orion trace provenance paths differ")
    trace_sha = _hex(value.get("reference_trace_sha256"), "reference trace SHA", 64)
    source_sha = _hex(value.get("reference_source_sha256"), "reference source SHA", 64)
    binary_sha = _hex(
        value.get("upstream_runtime_binary_sha256"),
        "upstream runtime binary SHA",
        64,
    )
    if trace_sha != sha256(reference_trace.resolve()):
        raise ValueError("Orion trace provenance trace SHA differs")
    if source_sha != sha256(reference_source.resolve()):
        raise ValueError("Orion trace provenance source SHA differs")
    if binary_sha != sha256(upstream_runtime_binary.resolve()):
        raise ValueError("Orion trace provenance runtime binary SHA differs")
    contract_path = Path(contract_value).resolve()
    if contract_path != Path(common_workload["contract_path"]).resolve():
        raise ValueError("Orion trace provenance workload path differs")
    contract_sha = _hex(value.get("common_workload_sha256"), "common workload SHA", 64)
    if contract_sha != common_workload["contract_sha256"]:
        raise ValueError("Orion trace provenance workload SHA differs")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "generator": value["generator"],
        "reference_trace_sha256": trace_sha,
        "reference_source_sha256": source_sha,
        "upstream_runtime_binary_path": str(binary_path),
        "upstream_runtime_binary_sha256": binary_sha,
        "common_workload_sha256": contract_sha,
    }


def _read_trace(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{path} is not newline-complete")
    rows: list[dict[str, Any]] = []
    arrivals: set[int] = set()
    decisions: set[int] = set()
    for line_number, line in enumerate(raw.splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from error
        if not isinstance(value, dict) or set(value) != TRACE_KEYS:
            raise ValueError(f"{path}:{line_number} canonical schema differs")
        if value.get("schema_version") != 1:
            raise ValueError(f"{path}:{line_number} schema version differs")
        case_id = value.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{path}:{line_number} case_id is invalid")
        arrival = _integer(value.get("arrival_sequence"), "arrival_sequence")
        decision = _integer(value.get("decision_sequence"), "decision_sequence")
        if arrival in arrivals or decision in decisions:
            raise ValueError(f"{path}:{line_number} duplicate trace sequence")
        arrivals.add(arrival)
        decisions.add(decision)
        if not isinstance(value.get("client_id"), int) or isinstance(
            value.get("client_id"), bool
        ) or value["client_id"] < 0:
            raise ValueError(f"{path}:{line_number} client_id is invalid")
        if value.get("priority") not in {"high", "best-effort"}:
            raise ValueError(f"{path}:{line_number} priority is invalid")
        if not isinstance(value.get("api"), str) or not value["api"]:
            raise ValueError(f"{path}:{line_number} API is invalid")
        _integer(value.get("profile_position"), "profile_position")
        _integer(value.get("sm_used"), "sm_used")
        _finite(value.get("duration_us"), "duration_us")
        if not isinstance(value.get("admitted"), bool) or not isinstance(
            value.get("reordered"), bool
        ):
            raise ValueError(f"{path}:{line_number} decision flags are invalid")
        if not isinstance(value.get("admission_reason"), str) or not value[
            "admission_reason"
        ]:
            raise ValueError(f"{path}:{line_number} admission reason is invalid")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} is empty")
    if sorted(arrivals) != list(range(len(rows))) or sorted(decisions) != list(
        range(len(rows))
    ):
        raise ValueError(f"{path} sequences are not dense and zero-based")
    rows.sort(key=lambda row: row["decision_sequence"])
    return rows


def _signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[key] for key in SIGNATURE_KEYS)


def verify(
    reference_path: Path,
    port_path: Path,
    *,
    upstream_commit: str = UPSTREAM_COMMIT,
    reference_source_path: Path,
    reference_source_sha256: str | None = None,
    upstream_runtime_binary_path: Path | None = None,
    upstream_runtime_binary_sha256: str | None = None,
    expected_cases: int | None = None,
    require_pinned_checkout: bool = True,
    common_workload_contract: Path | None = None,
    require_common_workload: bool = False,
    reference_trace_provenance: Path | None = None,
) -> dict[str, Any]:
    _hex(upstream_commit, "upstream_commit", 40)
    if upstream_commit != UPSTREAM_COMMIT:
        raise ValueError("upstream commit is not the pinned Orion artifact")
    if require_common_workload and common_workload_contract is None:
        raise ValueError("Orion fidelity gate requires a common workload contract")
    common_workload = (
        _load_common_workload_contract(common_workload_contract)
        if common_workload_contract is not None else None
    )
    if require_pinned_checkout and common_workload is not None and reference_trace_provenance is None:
        raise ValueError(
            "Orion numeric fidelity gate requires pinned upstream trace provenance"
        )
    source_path = reference_source_path.resolve()
    if not source_path.is_file():
        raise ValueError("reference_source_path must name a regular file")
    checkout = (
        _verify_pinned_checkout(source_path)
        if require_pinned_checkout
        else {
            "reference_git_root": None,
            "reference_git_head": None,
            "reference_source_relative_path": None,
        }
    )
    computed_source_sha256 = sha256(source_path)
    if reference_source_sha256 is not None:
        _hex(reference_source_sha256, "reference_source_sha256", 64)
        if reference_source_sha256 != computed_source_sha256:
            raise ValueError("reference source SHA256 does not match source bytes")
    runtime_binary = None
    computed_runtime_binary_sha256 = None
    if upstream_runtime_binary_path is not None:
        runtime_binary = upstream_runtime_binary_path.resolve()
        if not runtime_binary.is_file():
            raise ValueError("upstream_runtime_binary_path must name a regular file")
        computed_runtime_binary_sha256 = sha256(runtime_binary)
        if upstream_runtime_binary_sha256 is not None:
            _hex(upstream_runtime_binary_sha256, "upstream_runtime_binary_sha256", 64)
            if upstream_runtime_binary_sha256 != computed_runtime_binary_sha256:
                raise ValueError(
                    "upstream runtime binary SHA256 does not match binary bytes"
                )
    if require_pinned_checkout and common_workload is not None and runtime_binary is None:
        raise ValueError(
            "Orion numeric fidelity gate requires the pinned upstream runtime binary"
        )
    reference = _read_trace(reference_path.resolve())
    port = _read_trace(port_path.resolve())
    if expected_cases is not None and len(reference) != expected_cases:
        raise ValueError("reference trace case count differs")
    if len(reference) != len(port):
        raise ValueError("reference and Thor trace lengths differ")
    mismatches = [
        index for index, (left, right) in enumerate(zip(reference, port, strict=True))
        if _signature(left) != _signature(right)
    ]
    if mismatches:
        raise ValueError(
            "Orion differential decision mismatch at case(s): "
            + ",".join(str(index) for index in mismatches[:8])
        )
    provenance = None
    if reference_trace_provenance is not None:
        if common_workload is None or runtime_binary is None:
            raise ValueError("Orion trace provenance requires a common workload contract")
        provenance = _load_trace_provenance(
            reference_trace_provenance,
            reference_trace=reference_path,
            reference_source=source_path,
            upstream_runtime_binary=runtime_binary,
            common_workload=common_workload,
        )
    return {
        "schema_version": 1,
        "kind": "orion-differential-fidelity-gate",
        "system": "Orion",
        "status": "passed",
        "reference": "pinned-upstream-scheduler",
        "upstream_commit": UPSTREAM_COMMIT,
        "decision_cases": len(reference),
        "mismatch_cases": 0,
        "reference_source_path": str(source_path),
        "reference_source_sha256": computed_source_sha256,
        "reference_source_verified": True,
        "upstream_runtime_binary_path": str(runtime_binary) if runtime_binary else None,
        "upstream_runtime_binary_sha256": computed_runtime_binary_sha256,
        "upstream_runtime_binary_verified": runtime_binary is not None,
        "reference_checkout_verified": require_pinned_checkout,
        **checkout,
        "reference_trace_path": str(reference_path.resolve()),
        "reference_trace_sha256": sha256(reference_path.resolve()),
        "port_trace_path": str(port_path.resolve()),
        "port_trace_sha256": sha256(port_path.resolve()),
        "common_workload": common_workload,
        "reference_trace_provenance": provenance,
        "numeric_comparison_allowed": (
            require_pinned_checkout
            and common_workload is not None
            and provenance is not None
            and runtime_binary is not None
        ),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-trace", type=Path, required=True)
    parser.add_argument("--port-trace", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--reference-source-sha256")
    parser.add_argument("--upstream-runtime-binary", type=Path)
    parser.add_argument("--upstream-runtime-binary-sha256")
    parser.add_argument("--expected-cases", type=int)
    parser.add_argument("--common-workload-contract", type=Path)
    parser.add_argument("--require-common-workload", action="store_true")
    parser.add_argument("--reference-trace-provenance", type=Path)
    parser.add_argument(
        "--allow-unpinned-source", action="store_true",
        help="emit non-promoting local fixture evidence; numeric comparison remains disabled",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = verify(
        args.reference_trace,
        args.port_trace,
        reference_source_path=args.reference_source,
        reference_source_sha256=args.reference_source_sha256,
        upstream_runtime_binary_path=args.upstream_runtime_binary,
        upstream_runtime_binary_sha256=args.upstream_runtime_binary_sha256,
        expected_cases=args.expected_cases,
        require_pinned_checkout=not args.allow_unpinned_source,
        common_workload_contract=args.common_workload_contract,
        require_common_workload=args.require_common_workload,
        reference_trace_provenance=args.reference_trace_provenance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
