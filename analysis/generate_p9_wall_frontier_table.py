#!/usr/bin/env python3
"""Render the replayed production-wall frontier used by the P9 paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "docs" / "p9-comparator-manifest.json"
PRODUCTION_WALL_DEFINITION_V2 = (
    "arrival-to-consumer-completion-excludes-correctness-validation"
)
CORRECTNESS_PLACEMENT_V2 = "post-completion"


def _load_manifest() -> dict[str, Any]:
    raw = MANIFEST_PATH.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{MANIFEST_PATH} is not newline-complete")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("comparator manifest is not an object")
    policy = value.get("paper_table_policy")
    if not isinstance(policy, dict) or policy.get("proposed_system") != "QUIET":
        raise ValueError("comparator manifest lacks the QUIET paper-table policy")
    order = policy.get("numeric_frontier_order")
    if not isinstance(order, list) or not order or any(not isinstance(x, str) for x in order):
        raise ValueError("comparator manifest numeric frontier order is invalid")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _validate_application_accuracy_gate(value: dict[str, Any], path: Path) -> None:
    record = value.get("application_accuracy_gate")
    if not isinstance(record, dict):
        raise ValueError(f"{path}: formal frontier lacks application-accuracy gate")
    gate_path = record.get("path")
    expected_sha = record.get("sha256")
    if not isinstance(gate_path, str) or not gate_path:
        raise ValueError(f"{path}: application-accuracy gate path is missing")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError(f"{path}: application-accuracy gate SHA is invalid")
    try:
        int(expected_sha, 16)
    except ValueError as error:
        raise ValueError(f"{path}: application-accuracy gate SHA is invalid") from error
    gate_file = Path(gate_path).resolve()
    if not gate_file.is_file() or hashlib.sha256(gate_file.read_bytes()).hexdigest() != expected_sha:
        raise ValueError(f"{path}: application-accuracy gate changed or is missing")
    gate = json.loads(gate_file.read_bytes())
    if (
        not isinstance(gate, dict)
        or gate.get("kind") != "p9-application-accuracy-gate"
        or gate.get("status") != "passed"
        or gate.get("numeric_comparison_allowed") is not True
        or gate.get("application_input_binding_required") is not True
        or gate.get("application_input_binding_contract") != "passed"
    ):
        raise ValueError(f"{path}: application-accuracy gate is not passed")
    for prefix in ("reference", "candidate"):
        record = gate.get(f"{prefix}_pipeline_csv")
        if not isinstance(record, dict):
            raise ValueError(f"{path}: {prefix} input-bound pipeline CSV is missing")
        csv_path, csv_sha = record.get("path"), record.get("sha256")
        if not isinstance(csv_path, str) or not csv_path or not isinstance(csv_sha, str) or len(csv_sha) != 64:
            raise ValueError(f"{path}: {prefix} input-bound pipeline CSV digest is invalid")
        try:
            observed = hashlib.sha256(Path(csv_path).resolve().read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError(f"{path}: {prefix} input-bound pipeline CSV is missing") from error
        if observed != csv_sha:
            raise ValueError(f"{path}: {prefix} input-bound pipeline CSV changed")


def _validate_point(point: dict[str, Any], system: str, path: Path) -> None:
    if point.get("system") != system:
        raise ValueError(f"{path}: frontier point system does not match {system}")
    requests = _positive_int(point.get("requests"), f"{system} requests")
    misses = point.get("deadline_misses")
    if isinstance(misses, bool) or not isinstance(misses, int) or not 0 <= misses <= requests:
        raise ValueError(f"{path}: {system} deadline misses are invalid")
    for key in ("offered_rps", "p99_us", "background_goodput_rps", "dmr"):
        _finite(point.get(key), f"{system} {key}")
    expected_dmr = misses / requests
    if not math.isclose(float(point["dmr"]), expected_dmr, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{path}: {system} DMR does not match requests/misses")
    lock_sha = point.get("deadline_lock_sha256")
    if not isinstance(lock_sha, str) or len(lock_sha) != 64:
        raise ValueError(f"{path}: {system} lacks exact deadline-lock SHA")
    try:
        int(lock_sha, 16)
    except ValueError as exc:
        raise ValueError(f"{path}: {system} deadline-lock SHA is invalid") from exc
    evidence = point.get("correctness_evidence")
    if not isinstance(evidence, dict) or evidence.get("mode") != "inline":
        raise ValueError(f"{path}: {system} lacks inline correctness evidence")
    if evidence.get("checksum_failures") != 0:
        raise ValueError(f"{path}: {system} has checksum failures")
    _positive_int(evidence.get("unique_payload_checksums"), f"{system} payload checksums")
    _positive_int(evidence.get("unique_policy_output_checksums"), f"{system} output checksums")
    source = evidence.get("source")
    if source == "row":
        return
    if source != "sota_verification":
        raise ValueError(f"{path}: {system} has unknown correctness evidence source")
    verification_path = evidence.get("path")
    digest = evidence.get("sha256")
    if not isinstance(verification_path, str) or not verification_path:
        raise ValueError(f"{path}: {system} verification path is missing")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{path}: {system} verification digest is invalid")
    verification_file = Path(verification_path).resolve()
    if not verification_file.is_file():
        raise ValueError(f"{path}: {system} verification file is missing")
    actual = hashlib.sha256(verification_file.read_bytes()).hexdigest()
    if actual != digest:
        raise ValueError(f"{path}: {system} verification file changed")


def load(path: Path) -> dict:
    raw = path.resolve().read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{path} is not newline-complete")
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("kind") != "p9-common-production-wall-load-frontier":
        raise ValueError("unexpected production-wall frontier artifact")
    if value.get("proposed_system") != "QUIET":
        raise ValueError("the only proposed system name must be QUIET")
    # A frontier is a paper-facing numeric artifact.  Do not let a summary
    # produced by the pre-v2 binary re-enter the table after a rebuild.
    if value.get("production_wall_definition") != PRODUCTION_WALL_DEFINITION_V2:
        raise ValueError("frontier uses a stale production-wall contract")
    if value.get("correctness_validation_placement") != CORRECTNESS_PLACEMENT_V2:
        raise ValueError("frontier uses a stale correctness-validation placement")
    if value.get("formal") is True:
        _validate_application_accuracy_gate(value, path)
    frontier = value.get("frontier")
    if not isinstance(frontier, dict):
        raise ValueError("frontier is missing")
    manifest = _load_manifest()
    numeric_order = manifest["paper_table_policy"]["numeric_frontier_order"]
    if value.get("numeric_frontier_systems") != numeric_order:
        raise ValueError("frontier numeric system order differs from comparator manifest")
    frontier_systems = set(frontier)
    expected_exploratory = sorted(frontier_systems - set(numeric_order))
    if value.get("exploratory_systems") != expected_exploratory:
        raise ValueError("frontier exploratory systems are not explicitly separated")
    if value.get("ranking_allowed") is not False:
        raise ValueError("exploratory frontier cannot authorize ranking")
    missing = [name for name in numeric_order if name not in frontier]
    if missing:
        raise ValueError(f"frontier lacks manifest numeric rows: {', '.join(missing)}")
    manifest_rows = manifest.get("rows", {})
    for name in numeric_order:
        row_contract = manifest_rows.get(name)
        if not isinstance(row_contract, dict) or row_contract.get("numeric_comparison_allowed") is not True:
            raise ValueError(f"manifest does not authorize numeric row: {name}")
    lock_shas: set[str] = set()
    for system, item in frontier.items():
        if not isinstance(item, dict) or not isinstance(item.get("points"), list):
            raise ValueError(f"{path}: {system} frontier points are missing")
        for point in item["points"]:
            if not isinstance(point, dict):
                raise ValueError(f"{path}: {system} frontier point is not an object")
            _validate_point(point, system, path)
            lock_shas.add(point["deadline_lock_sha256"])
    if len(lock_shas) != 1:
        raise ValueError("frontier rows do not share one exact deadline lock")
    return value


def render(value: dict) -> str:
    systems = value["frontier"]
    numeric_order = _load_manifest()["paper_table_policy"]["numeric_frontier_order"]
    scope = "formal" if value.get("formal") is True else "exploratory"
    lines = [
        "% Generated by analysis/generate_p9_wall_frontier_table.py; do not edit.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{scope.capitalize()} v2 production-wall common-workload frontier under the same $773.730$-$\\mu$s deadline. Correctness validation occurs after the measured consumer-completion boundary; only rows authorized by the frozen comparator manifest are ranked.}}",
        "\\label{tab:p9-wall-frontier}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "System & p99 at 250 ($\\mu$s) & Sweep misses/300 & Mean BE req/s & Max qualified offer \\\\",
        "\\midrule",
    ]
    for name in numeric_order:
        item = systems[name]
        point = next(p for p in item["points"] if abs(float(p["offered_rps"]) - 250.0) < 1e-6)
        max_offer = item.get("max_slo_qualified_offered_rps")
        offer = f"{float(max_offer):.0f}" if max_offer is not None else "--"
        total_misses = sum(int(p["deadline_misses"]) for p in item["points"])
        total_requests = sum(int(p["requests"]) for p in item["points"])
        mean_goodput = sum(float(p["background_goodput_rps"]) for p in item["points"]) / len(item["points"])
        label = "\\textbf{QUIET}" if name == "QUIET" else name
        lines.append(
            f"{label} & {float(point['p99_us']):.2f} & "
            f"{total_misses}/{total_requests} & "
            f"{mean_goodput:.2f} & {offer} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(load(args.artifact)), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
