#!/usr/bin/env python3
"""Aggregate independent common-deadline SOTA smokes without overstating balance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import beta


SYSTEMS = ("NVIDIA MIG", "NVIDIA MPS", "Orion", "XSched", "gpulet", "QUIET")
HEADLINE_SYSTEMS = ("NVIDIA MIG", "NVIDIA MPS", "Orion", "XSched", "Pantheon", "QUIET")
NUMERIC_INPUTS = ("frontier", "orion_numeric", "xsched_numeric", "gpulet_numeric")
COMMON_WORKLOAD_KEYS = (
    "schema_version", "workload_id", "topology", "placement", "input_tensor",
    "payload_bytes", "arrival_trace_path", "arrival_trace_sha256",
    "dataset_manifest_path", "dataset_manifest_sha256", "contract_path",
    "contract_sha256",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cp95_upper(misses: int, requests: int) -> float:
    if requests <= 0 or not 0 <= misses <= requests:
        raise ValueError("invalid miss count")
    return 1.0 if misses == requests else float(beta.ppf(0.95, misses + 1, requests - misses))


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def resolve_input(summary: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    direct = (Path.cwd() / path).resolve()
    return direct if direct.exists() else (summary.parent / path).resolve()


def validate_common_workload(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or any(key not in value for key in COMMON_WORKLOAD_KEYS):
        raise ValueError("common workload contract is malformed")
    if value.get("schema_version") != 1:
        raise ValueError("common workload contract schema differs")
    for path_key, digest_key in (
        ("arrival_trace_path", "arrival_trace_sha256"),
        ("dataset_manifest_path", "dataset_manifest_sha256"),
        ("contract_path", "contract_sha256"),
    ):
        path = Path(value[path_key]).resolve()
        digest = value[digest_key]
        if (
            not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not path.is_file() or sha256(path) != digest
        ):
            raise ValueError(f"common workload evidence SHA differs: {path_key}")
    return dict(value)


def summarize(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two independent comparison smokes are required")
    rows: dict[str, list[dict[str, float | int]]] = {name: [] for name in SYSTEMS}
    evidence: list[dict[str, Any]] = []
    used_inputs: dict[str, set[str]] = {name: set() for name in NUMERIC_INPUTS}
    contract: dict[str, Any] | None = None
    headline_contract: dict[str, Any] | None = None
    headline_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in HEADLINE_SYSTEMS}
    common_workload: dict[str, Any] | None = None
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        current = {
            "kind": summary.get("kind"),
            "workload": summary.get("workload"),
            "scope": summary.get("scope"),
            "offered_background_rps": summary.get("offered_background_rps"),
            "common_deadline_us": summary.get("common_deadline_us"),
            "proposed_system": summary.get("proposed_system"),
        }
        if (
            current["kind"] != "p9-dependent-payload-six-system-smoke"
            or current["proposed_system"] != "QUIET"
            or finite(current["common_deadline_us"], "deadline") <= 0.0
        ):
            raise ValueError("comparison contract differs")
        current_common = summary.get("common_workload")
        if current_common is None:
            if common_workload is not None:
                raise ValueError("repeat common workload evidence is incomplete")
        else:
            checked_common = validate_common_workload(current_common)
            if common_workload is None:
                common_workload = checked_common
            elif checked_common != common_workload:
                raise ValueError("repeat common workload contract differs")
        if contract is None:
            contract = current
        elif current != contract:
            raise ValueError("repeat contracts differ")
        inputs = summary.get("inputs")
        if not isinstance(inputs, dict):
            raise ValueError("comparison inputs are missing")
        checked: dict[str, dict[str, str]] = {}
        for name in NUMERIC_INPUTS:
            item = inputs.get(name)
            if not isinstance(item, dict):
                raise ValueError(f"{name} evidence is missing")
            source = resolve_input(path, item["path"])
            digest = sha256(source)
            if digest != item["sha256"] or digest in used_inputs[name]:
                raise ValueError(f"{name} evidence differs or is reused")
            used_inputs[name].add(digest)
            checked[name] = {"path": str(source), "sha256": digest}
        systems = summary.get("systems")
        if not isinstance(systems, list) or tuple(row.get("system") for row in systems) != SYSTEMS:
            raise ValueError("public system rows differ")
        for row in systems:
            requests = row.get("requests")
            misses = row.get("misses")
            if (
                isinstance(requests, bool) or not isinstance(requests, int) or requests <= 0
                or isinstance(misses, bool) or not isinstance(misses, int)
                or not 0 <= misses <= requests
            ):
                raise ValueError("request totals differ")
            deadline = finite(row.get("deadline_us"), "row deadline")
            if not math.isclose(deadline, float(contract["common_deadline_us"]), abs_tol=1e-9):
                raise ValueError("row deadline differs")
            rows[row["system"]].append(
                {
                    "requests": requests,
                    "misses": misses,
                    "p99_us": finite(row.get("deadline_p99_us"), "p99"),
                    "goodput_rps": finite(row.get("background_goodput_rps"), "goodput"),
                }
            )
        advertised_headline = summary.get("headline_systems")
        if advertised_headline is not None:
            if (
                not isinstance(advertised_headline, list)
                or tuple(row.get("system") for row in advertised_headline)
                != HEADLINE_SYSTEMS
            ):
                raise ValueError("headline system rows differ")
            current_headline_contract = summary.get("headline_contract")
            if not isinstance(current_headline_contract, dict):
                raise ValueError("headline contract is missing")
            if headline_contract is None:
                headline_contract = current_headline_contract
            elif current_headline_contract != headline_contract:
                raise ValueError("headline contracts differ")
            for row in advertised_headline:
                name = row["system"]
                allowed = (
                    row.get("numeric_comparison_allowed") is True
                    and common_workload is not None
                )
                if not allowed:
                    headline_rows[name].append(
                        {
                            "numeric_comparison_allowed": False,
                            "comparison_status": row.get("comparison_status", "functional-only"),
                        }
                    )
                    continue
                requests = row.get("requests")
                misses = row.get("misses")
                if (
                    isinstance(requests, bool) or not isinstance(requests, int) or requests <= 0
                    or isinstance(misses, bool) or not isinstance(misses, int)
                    or not 0 <= misses <= requests
                ):
                    raise ValueError("headline request totals differ")
                headline_rows[name].append(
                    {
                        "numeric_comparison_allowed": True,
                        "requests": requests,
                        "misses": misses,
                        "p99_us": finite(row.get("deadline_p99_us"), "headline p99"),
                        "goodput_rps": finite(row.get("background_goodput_rps"), "headline goodput"),
                    }
                )
        evidence.append({"path": str(path.resolve()), "sha256": sha256(path), "inputs": checked})
    assert contract is not None
    systems_out: dict[str, Any] = {}
    for name in SYSTEMS:
        samples = rows[name]
        requests = sum(int(item["requests"]) for item in samples)
        misses = sum(int(item["misses"]) for item in samples)
        p99s = [float(item["p99_us"]) for item in samples]
        goodputs = [float(item["goodput_rps"]) for item in samples]
        systems_out[name] = {
            "runs": len(samples),
            "requests": requests,
            "misses": misses,
            "observed_dmr": misses / requests,
            "dmr_cp95_upper": cp95_upper(misses, requests),
            "per_run_p99_us": p99s,
            "p99_us_range": [min(p99s), max(p99s)],
            "background_goodput_rps_mean": statistics.fmean(goodputs),
            "background_goodput_rps_range": [min(goodputs), max(goodputs)],
        }
    headline_out: dict[str, Any] = {}
    for name in HEADLINE_SYSTEMS:
        samples = headline_rows[name]
        if not samples:
            continue
        if any(item.get("numeric_comparison_allowed") is not True for item in samples):
            headline_out[name] = {
                "runs": len(samples),
                "numeric_comparison_allowed": False,
                "comparison_status": samples[0].get("comparison_status", "functional-only"),
            }
            continue
        requests = sum(int(item["requests"]) for item in samples)
        misses = sum(int(item["misses"]) for item in samples)
        p99s = [float(item["p99_us"]) for item in samples]
        goodputs = [float(item["goodput_rps"]) for item in samples]
        headline_out[name] = {
            "runs": len(samples),
            "numeric_comparison_allowed": True,
            "requests": requests,
            "misses": misses,
            "observed_dmr": misses / requests,
            "dmr_cp95_upper": cp95_upper(misses, requests),
            "per_run_p99_us": p99s,
            "p99_us_range": [min(p99s), max(p99s)],
            "background_goodput_rps_mean": statistics.fmean(goodputs),
            "background_goodput_rps_range": [min(goodputs), max(goodputs)],
        }
    return {
        "schema_version": 1,
        "kind": "p9-common-sota-exploratory-repeats",
        "scope": "independent-hardware-repeats-not-counterbalanced-not-formal",
        "proposed_system": "QUIET",
        "contract": contract,
        "common_workload": common_workload,
        "inputs": evidence,
        "systems": systems_out,
        "headline_systems": headline_out,
        "headline_contract": headline_contract,
        "claim_guard": (
            "Per-run ranges only; execute a frozen Williams campaign before inferential claims."
            if common_workload is not None
            else "Common workload contract is missing; legacy repeats remain non-promoting raw evidence."
        ),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize([path.resolve() for path in args.input])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
