#!/usr/bin/env python3
"""Build the honest public-system P9 smoke table from verified artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


DEADLINE_LOCK_SHA256 = "4b383e300d756f7da0987d0077bec2416c01588ca1a019b04c0e3e05b2b5ab48"
ORION_MAX_BE_DURATION_US = 1548.3517299999999
ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_comparator_manifest() -> tuple[dict[str, Any], str]:
    path = ROOT / "docs" / "p9-comparator-manifest.json"
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("comparator manifest is not newline-complete")
    value = json.loads(raw)
    if (
        not isinstance(value, dict)
        or value.get("proposed_system") != "QUIET"
        or not isinstance(value.get("rows"), dict)
        or not isinstance(value.get("structural_controls"), list)
    ):
        raise ValueError("comparator manifest is malformed")
    return value, hashlib.sha256(raw).hexdigest()


def comparison_contract(manifest: dict[str, Any], system: str) -> tuple[bool, str]:
    entry = manifest["rows"].get(system)
    if isinstance(entry, dict):
        allowed, status = entry.get("numeric_comparison_allowed"), entry.get("status")
        if not isinstance(allowed, bool) or not isinstance(status, str) or not status:
            raise ValueError(f"manifest contract is malformed for {system}")
        return allowed, status
    if system in manifest["structural_controls"]:
        return False, "structural-only"
    raise ValueError(f"system {system!r} is absent from comparator manifest")


def historical_port_contract(
    manifest: dict[str, Any], system: str
) -> tuple[bool, str]:
    """Keep this legacy Thor-port smoke outside the current numeric boundary.

    The manifest's XSched row describes the separately verified current native
    runtime gate.  This older Whisper artifact is labelled as a Thor port and
    lacks that native gate's current workload, trace, and application contract;
    allowing the manifest row to promote it would silently mix two evidence
    generations.
    """
    _allowed, status = comparison_contract(manifest, system)
    if system == "XSched":
        return False, f"historical-port-not-current-native-gate:{status}"
    return _allowed, status


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def numeric(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def measured_rows(
    summary: dict[str, Any], deadline_us: float, manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    if (
        summary.get("kind") != "p9-dependent-small-stress-smoke"
        or summary.get("workload") != "whisper-projection"
        or summary.get("deadline_source") != "frozen-independent-pipeline-p99-factor"
        or summary.get("deadline_lock", {}).get("sha256") != DEADLINE_LOCK_SHA256
        or not math.isclose(numeric(summary.get("deadline_us"), "deadline"), deadline_us)
    ):
        raise ValueError("vendor/QUIET summary contract differs")
    by_name = {
        row.get("system"): row
        for row in summary.get("results", [])
        if isinstance(row, dict)
    }
    rows = []
    for name in ("NVIDIA MIG", "NVIDIA MPS", "QUIET"):
        row = by_name.get(name)
        if not isinstance(row, dict):
            raise ValueError(f"summary lacks {name}")
        requests = row.get("pipeline_requests")
        misses = row.get("deadline_misses")
        if (
            isinstance(requests, bool)
            or not isinstance(requests, int)
            or requests != 1000
            or isinstance(misses, bool)
            or not isinstance(misses, int)
            or not 0 <= misses <= requests
            or row.get("payload_bytes") != 2_304_000
            or row.get("unique_payload_checksums", 0) < 2
            or row.get("unique_policy_output_checksums", 0) < 2
        ):
            raise ValueError(f"{name} measured evidence differs")
        rows.append({
            "system": name,
            "numeric_comparison_allowed": comparison_contract(manifest, name)[0],
            "comparison_status": comparison_contract(manifest, name)[1],
            "evidence": "measured-smoke",
            "requests": requests,
            "misses": misses,
            "dmr": misses / requests,
            "p99_us": numeric(row.get("pipeline_p99_us"), f"{name} p99"),
            "background_goodput_rps": numeric(
                row.get("background_goodput_rps"), f"{name} goodput"
            ),
            "observed_deadline_feasible": misses == 0,
        })
    return rows


def summarize(
    baseline_path: Path,
    boer_path: Path,
    orion_path: Path,
    xsched_path: Path,
    bless_path: Path,
    parvagpu_path: Path,
) -> dict[str, Any]:
    manifest, manifest_sha256 = load_comparator_manifest()
    baseline = load(baseline_path)
    boer = load(boer_path)
    orion = load(orion_path)
    xsched = load(xsched_path)
    bless = load(bless_path)
    parvagpu = load(parvagpu_path)
    deadline_us = numeric(baseline.get("deadline_us"), "deadline")

    if (
        boer.get("system") != "BOER"
        or boer.get("status") != "no-feasible-configuration"
        or boer.get("selected") is not None
        or boer.get("contract", {}).get("deadline_lock_sha256")
        != DEADLINE_LOCK_SHA256
        or not math.isclose(
            numeric(boer.get("contract", {}).get("deadline_us"), "BOER deadline"),
            deadline_us,
        )
        or boer.get("provenance", {}).get("fidelity")
        != "algorithm-preserving-thor-port"
    ):
        raise ValueError("BOER evidence differs")
    hardware = [
        observation.get("metrics")
        for observation in boer.get("observations", [])
        if isinstance(observation, dict)
        and isinstance(observation.get("metrics"), dict)
        and observation["metrics"].get("worst_p99_ms") is not None
    ]
    if not hardware:
        raise ValueError("BOER has no hardware observations")
    best_boer = min(hardware, key=lambda item: numeric(item["worst_p99_ms"], "BOER p99"))
    boer_evidence = best_boer.get("evidence")
    if not isinstance(boer_evidence, dict):
        raise ValueError("BOER best observation lacks evidence")
    boer_pipeline_path = Path(str(boer_evidence.get("result_dir"))) / "pipeline.json"
    expected_pipeline_sha = boer_evidence.get("sha256", {}).get("pipeline.json")
    if not boer_pipeline_path.is_file() or sha256(boer_pipeline_path) != expected_pipeline_sha:
        raise ValueError("BOER best observation pipeline evidence differs")
    boer_pipeline = load(boer_pipeline_path)
    boer_requests = boer_pipeline.get("iterations")
    boer_misses = boer_pipeline.get("deadline_misses")
    if boer_requests != 1000 or boer_misses != 1000:
        raise ValueError("BOER best observation request counts differ")

    if (
        orion.get("kind") != "orion-dependent-whisper-numeric-smoke-verification"
        or orion.get("system") != "Orion"
        or orion.get("functional_gate_passed") is not True
        or orion.get("numeric_smoke_valid") is not True
        or orion.get("inputs", {}).get("deadline_lock_sha256")
        != DEADLINE_LOCK_SHA256
        or orion.get("run_contract", {}).get("max_be_duration_source")
        != "frozen-isolated-pipeline-p99"
        or not math.isclose(
            numeric(
                orion.get("run_contract", {}).get("max_be_duration_us"),
                "Orion maximum BE duration",
            ),
            ORION_MAX_BE_DURATION_US,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(numeric(orion.get("deadline_us"), "Orion deadline"), deadline_us)
    ):
        raise ValueError("Orion evidence differs")

    if (
        bless.get("kind") != "bless-thor-tensorrt-fidelity-gate"
        or bless.get("status") != "passed-functional-gates"
        or bless.get("numeric_comparison_allowed") is not False
        or bless.get("native_squad", {}).get("kernels") != 24
        or bless.get("native_squad", {}).get("estimators")
        != ["interference-free", "workload-equivalence"]
        or bless.get("tensorrt_replicas", {}).get("driver_launch_records", 0) <= 0
        or bless.get("activation_replication", {}).get("post_copy_inference_passed")
        is not True
    ):
        raise ValueError("BLESS fidelity gate differs")

    if (
        xsched.get("kind") != "xsched-dependent-whisper-numeric-smoke-verification"
        or xsched.get("system") != "XSched"
        or xsched.get("functional_gate_passed") is not True
        or xsched.get("numeric_smoke_valid") is not True
        or xsched.get("inputs", {}).get("deadline_lock_sha256")
        != DEADLINE_LOCK_SHA256
        or not math.isclose(
            numeric(xsched.get("deadline_us"), "XSched deadline"), deadline_us
        )
    ):
        raise ValueError("XSched evidence differs")

    if (
        parvagpu.get("system") != "ParvaGPU"
        or parvagpu.get("feasible") is not False
        or parvagpu.get("contract", {}).get("deadline_lock_sha256")
        != DEADLINE_LOCK_SHA256
        or not math.isclose(
            numeric(
                parvagpu.get("contract", {}).get("pipeline_deadline_us"),
                "ParvaGPU deadline",
            ),
            deadline_us,
        )
    ):
        raise ValueError("ParvaGPU evidence differs")

    measured = {
        row["system"]: row
        for row in measured_rows(baseline, deadline_us, manifest)
    }
    rows = [
        measured["NVIDIA MIG"],
        measured["NVIDIA MPS"],
        {
            "system": "BOER (Thor port)",
            "numeric_comparison_allowed": comparison_contract(manifest, "BOER")[0],
            "comparison_status": comparison_contract(manifest, "BOER")[1],
            "evidence": "measured-search-no-feasible-configuration",
            "requests": boer_requests,
            "misses": boer_misses,
            "dmr": numeric(best_boer["deadline_miss_rate"], "BOER DMR"),
            "p99_us": numeric(best_boer["worst_p99_ms"], "BOER p99") * 1000.0,
            "background_goodput_rps": None,
            "observed_deadline_feasible": False,
            "note": "no feasible point; minimum-p99 search observation shown",
        },
        {
            "system": "Orion (Thor port)",
            "numeric_comparison_allowed": historical_port_contract(manifest, "Orion")[0],
            "comparison_status": historical_port_contract(manifest, "Orion")[1],
            "evidence": "measured-smoke",
            "requests": orion["requests"],
            "misses": orion["misses"],
            "dmr": numeric(orion["dmr"], "Orion DMR"),
            "p99_us": numeric(orion["p99_us"], "Orion p99"),
            "background_goodput_rps": numeric(
                orion["background_goodput_rps"], "Orion goodput"
            ),
            "observed_deadline_feasible": False,
        },
        {
            "system": "XSched (Thor port)",
            "numeric_comparison_allowed": historical_port_contract(manifest, "XSched")[0],
            "comparison_status": historical_port_contract(manifest, "XSched")[1],
            "evidence": "measured-smoke",
            "requests": xsched["requests"],
            "misses": xsched["misses"],
            "dmr": numeric(xsched["dmr"], "XSched DMR"),
            "p99_us": numeric(xsched["p99_us"], "XSched p99"),
            "background_goodput_rps": numeric(
                xsched["background_window"]["completion_goodput_rps"],
                "XSched goodput",
            ),
            "observed_deadline_feasible": False,
        },
        {
            "system": "BLESS (Thor reimplementation)",
            "numeric_comparison_allowed": comparison_contract(manifest, "BLESS")[0],
            "comparison_status": comparison_contract(manifest, "BLESS")[1],
            "evidence": "functional-only",
            "requests": None,
            "misses": None,
            "dmr": None,
            "p99_us": None,
            "background_goodput_rps": None,
            "observed_deadline_feasible": None,
            "note": (
                "native squads, traced replicas, activation handoff, and selected-only "
                "logical launch admission pass at a verified safe boundary; independent "
                "boundary profiling plus common-workload scheduling remains"
            ),
        },
        measured["QUIET"],
    ]
    return {
        "schema_version": 1,
        "kind": "p9-published-sota-dependent-smoke",
        "scope": "functional-and-numeric-smoke-not-formal-statistics",
        "proposed_system": "QUIET",
        "workload": "Whisper Tiny -> 2.304-MiB coherent edge -> projection",
        "deadline_us": deadline_us,
        "deadline_lock_sha256": DEADLINE_LOCK_SHA256,
        "comparator_manifest": {
            "path": str((ROOT / "docs/p9-comparator-manifest.json").resolve()),
            "sha256": manifest_sha256,
        },
        "rows": rows,
        "secondary_controls": [{
            "system": "ParvaGPU (Thor port)",
            "status": "allocation-infeasible",
            "numeric_comparison_allowed": comparison_contract(manifest, "ParvaGPU")[0],
            "comparison_status": comparison_contract(manifest, "ParvaGPU")[1],
            "reason": parvagpu.get("reason"),
        }],
        "input_sha256": {
            "baseline": sha256(baseline_path),
            "boer": sha256(boer_path),
            "orion": sha256(orion_path),
            "xsched": sha256(xsched_path),
            "bless": sha256(bless_path),
            "parvagpu": sha256(parvagpu_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--boer", type=Path, required=True)
    parser.add_argument("--orion", type=Path, required=True)
    parser.add_argument("--xsched", type=Path, required=True)
    parser.add_argument("--bless", type=Path, required=True)
    parser.add_argument("--parvagpu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.baseline.resolve(), args.boer.resolve(), args.orion.resolve(),
        args.xsched.resolve(), args.bless.resolve(), args.parvagpu.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
