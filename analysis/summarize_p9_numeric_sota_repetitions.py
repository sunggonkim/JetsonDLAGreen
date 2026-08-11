#!/usr/bin/env python3
"""Replay and aggregate the six-treatment mechanism-ablation campaign."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.summarize_p9_dependent_repetitions import (  # noqa: E402
    DMR_TARGET,
    clopper_pearson_upper,
    finite,
    percentile,
    replay_trace,
    sha256,
)
from scripts.run_p9_numeric_sota_smoke import (  # noqa: E402
    CANONICAL_NAMES,
    SYSTEM_BY_NAME,
    williams_orders,
)


EXPECTED_ACTIONS = {
    "NVIDIA MIG": (100, 100, "producer"),
    "NVIDIA MPS": (100, 100, "producer"),
    "Quota-only provisioning": (90, 10, "producer"),
    "Partition-only planning": (90, 10, "producer"),
    "Full-DAG quiescence": (100, 100, "pipeline"),
    "QUIET": (100, 100, "producer"),
}


def resolve_trace(summary_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    direct = (ROOT / path).resolve()
    if direct.is_file():
        return direct
    return (summary_path.parent / path).resolve()


def mean_t95(values: list[float]) -> dict[str, float]:
    if len(values) < 2 or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("paired ratio confidence interval needs positive finite repeats")
    mean = statistics.fmean(values)
    half = float(t.ppf(0.975, len(values) - 1)) * statistics.stdev(values) / math.sqrt(
        len(values)
    )
    return {"mean": mean, "lower": mean - half, "upper": mean + half}


def summarize(paths: list[Path]) -> dict[str, Any]:
    if len(paths) != 6:
        raise ValueError("mechanism campaign requires exactly six Williams sequences")
    expected_orders = williams_orders()
    seen_orders: set[tuple[str, ...]] = set()
    per_system: dict[str, list[dict[str, Any]]] = {
        name: [] for name in CANONICAL_NAMES
    }
    inputs: list[dict[str, str]] = []
    contract: dict[str, Any] | None = None
    for summary_path in paths:
        run = json.loads(summary_path.read_text(encoding="utf-8"))
        current_contract = {
            "kind": run.get("kind"),
            "workload": run.get("workload"),
            "background": run.get("background"),
            "deadline_lock": run.get("deadline_lock"),
            "iterations_per_system": run.get("iterations_per_system"),
        }
        if contract is None:
            contract = current_contract
        elif current_contract != contract:
            raise ValueError("mechanism campaign contracts differ")
        if (
            run.get("kind") != "p9-numeric-mechanism-smoke"
            or run.get("order_design") != "six-treatment-williams"
        ):
            raise ValueError("input is not a Williams mechanism run")
        index = run.get("sequence_index")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 6:
            raise ValueError("invalid Williams sequence index")
        order = tuple(run.get("execution_order", ()))
        if order != expected_orders[index] or order in seen_orders:
            raise ValueError("Williams order differs or is duplicated")
        seen_orders.add(order)
        rows = run.get("results")
        if not isinstance(rows, list) or tuple(row.get("system") for row in rows) != order:
            raise ValueError("result rows differ from execution order")
        inputs.append({"path": str(summary_path), "sha256": sha256(summary_path)})
        for row in rows:
            name = row["system"]
            expected = EXPECTED_ACTIONS[name]
            actual = (
                row.get("producer_quota_percent"),
                row.get("background_quota_percent"),
                row.get("gate_scope"),
            )
            if actual != expected:
                raise ValueError(f"{name} executed action differs from frozen action")
            trace = row.get("request_trace")
            if not isinstance(trace, dict):
                raise ValueError(f"{name} lacks raw trace provenance")
            trace_path = resolve_trace(summary_path, trace["path"])
            lock = run["deadline_lock"]
            lock_path = resolve_trace(summary_path, lock["path"])
            if sha256(lock_path) != lock["sha256"]:
                raise ValueError("deadline lock hash differs")
            deadline = finite(
                json.loads(lock_path.read_text(encoding="utf-8"))["deadline_us"],
                "deadline",
            )
            replay = replay_trace(trace_path, trace["sha256"], deadline)
            if (
                replay["requests"] != row.get("pipeline_requests")
                or replay["misses"] != row.get("deadline_misses")
                or not math.isclose(
                    percentile(replay["latencies"], 0.99),
                    finite(row.get("pipeline_p99_us"), "pipeline p99"),
                    rel_tol=1e-9,
                    abs_tol=0.01,
                )
            ):
                raise ValueError(f"{name} raw replay differs from summary")
            per_system[name].append(
                {
                    "latencies": replay["latencies"],
                    "requests": replay["requests"],
                    "misses": replay["misses"],
                    "goodput": finite(row["background_goodput_rps"], "goodput"),
                    "trace": {"path": str(trace_path), "sha256": trace["sha256"]},
                }
            )
    if seen_orders != set(expected_orders):
        raise ValueError("Williams design is incomplete")
    systems: dict[str, dict[str, Any]] = {}
    for name in CANONICAL_NAMES:
        samples = per_system[name]
        requests = sum(item["requests"] for item in samples)
        misses = sum(item["misses"] for item in samples)
        latencies = [value for item in samples for value in item["latencies"]]
        goodputs = [item["goodput"] for item in samples]
        cp95 = clopper_pearson_upper(misses, requests)
        scheduler_feasible = name != "Partition-only planning"
        systems[name] = {
            "runs": len(samples),
            "requests": requests,
            "misses": misses,
            "observed_dmr": misses / requests,
            "dmr_cp95_upper": cp95,
            "dmr_target": DMR_TARGET,
            "slo_confidence_qualified": cp95 <= DMR_TARGET,
            "scheduler_feasible": scheduler_feasible,
            "same_slo_goodput_comparable": scheduler_feasible and cp95 <= DMR_TARGET,
            "pooled_p99_us": percentile(latencies, 0.99),
            "pooled_p999_us": percentile(latencies, 0.999),
            "maximum_us": max(latencies),
            "background_goodput_rps_mean": statistics.fmean(goodputs),
            "background_goodput_rps_range": [min(goodputs), max(goodputs)],
            "trace_inputs": [item["trace"] for item in samples],
        }
    quiet = systems["QUIET"]
    full_dag = systems["Full-DAG quiescence"]
    defined = quiet["same_slo_goodput_comparable"] and full_dag[
        "same_slo_goodput_comparable"
    ]
    descriptive_ratio = (
        quiet["background_goodput_rps_mean"]
        / full_dag["background_goodput_rps_mean"]
        if full_dag["background_goodput_rps_mean"] > 0
        else None
    )
    paired_ratios = [
        quiet_sample["goodput"] / full_dag_sample["goodput"]
        for quiet_sample, full_dag_sample in zip(
            per_system["QUIET"],
            per_system["Full-DAG quiescence"],
            strict=True,
        )
    ]
    paired_ci = mean_t95(paired_ratios)
    comparison = {
        "defined": defined,
        "quiet_vs_full_dag_quiescence_goodput_ratio": descriptive_ratio if defined else None,
        "descriptive_ratio_not_for_claim": descriptive_ratio if not defined else None,
        "paired_run_ratios": paired_ratios,
        "paired_ratio_mean_t95": paired_ci,
        "gain_supported": defined and paired_ci["lower"] > 1.0,
    }
    return {
        "schema_version": 1,
        "kind": "p9-numeric-mechanism-williams-aggregate",
        "scope": "raw-replayed-balanced-campaign-with-exact-binomial-screen",
        "published_system_comparison": False,
        "proposed_system": "QUIET",
        "contract": contract,
        "order_design": "six-treatment-williams",
        "inputs": inputs,
        "systems": systems,
        "primary_feasible_comparison": comparison,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize([path.resolve() for path in args.input])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
