#!/usr/bin/env python3

import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dependent_repetitions",
    ROOT / "analysis" / "summarize_p9_dependent_repetitions.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DependentRepetitionsTest(unittest.TestCase):
    def test_exact_zero_miss_upper_bound(self) -> None:
        self.assertAlmostEqual(
            MODULE.clopper_pearson_upper(0, 4000),
            1.0 - math.pow(0.05, 1.0 / 4000),
        )

    def make_run(
        self,
        root: Path,
        repeat: int,
        tamper: bool = False,
        order: tuple[str, ...] = MODULE.SYSTEM_ORDER,
    ) -> Path:
        rows = []
        for system_index, system in enumerate(order):
            trace = root / f"r{repeat}-{system_index}.csv"
            with trace.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=MODULE.TRACE_COLUMNS)
                writer.writeheader()
                for request, latency in enumerate((1500.0, 1600.0, 1700.0)):
                    row = {name: 0 for name in MODULE.TRACE_COLUMNS}
                    row.update(
                        request=request,
                        validation_excluded_end_to_end_us=latency,
                        wall_end_to_end_us=latency + 100,
                        deadline_miss=int(latency > 1620),
                    )
                    writer.writerow(row)
            digest = hashlib.sha256(trace.read_bytes()).hexdigest()
            rows.append(
                {
                    "system": system,
                    "pipeline_requests": 3,
                    "deadline_misses": 1,
                    "pipeline_p99_us": MODULE.percentile([1500, 1600, 1700], 0.99),
                    "background_goodput_rps": 249.0 + repeat,
                    "request_trace": {"path": str(trace), "sha256": digest},
                }
            )
        if tamper:
            rows[0]["deadline_misses"] = 0
        summary = {
            "kind": "p9-dependent-small-stress-smoke",
            "deadline_us": 1620.0,
            "iterations": 3,
            "background_period_ms": 4.0,
            "background_offered_rps": 250.0,
            "producer_quota_percent": 100,
            "background_quota_percent": 100,
            "workload": "whisper-projection",
            "quiet_gate_scope": "producer",
            "deadline_source": "frozen-independent-pipeline-p99-factor",
            "deadline_lock": {"path": "/lock.json", "sha256": "a" * 64},
            "execution_order": list(order),
            "results": rows,
        }
        path = root / f"summary-{repeat}.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        return path

    def test_replays_traces_and_separates_ablation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = MODULE.summarize([self.make_run(root, 1), self.make_run(root, 2)])
        self.assertEqual(result["proposed_system"], "QUIET")
        self.assertEqual(
            result["scope"], "repeated-functional-smoke-with-exact-binomial-screen"
        )
        self.assertEqual(result["systems"]["QUIET"]["misses"], 2)
        self.assertNotIn("Process-stop ablation", result["systems"])
        self.assertIn("Process-stop ablation", result["ablations"])

    def test_rejects_summary_trace_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            one = self.make_run(root, 1, tamper=True)
            two = self.make_run(root, 2)
            with self.assertRaisesRegex(ValueError, "trace replay differs"):
                MODULE.summarize([one, two])

    def test_accepts_exact_four_run_williams_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                self.make_run(root, index, order=order)
                for index, order in enumerate(MODULE.WILLIAMS_4, 1)
            ]
            result = MODULE.summarize(paths)
        self.assertEqual(result["order_design"], "four-treatment-williams")


if __name__ == "__main__":
    unittest.main()
