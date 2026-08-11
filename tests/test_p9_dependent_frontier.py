#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dependent_frontier", ROOT / "analysis" / "summarize_p9_dependent_frontier.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DependentFrontierTest(unittest.TestCase):
    def test_charges_uncovered_quiet_guard(self) -> None:
        systems = []
        for system in MODULE.LABELS:
            systems.append(
                {
                    "system": system,
                    "pipeline_requests": 100,
                    "deadline_misses": 0,
                    "pipeline_p99_us": 700,
                    "stage_latency_us": {"producer_compute_p99": 600},
                    "payload_bytes": 14720,
                    "unique_payload_checksums": 4,
                    "unique_policy_output_checksums": 4,
                    "background_goodput_rps": 100,
                    "gate_p99_us": 1500 if system == "quiet" else None,
                }
            )
        raw = {
            "kind": "p9-dependent-small-stress-smoke",
            "deadline_us": 760,
            "background_offered_rps": 100,
            "results": systems,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps(raw))
            result = MODULE.summarize([path], 1000)
        quiet = next(row for row in result["rows"] if row["system"] == "QUIET")
        self.assertTrue(quiet["post_release_zero_miss"])
        self.assertFalse(quiet["arrival_bound_feasible"])
        self.assertEqual(quiet["uncovered_guard_us"], 500)
        self.assertEqual(
            [row["role"] for row in result["rows"]].count("proposed"), 1
        )


if __name__ == "__main__":
    unittest.main()
