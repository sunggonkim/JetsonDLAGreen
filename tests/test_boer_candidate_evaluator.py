#!/usr/bin/env python3
import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "boer_candidate_evaluator",
    ROOT / "baselines" / "boer" / "evaluate_candidate.py",
)
assert SPEC is not None and SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EVALUATOR
SPEC.loader.exec_module(EVALUATOR)


class BoerCandidateEvaluatorTest(unittest.TestCase):
    def test_extracts_same_contract_metrics(self) -> None:
        summary = {
            "policies": [
                {
                    "name": "uncoordinated-borrow",
                    "goodput_by_modality": {"audio": 400.0, "language": 500.0},
                    "deadline_miss_rate": 0.0,
                    "critical_p99_ms_max": 5.2,
                }
            ]
        }
        metrics = EVALUATOR.metrics_from_summary(summary, 6.0, 0.0005)
        self.assertEqual(metrics["feasible"], 1.0)
        self.assertEqual(metrics["served_rps_0"], 400.0)
        self.assertEqual(metrics["served_rps_1"], 500.0)

    def test_search_uses_p99_while_preserving_stricter_dmr(self) -> None:
        summary = {
            "policies": [
                {
                    "name": "uncoordinated-borrow",
                    "goodput_by_modality": {"audio": 400.0, "language": 500.0},
                    "deadline_miss_rate": 0.01,
                    "critical_p99_ms_max": 5.9,
                }
            ]
        }
        metrics = EVALUATOR.metrics_from_summary(summary, 6.0, 0.0005)
        self.assertEqual(metrics["feasible"], 1.0)
        self.assertEqual(metrics["deadline_miss_rate"], 0.01)
        self.assertEqual(metrics["dmr_target"], 0.0005)


if __name__ == "__main__":
    unittest.main()
