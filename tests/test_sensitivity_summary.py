#!/usr/bin/env python3
import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_p8_sensitivity",
    ROOT / "analysis" / "summarize_p8_sensitivity.py",
)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUMMARY
SPEC.loader.exec_module(SUMMARY)


def sample_run(label: str, *, guard: float | None = None) -> dict:
    return {
        "schema_version": 1,
        "config": {
            "experiment_label": label,
            "guard_override_ms": guard,
            "burst_size": 8,
            "period_ms": 12.0,
        },
        "deadline_ms": 5.3,
        "policies": [
            {
                "name": "profiled-guard",
                "deadline_miss_rate": 0.001,
                "critical_p99_ms_max": 5.1,
                "pressure_goodput_per_second": 500.0,
                "gate_overhead_mean_ms": 0.002,
            }
        ],
    }


class SensitivitySummaryTest(unittest.TestCase):
    def test_guard_point_key(self) -> None:
        self.assertEqual(SUMMARY.point_key(sample_run("guard", guard=2.0)), ("guard", 2.0))

    def test_missing_guard_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SUMMARY.point_key(sample_run("guard"))

    def test_metrics_require_one_policy(self) -> None:
        run = sample_run("burst")
        run["policies"].append(dict(run["policies"][0]))
        with self.assertRaises(ValueError):
            SUMMARY.metrics(run)

    def test_confidence_interval(self) -> None:
        result = SUMMARY.confidence([1.0, 2.0, 3.0])
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["mean"], 2.0)
        self.assertGreater(result["ci95"], 0.0)


if __name__ == "__main__":
    unittest.main()
