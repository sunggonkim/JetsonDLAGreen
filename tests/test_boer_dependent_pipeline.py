#!/usr/bin/env python3

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "boer_dependent_pipeline",
    ROOT / "baselines" / "boer" / "evaluate_dependent_pipeline.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BoerDependentPipelineTest(unittest.TestCase):
    def test_boer_uses_complementary_mps_shares(self) -> None:
        self.assertEqual(MODULE.quota_pair(25), (25, 75))
        self.assertEqual(MODULE.quota_pair(90), (90, 10))
        with self.assertRaisesRegex(ValueError, "complementary"):
            MODULE.quota_pair(60)

    def test_preserves_boer_p99_rule_and_reports_dmr(self) -> None:
        metrics = MODULE.metrics_from_results(
            {
                "status": "ok",
                "checksum_failures": 0,
                "iterations": 1000,
                "end_to_end_us": {"p99": 750.0},
                "deadline_misses": 10,
                "pipeline_rps": 1100.0,
            },
            {"throughput_per_second": 700.0},
            760.0,
            1000,
        )
        self.assertEqual(metrics["feasible"], 1.0)
        self.assertEqual(metrics["deadline_miss_rate"], 0.01)
        self.assertEqual(metrics["worst_p99_ms"], 0.75)

    def test_uses_corrected_validation_excluded_whisper_metric(self) -> None:
        metrics = MODULE.metrics_from_results(
            {
                "status": "ok",
                "checksum_failures": 0,
                "iterations": 1000,
                "deadline_mode": "validation-excluded",
                "end_to_end_us": {"p99": 7000.0},
                "stage_latency_us": {
                    "validation_excluded_end_to_end_p99": 1590.0
                },
                "deadline_misses": 0,
                "pipeline_rps": 600.0,
            },
            {"throughput_per_second": 250.0},
            1620.0,
            1000,
        )
        self.assertEqual(metrics["feasible"], 1.0)
        self.assertEqual(metrics["worst_p99_ms"], 1.59)
        self.assertEqual(
            MODULE.workload_contract("whisper-projection"),
            ("whisper-tiny-encoder", "validation-excluded"),
        )

    def test_rejects_payload_corruption(self) -> None:
        with self.assertRaisesRegex(ValueError, "correctness"):
            MODULE.metrics_from_results(
                {
                    "status": "ok",
                    "checksum_failures": 1,
                    "iterations": 1,
                },
                {"throughput_per_second": 1.0},
                760.0,
                1,
            )


if __name__ == "__main__":
    unittest.main()
