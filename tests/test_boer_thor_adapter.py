#!/usr/bin/env python3
import importlib.util
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "boer_thor_adapter", ROOT / "baselines" / "boer" / "thor_adapter.py"
)
assert SPEC is not None and SPEC.loader is not None
BOER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BOER
SPEC.loader.exec_module(BOER)


class BoerThorAdapterTest(unittest.TestCase):
    @staticmethod
    def spec() -> dict:
        return {
            "schema_version": 1,
            "system": "BOER",
            "upstream_commit": BOER.UPSTREAM_COMMIT,
            "contract": {"pressure_layout": "1g+2g"},
            "evaluator_command": ["unused"],
            "seed": 4,
            "tenant_demands_rps": [500, 500],
            "static_capacity_profile": [
                {"sm_percent": 10, "max_rps": 200},
                {"sm_percent": 90, "max_rps": 1800},
            ],
            "candidates": [
                {"id": f"q{sm}-r{rps}", "sm_percent": sm, "offered_rps": rps}
                for sm in range(10, 100, 10)
                for rps in (100, 300)
            ],
        }

    def test_search_preserves_bounds_and_selects_measured_point(self) -> None:
        def evaluator(_spec, candidate):
            return {
                "feasible": 1.0,
                "slo_limit_ms": 6.0,
                "worst_p99_ms": 5.0,
                "served_rps_0": float(min(candidate.offered_rps, 500)),
                "served_rps_1": float(min(1000 - candidate.sm_percent * 5, 500)),
            }

        result = BOER.run_search(self.spec(), evaluator)
        self.assertEqual(result["system"], "BOER")
        self.assertGreaterEqual(len(result["observations"]), 6)
        self.assertLessEqual(len(result["observations"]), 20)
        self.assertTrue(result["selected"]["metrics"])
        self.assertEqual(
            result["selected"]["sm_percent"]
            + result["selected"]["complement_sm_percent"],
            100,
        )

    def test_dynamic_pruning_matches_upstream_dominance(self) -> None:
        failure = BOER.Candidate("failed", 50, 300)
        self.assertFalse(BOER.dynamically_feasible(BOER.Candidate("dominated", 40, 400), [failure]))
        self.assertTrue(BOER.dynamically_feasible(BOER.Candidate("more-sm", 60, 400), [failure]))
        self.assertTrue(BOER.dynamically_feasible(BOER.Candidate("less-rps", 40, 200), [failure]))

    def test_objective_matches_upstream_normalization(self) -> None:
        metrics = {
            "feasible": 1.0,
            "slo_limit_ms": 6.0,
            "worst_p99_ms": 5.0,
            "served_rps_0": 250.0,
            "served_rps_1": 500.0,
        }
        self.assertAlmostEqual(BOER.boer_objective(metrics, (500.0, 500.0)), 0.875)

    def test_next_point_uses_upstream_expected_improvement(self) -> None:
        domain = BOER.candidates_from_spec(self.spec())
        observations = [
            BOER.Observation(domain[0], 0.2, False, {}, "hardware"),
            BOER.Observation(domain[1], 0.8, True, {}, "hardware"),
            BOER.Observation(domain[2], 0.4, False, {}, "hardware"),
            BOER.Observation(domain[3], 0.6, True, {}, "hardware"),
            BOER.Observation(domain[4], 0.3, False, {}, "hardware"),
            BOER.Observation(domain[5], 0.5, True, {}, "hardware"),
        ]
        remaining = domain[6:]
        selected = BOER.select_expected_improvement(remaining, observations, domain)
        self.assertIn(selected, remaining)

    def test_can_preserve_no_feasible_hardware_result(self) -> None:
        spec = self.spec()
        spec["allow_no_feasible"] = True

        def evaluator(_spec, _candidate):
            return {
                "feasible": 0.0,
                "slo_limit_ms": 0.76,
                "worst_p99_ms": 1.0,
                "served_rps_0": 1.0,
                "served_rps_1": 1.0,
            }

        result = BOER.run_search(spec, evaluator)
        self.assertEqual(result["status"], "no-feasible-configuration")
        self.assertIsNone(result["selected"])
        self.assertGreaterEqual(len(result["observations"]), 6)
        self.assertEqual(result["search"]["acquisition"], "expected-improvement")
        self.assertEqual(result["search"]["xi"], 0.2)
        self.assertFalse(result["search"]["numeric_comparison_allowed"])

    def test_binds_raw_hardware_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            for name in ("pipeline.csv", "pipeline.json", "background.json"):
                (directory / name).write_text(name + "\n", encoding="utf-8")
            evidence = BOER.bind_evaluator_evidence({"result_dir": str(directory)})
        self.assertEqual(set(evidence["sha256"]), {
            "pipeline.csv", "pipeline.json", "background.json"
        })
        self.assertEqual(len(set(evidence["sha256"].values())), 3)

    def test_rejects_incomplete_hardware_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            (directory / "pipeline.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                BOER.bind_evaluator_evidence({"result_dir": str(directory)})


if __name__ == "__main__":
    unittest.main()
