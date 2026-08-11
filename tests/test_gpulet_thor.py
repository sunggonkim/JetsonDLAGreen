import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gpulet_thor", ROOT / "baselines/gpulet/run_thor.py"
)
assert SPEC is not None and SPEC.loader is not None
run_thor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_thor
SPEC.loader.exec_module(run_thor)


def profile(quota: int, p99: float, rate: float) -> run_thor.Profile:
    return run_thor.Profile(
        quota, 100 - quota, p99 * 0.8, p99, 1.0, rate, 0, "/x", "a" * 64
    )


class GpuletThorTest(unittest.TestCase):
    def test_selects_smallest_schedulable_best_fit(self):
        selected, feasible, decisions = run_thor.select_partition(
            [profile(25, 1.8, 500), profile(50, 1.6, 500), profile(75, 1.5, 500)],
            deadline_ms=1.7, background_target_rps=500,
        )
        self.assertTrue(feasible)
        self.assertEqual(selected.producer_quota, 50)
        self.assertEqual([row["schedulable"] for row in decisions], [False, True, True])

    def test_rate_is_part_of_feasibility(self):
        selected, feasible, _ = run_thor.select_partition(
            [profile(50, 1.6, 400), profile(75, 1.6, 480)],
            deadline_ms=1.7, background_target_rps=500,
        )
        self.assertTrue(feasible)
        self.assertEqual(selected.producer_quota, 75)

    def test_unschedulable_executes_diagnostic_not_fake_feasible(self):
        selected, feasible, _ = run_thor.select_partition(
            [profile(50, 2.0, 500), profile(90, 1.9, 500)],
            deadline_ms=1.7, background_target_rps=500,
        )
        self.assertFalse(feasible)
        self.assertEqual(selected.producer_quota, 90)

    def test_upstream_commit_is_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            scheduler = source / "src/scheduler"
            scheduler.mkdir(parents=True)
            (scheduler / "scheduler_incremental.cpp").write_text(
                "getMaxReturnPart getMinPart findBestFit", encoding="utf-8"
            )
            completed = mock.Mock(stdout=run_thor.UPSTREAM_COMMIT + "\n")
            with mock.patch.object(run_thor.subprocess, "run", return_value=completed):
                run_thor.verify_upstream(source)


if __name__ == "__main__":
    unittest.main()
