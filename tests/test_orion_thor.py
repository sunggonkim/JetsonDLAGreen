import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "orion_thor", ROOT / "baselines/orion/run_thor.py"
)
assert SPEC is not None and SPEC.loader is not None
run_thor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_thor
SPEC.loader.exec_module(run_thor)


class OrionThorTest(unittest.TestCase):
    def test_upstream_policy_surface_is_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            scheduler = source / "src/scheduler"
            scheduler.mkdir(parents=True)
            (scheduler / "scheduler_eval.cpp").write_text(
                "sm_threshold hp_limit op_info_0.duration", encoding="utf-8"
            )
            completed = mock.Mock(stdout=run_thor.UPSTREAM_COMMIT + "\n")
            with mock.patch.object(run_thor.subprocess, "run", return_value=completed):
                run_thor.verify_upstream(source)

    def test_wrong_revision_is_rejected(self):
        completed = mock.Mock(stdout="0" * 40 + "\n")
        with mock.patch.object(run_thor.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(ValueError, "pinned upstream"):
                run_thor.verify_upstream(Path("/tmp"))


if __name__ == "__main__":
    unittest.main()
