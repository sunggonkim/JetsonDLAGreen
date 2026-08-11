from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p9_orion_resnet_control_smoke.sh"


class OrionResnetRunnerTest(unittest.TestCase):
    def test_runner_is_syntax_valid_and_binds_common_contract(self) -> None:
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)
        text = RUNNER.read_text(encoding="utf-8")
        for required in (
            "freeze_p9_pipeline_deadline.py",
            "freeze_p9_common_placement_deadline.py",
            "deadline_kind",
            ".contract.workload",
            "--workload resnet-control",
            "--deadline-mode wall",
            "--checksum-trace-csv",
            "--orion-profile-aware true",
            "verify_resnet_control_smoke.py",
            "SHA256SUMS",
        ):
            self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
