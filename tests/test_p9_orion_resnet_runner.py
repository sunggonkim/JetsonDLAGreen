from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p9_orion_resnet_control_smoke.sh"
IMAGENETTE_RUNNER = ROOT / "scripts/run_p9_orion_resnet50_imagenette_smoke.sh"


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

    def test_imagenette_runner_uses_the_locked_deadline_for_accuracy(self) -> None:
        subprocess.run(["bash", "-n", str(IMAGENETTE_RUNNER)], check=True)
        text = IMAGENETTE_RUNNER.read_text(encoding="utf-8")
        self.assertIn("accuracy_deadline_us=${ACCURACY_DEADLINE_US:-}", text)
        self.assertIn("accuracy_deadline_us=${accuracy_deadline_us:-$deadline_us}", text)
        self.assertNotIn("ACCURACY_DEADLINE_US:-1000000", text)
        self.assertIn("reference-predictions-current-deadline.jsonl", text)
        self.assertIn("reference-current-deadline.csv", text)
        self.assertIn('--reference-trace "$reference_predictions"', text)
        self.assertIn('--reference-pipeline-csv "$reference_pipeline"', text)


if __name__ == "__main__":
    unittest.main()
