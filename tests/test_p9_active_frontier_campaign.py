import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_p9_active_frontier_campaign.sh"
LEGACY = ROOT / "scripts" / "run_p9_sota_performance_campaign.sh"


class ActiveFrontierCampaignTest(unittest.TestCase):
    def test_help_describes_only_active_numeric_matrix(self) -> None:
        result = subprocess.run(
            [str(SCRIPT), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("NVIDIA MPS, XSched (Thor port), and QUIET", result.stdout)
        self.assertIn("never runs BOER", result.stdout)
        self.assertIn("PRODUCER_INPUT_TRACE", result.stdout)
        self.assertIn("APPLICATION_ACCURACY_GATE", result.stdout)
        self.assertIn("APPLICATION_ACCURACY_REFERENCE_TRACE", result.stdout)
        self.assertIn("APPLICATION_ACCURACY_CLASS_MAP", result.stdout)
        self.assertIn("WARMUP", result.stdout)

    def test_learned_workload_trace_is_forwarded_to_williams_runner(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('PRODUCER_INPUT_TRACE="${PRODUCER_INPUT_TRACE:-}"', text)
        self.assertIn(
            'PRODUCER_INPUT_TRACE is required for learned ResNet workloads',
            text,
        )
        self.assertIn(
            'APPLICATION_ACCURACY_GATE is required for learned workloads',
            text,
        )
        self.assertIn(
            'application accuracy gate does not meet the frozen accuracy floor',
            text,
        )
        self.assertIn(
            'args+=(--producer-input-trace "${PRODUCER_INPUT_TRACE}")',
            text,
        )
        self.assertIn('WARMUP="${WARMUP:-10}"', text)
        self.assertIn('--warmup "${WARMUP}"', text)
        self.assertIn('analysis/bind_p9_active_accuracy.py', text)
        self.assertIn('bind_sequence_accuracy', text)
        self.assertIn('unsupported deadline lock kind', text)
        self.assertIn('freeze_p9_pipeline_deadline.py', text)

    def test_legacy_entrypoint_delegates_without_local_policy_labels(self) -> None:
        text = LEGACY.read_text(encoding="utf-8")
        self.assertIn("run_p9_active_frontier_campaign.sh", text)
        self.assertNotIn("POLICY_ORDER", text)
        self.assertNotIn("uncoordinated-borrow", text)


if __name__ == "__main__":
    unittest.main()
