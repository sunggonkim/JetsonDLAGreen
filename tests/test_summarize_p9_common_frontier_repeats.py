import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.summarize_p9_common_frontier_repeats import summarize


def fixture(system: str, name: str) -> dict:
    return {
        "kind": "p9-dependent-small-stress-smoke", "workload": "resnet-control",
        "latency_contract": "production-wall-arrival-to-completion", "deadline_mode": "wall",
        "checksum_mode": "inline", "deadline_us": 773.730452, "iterations": 100,
        "background_period_ms": 4.0, "background_offered_rps": 250.0,
        "deadline_lock": {"sha256": "a" * 64},
        "results": [{"system": system, "correctness_validated": True,
                     "pipeline_requests": 100, "deadline_misses": 0,
                     "wall_pipeline_p99_us": 700.0, "background_goodput_rps": 240.0,
                     "checksum_failures": 0, "unique_payload_checksums": 4,
                     "unique_policy_output_checksums": 4}],
        "name": name,
    }


class CommonFrontierRepeatTest(unittest.TestCase):
    def test_requires_all_systems(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for system in ("QUIET", "NVIDIA MPS", "XSched"):
                for index in range(3):
                    path = root / f"{system.replace(' ', '-')}-{index}.json"
                    path.write_text(json.dumps(fixture(system, path.name)) + "\n")
                    paths.append(path)
            result = summarize(paths)
        self.assertEqual(result["systems"]["QUIET"]["repeat_count"], 3)
        self.assertEqual(result["systems"]["XSched"]["qualified_repeat_count"], 3)
        self.assertEqual(result["statistical_unit"], "paired-session")
        self.assertEqual(result["numeric_frontier_systems"], ["NVIDIA MPS", "QUIET"])
        self.assertEqual(result["exploratory_systems"], ["XSched"])
        self.assertFalse(result["ranking_allowed"])
        self.assertEqual(
            result["paired_session_statistics"]["NVIDIA MPS"]["status"],
            "descriptive",
        )
        self.assertEqual(
            result["paired_session_statistics"]["XSched"]["p99_delta_us_quiet_minus_baseline"]["t95"]["n"],
            3,
        )

    def test_rejects_contract_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for system in ("QUIET", "NVIDIA MPS", "XSched"):
                for index in range(3):
                    value = fixture(system, f"{system}-{index}")
                    if system == "XSched" and index == 0:
                        value["deadline_us"] = 700.0
                    path = root / f"{system.replace(' ', '-')}-{index}.json"
                    path.write_text(json.dumps(value) + "\n")
                    paths.append(path)
            with self.assertRaisesRegex(ValueError, "common contract"):
                summarize(paths)

    def test_rejects_same_numeric_deadline_with_different_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for system in ("QUIET", "NVIDIA MPS", "XSched"):
                for index in range(3):
                    value = fixture(system, f"{system}-{index}")
                    if system == "XSched":
                        value["deadline_lock"] = {"sha256": "b" * 64}
                    path = root / f"{system.replace(' ', '-')}-{index}.json"
                    path.write_text(json.dumps(value) + "\n")
                    paths.append(path)
            with self.assertRaisesRegex(ValueError, "common contract"):
                summarize(paths)

    def test_accepts_static_full_gate_only_as_optional_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for system in ("QUIET", "NVIDIA MPS", "XSched", "Static full gating"):
                for index in range(3):
                    path = root / f"{system.replace(' ', '-')}-{index}.json"
                    path.write_text(json.dumps(fixture(system, path.name)) + "\n")
                    paths.append(path)
            result = summarize(paths)
        self.assertEqual(result["systems"]["Static full gating"]["repeat_count"], 3)
        self.assertTrue(result["systems"]["Static full gating"]["all_repeats_slo_qualified"])

    def test_rejects_unbalanced_repeat_ids_from_paired_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for system in ("QUIET", "NVIDIA MPS", "XSched"):
                for index in range(3):
                    value = fixture(system, f"{system}-r{index + 1:02d}")
                    suffix = "-r04" if system == "XSched" and index == 2 else f"-r{index + 1:02d}"
                    path = root / f"{system.replace(' ', '-')}{suffix}.json"
                    path.write_text(json.dumps(value) + "\n")
                    paths.append(path)
            result = summarize(paths)
        self.assertEqual(result["paired_session_statistics"]["NVIDIA MPS"]["status"], "descriptive")
        self.assertEqual(result["paired_session_statistics"]["XSched"]["status"], "unbalanced-session-keys")


if __name__ == "__main__":
    unittest.main()
