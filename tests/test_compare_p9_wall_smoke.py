import json
import tempfile
import unittest
from pathlib import Path

from analysis.compare_p9_wall_smoke import compare


def summary(system="QUIET", *, deadline_mode="wall", checksum_mode="inline", lock="abc"):
    return {
        "kind": "p9-dependent-small-stress-smoke",
        "workload": "resnet-control",
        "deadline_mode": deadline_mode,
        "latency_contract": "production-wall-arrival-to-completion",
        "checksum_mode": checksum_mode,
        "deadline_us": 770.0,
        "deadline_lock": {"sha256": lock},
        "background_period_ms": 4.0,
        "background_offered_rps": 250.0,
        "iterations": 100,
        "results": [{
            "system": system, "pipeline_requests": 100, "deadline_misses": 0,
            "pipeline_p99_us": 700.0 if system == "QUIET" else 900.0,
            "background_goodput_rps": 249.0, "deadline_mode": deadline_mode,
            "latency_contract": "production-wall-arrival-to-completion",
            "checksum_mode": checksum_mode, "correctness_validated": True,
        }],
    }


class WallSmokeComparisonTest(unittest.TestCase):
    def write(self, directory, name, value):
        path = Path(directory) / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_reports_paired_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            report = compare(
                self.write(directory, "quiet.json", summary()),
                self.write(directory, "mps.json", summary("NVIDIA MPS")),
            )
        self.assertFalse(report["formal"])
        self.assertEqual(report["quiet_delta_vs_baseline"]["p99_us_delta"], -200.0)

    def test_rejects_lock_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            left = self.write(directory, "quiet.json", summary())
            right = self.write(directory, "mps.json", summary("NVIDIA MPS", lock="different"))
            with self.assertRaisesRegex(ValueError, "incomparable"):
                compare(left, right)

    def test_rejects_checksum_off(self):
        with tempfile.TemporaryDirectory() as directory:
            left = self.write(directory, "quiet.json", summary())
            right = self.write(directory, "mps.json", summary("NVIDIA MPS", checksum_mode="off"))
            with self.assertRaisesRegex(ValueError, "incomparable"):
                compare(left, right)

    def test_selects_rows_from_runner_multirow_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            value = summary()
            value["results"].append(summary("NVIDIA MPS")["results"][0])
            path = self.write(directory, "combined.json", value)
            report = compare(path, path, "QUIET", "NVIDIA MPS")
        self.assertEqual(report["systems"], ["QUIET", "NVIDIA MPS"])
        self.assertEqual(report["quiet_delta_vs_baseline"]["p99_us_delta"], -200.0)


if __name__ == "__main__":
    unittest.main()
