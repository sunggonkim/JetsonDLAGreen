import importlib.util
import csv
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "common_sota_williams_summary",
    ROOT / "analysis/summarize_p9_common_sota_williams.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CommonSotaWilliamsSummaryTest(unittest.TestCase):
    def test_replays_current_six_sequence_hardware_pilot(self) -> None:
        paths = []
        for index in range(6):
            candidates = sorted(ROOT.glob(f"results/p9-common-sota-williams-seq{index}-100r-*/summary.json"))
            if not candidates:
                self.skipTest("Williams hardware pilot is absent")
            paths.append(candidates[-1])
        result = MODULE.summarize(paths)
        self.assertEqual(result["systems"]["QUIET"]["misses"], 0)
        self.assertEqual(result["systems"]["XSched"]["misses"], 600)
        self.assertFalse(result["systems"]["QUIET"]["slo_confidence_qualified"])
        self.assertFalse(result["systems"]["QUIET"]["numeric_comparison_allowed"])
        self.assertIsNone(result["common_workload"])

    def test_requires_all_six_sequences(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly six"):
            MODULE.summarize([])

    def test_replays_formal_size_hardware_when_present(self) -> None:
        paths = []
        for index in range(6):
            candidates = sorted(ROOT.glob(
                f"results/p9-common-sota-williams-nonthermal-formal-seq{index}-1100r-*/summary.json"
            ))
            if not candidates:
                self.skipTest("formal-size Williams evidence is absent")
            paths.append(candidates[-1])
        result = MODULE.summarize(paths)
        quiet = result["systems"]["QUIET"]
        self.assertEqual(result["confidence_qualified_systems"], ["QUIET"])
        self.assertEqual(quiet["misses"], 0)
        self.assertTrue(quiet["slo_confidence_qualified"])
        self.assertFalse(quiet["numeric_comparison_allowed"])
        self.assertIsNone(result["common_workload"])
        self.assertLess(quiet["maximum_us"], 770.605407)

    def test_raw_trace_hash_and_deadline_classification_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.csv"
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=MODULE.TRACE_COLUMNS)
                writer.writeheader()
                row = {name: 0 for name in MODULE.TRACE_COLUMNS}
                row.update(request=0, wall_end_to_end_us=800.0, deadline_miss=0)
                writer.writerow(row)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "classification"):
                MODULE.replay_trace(path, digest, 770.0)
            with self.assertRaisesRegex(ValueError, "hash"):
                MODULE.replay_trace(path, "0" * 64, 770.0)

    def test_whisper_replay_uses_validation_excluded_latency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.csv"
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=MODULE.TRACE_COLUMNS)
                writer.writeheader()
                row = {name: 0 for name in MODULE.TRACE_COLUMNS}
                row.update(
                    request=0,
                    validation_excluded_end_to_end_us=700.0,
                    wall_end_to_end_us=7000.0,
                    deadline_miss=0,
                )
                writer.writerow(row)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            replay = MODULE.replay_trace(
                path, digest, 770.0, "validation_excluded_end_to_end_us",
            )
            self.assertEqual(replay["misses"], 0)
            self.assertEqual(replay["stages"]["producer_compute_us"], [0.0])

    def test_input_bound_trace_replays_against_arrival_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.csv"
            digest = "a" * 64
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=MODULE.TRACE_COLUMNS_WITH_INPUT)
                writer.writeheader()
                row = {name: 0 for name in MODULE.TRACE_COLUMNS_WITH_INPUT}
                row.update(request=10, input_sha256=digest, wall_end_to_end_us=700.0, deadline_miss=0)
                writer.writerow(row)
            trace_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            replay = MODULE.replay_trace(
                path, trace_hash, 770.0, expected_inputs={10: digest},
            )
            self.assertTrue(replay["input_binding"])
            path.write_text(path.read_text().replace(digest, "b" * 64), encoding="utf-8")
            changed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "raw input binding"):
                MODULE.replay_trace(
                    path, changed_hash, 770.0, expected_inputs={10: digest},
                )

    def test_manifest_contract_is_recorded(self) -> None:
        paths = []
        for index in range(6):
            candidates = sorted(ROOT.glob(
                f"results/p9-common-sota-williams-seq{index}-100r-*/summary.json"
            ))
            if not candidates:
                self.skipTest("Williams hardware pilot is absent")
            paths.append(candidates[-1])
        result = MODULE.summarize(paths)
        manifest = result["comparator_manifest"]
        self.assertEqual(manifest["path"], str((ROOT / "docs/p9-comparator-manifest.json").resolve()))
        self.assertEqual(len(manifest["sha256"]), 64)
        self.assertFalse(result["systems"]["Orion"]["numeric_comparison_allowed"])
        self.assertFalse(result["systems"]["gpulet"]["numeric_comparison_allowed"])


if __name__ == "__main__":
    unittest.main()
