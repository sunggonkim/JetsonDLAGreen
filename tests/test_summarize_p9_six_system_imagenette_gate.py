from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_p9_six_system_imagenette_gate",
    ROOT / "analysis/summarize_p9_six_system_imagenette_gate.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SixSystemImageNetteGateTest(unittest.TestCase):
    def test_fixed_public_order_starts_with_quiet_and_keeps_mig(self) -> None:
        self.assertEqual(
            MODULE.SYSTEM_ORDER,
            ("QUIET", "NVIDIA MIG", "NVIDIA MPS", "XSched", "Orion", "Pantheon"),
        )

    def test_zero_miss_directional_gate_cannot_certify_formal_target(self) -> None:
        self.assertGreater(MODULE._cp95(0, 90), 0.0005)

    def test_declared_record_hash_is_replayed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.bin"
            path.write_bytes(b"trace")
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                MODULE._record({"path": str(path), "sha256": "0" * 64}, "trace")

    def test_current_hardware_evidence_replays_when_available(self) -> None:
        required = (
            MODULE.DEFAULT_COMMON, MODULE.DEFAULT_LOCK, MODULE.DEFAULT_QUIET,
            MODULE.DEFAULT_MIG, MODULE.DEFAULT_MIG_COLOCATED,
            MODULE.DEFAULT_MPS, MODULE.DEFAULT_XSCHED, MODULE.DEFAULT_ORION,
            MODULE.DEFAULT_PANTHEON,
        )
        if not all(path.is_file() for path in required):
            self.skipTest("local ignored hardware corpus is unavailable")
        value = MODULE.summarize(
            common_path=MODULE.DEFAULT_COMMON,
            deadline_path=MODULE.DEFAULT_LOCK,
            quiet_path=MODULE.DEFAULT_QUIET,
            mig_path=MODULE.DEFAULT_MIG,
            mig_colocated_path=MODULE.DEFAULT_MIG_COLOCATED,
            mps_path=MODULE.DEFAULT_MPS,
            xsched_path=MODULE.DEFAULT_XSCHED,
            orion_path=MODULE.DEFAULT_ORION,
            pantheon_path=MODULE.DEFAULT_PANTHEON,
        )
        self.assertEqual(value["system_order"], list(MODULE.SYSTEM_ORDER))
        mig = value["systems"]["NVIDIA MIG"]
        self.assertTrue(mig["background_goodput_applicable"])
        self.assertGreater(mig["background_goodput_rps"], 249.0)
        self.assertEqual(
            mig["topology"],
            "colocated-2g-critical-dag-with-isolated-1g-best-effort",
        )
        variants = value["mig_topology_comparisons"]
        self.assertFalse(
            variants["split-critical-stages"]["background_goodput_applicable"]
        )
        self.assertEqual(
            mig["source"], variants["colocated-critical-dag"]["source"]
        )
        self.assertFalse(value["mig_topology_constraints"]["three_way_1g_supported"])
        self.assertEqual(value["systems"]["Orion"]["misses"], 90)
        self.assertEqual(value["systems"]["Pantheon"]["misses"], 2)
        self.assertFalse(value["formal"])
        self.assertFalse(value["ranking_allowed"])

    def test_checked_in_summary_keeps_every_row(self) -> None:
        path = ROOT / "paper/eurosys27/generated/p9-six-system-imagenette-gate.json"
        if not path.is_file():
            self.skipTest("generated summary has not been built")
        value = json.loads(path.read_bytes())
        self.assertEqual(value["system_order"], list(MODULE.SYSTEM_ORDER))
        self.assertEqual(tuple(value["systems"]), MODULE.SYSTEM_ORDER)
        self.assertIn("NVIDIA MIG", value["systems"])
        self.assertIn("split-critical-stages", value["mig_topology_comparisons"])
        self.assertIn("colocated-critical-dag", value["mig_topology_comparisons"])


if __name__ == "__main__":
    unittest.main()
