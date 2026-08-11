import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "checksum_probe", ROOT / "analysis/summarize_p9_checksum_mode_probe.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def fixture(mode: str, *, lock: str = "a" * 64) -> dict:
    validated = mode == "inline"
    return {
        "kind": "p9-dependent-small-stress-smoke",
        "checksum_mode": mode,
        "workload": "resnet-control",
        "latency_contract": "production-wall-arrival-to-completion",
        "deadline_mode": "wall",
        "iterations": 100,
        "warmup": 10,
        "placement_variant": "fixed-1g-producer-2g-consumer",
        "deadline_us": 773.730452,
        "deadline_lock": {"sha256": lock},
        "results": [
            {
                "system": system,
                "pipeline_requests": 100,
                "deadline_misses": 0 if system == "QUIET" else 80,
                "wall_pipeline_p99_us": (
                    (700.0 if system == "QUIET" else 900.0)
                    + ({"inline": 0.0, "sampled": 10.0, "off": -20.0}[mode])
                ),
                "background_goodput_rps": 240.0,
                "correctness_validated": validated,
                "checksum_failures": 0 if validated else None,
                "unique_payload_checksums": 4 if validated else 0,
                "unique_policy_output_checksums": 4 if validated else 0,
            }
            for system in MODULE.SYSTEMS
        ],
    }


class ChecksumModeProbeTest(unittest.TestCase):
    def test_reports_modes_without_promoting_timing_only_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for mode in MODULE.MODES:
                path = root / f"{mode}.json"
                path.write_text(json.dumps(fixture(mode)) + "\n", encoding="utf-8")
                paths.append(path)
            result = MODULE.summarize(paths)
        self.assertEqual(result["claim_guard"], "timing-mode diagnostic only; not a numeric SLO frontier")
        self.assertFalse(result["systems"]["QUIET"]["modes"]["off"]["correctness_validated"])
        self.assertEqual(result["systems"]["NVIDIA MPS"]["modes"]["sampled"]["delta_p99_vs_inline_us"], 10.0)

    def test_rejects_contract_or_mode_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for mode in MODULE.MODES:
                value = fixture(mode, lock=("b" * 64 if mode == "off" else "a" * 64))
                path = root / f"{mode}.json"
                path.write_text(json.dumps(value) + "\n", encoding="utf-8")
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "execution contract"):
                MODULE.summarize(paths)


if __name__ == "__main__":
    unittest.main()
