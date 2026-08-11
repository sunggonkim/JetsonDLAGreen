import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "orion_driver_capture", ROOT / "baselines/orion/verify_driver_capture.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def record(sequence: int, api: str = "cuLaunchKernelEx") -> dict[str, object]:
    return {
        "schema_version": 1,
        "sequence": sequence,
        "api": api,
        "tid": 123,
        "start_monotonic_ns": 100 + sequence * 20,
        "end_monotonic_ns": 110 + sequence * 20,
        "function": f"0x{sequence + 1:x}",
        "stream": "0x2",
        "grid": [1, 2, 3],
        "block": [32, 1, 1],
        "shared_mem_bytes": 0,
        "attributes": 0,
        "result": 0,
    }


class OrionDriverCaptureTest(unittest.TestCase):
    def test_accepts_ex_launches(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.jsonl"
            path.write_text(json.dumps(record(0)) + "\n", encoding="utf-8")
            result = MODULE.verify(path)
            self.assertEqual(result["status"], "captured")
            self.assertFalse(result["numeric_comparison_allowed"])

    def test_rejects_non_ex_only_and_truncation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.jsonl"
            path.write_text(json.dumps(record(0, "cuLaunchKernel")) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cuLaunchKernelEx"):
                MODULE.verify(path)
            path.write_text(json.dumps(record(0)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "truncated"):
                MODULE.verify(path)

    def test_binds_benchmark_and_mig(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = root / "trace.jsonl"
            benchmark = root / "benchmark.json"
            trace.write_text(json.dumps(record(0)) + "\n", encoding="utf-8")
            benchmark.write_text(json.dumps({
                "schema_version": 1,
                "model": "resnet10-detection",
                "completed_requests": 6,
                "execution_environment": {"cuda_visible_devices": "MIG-big"},
                "gpu": {"multiprocessors": 12},
            }) + "\n", encoding="utf-8")
            result = MODULE.verify(trace, benchmark, 6, "MIG-big")
            self.assertEqual(result["benchmark"]["completed_requests"], 6)
            with self.assertRaisesRegex(ValueError, "MIG UUID"):
                MODULE.verify(trace, benchmark, 6, "MIG-other")


if __name__ == "__main__":
    unittest.main()
