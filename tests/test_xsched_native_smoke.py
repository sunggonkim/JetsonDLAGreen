import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "xsched_native_smoke", ROOT / "baselines/xsched/verify_native_smoke.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class XSchedNativeSmokeTest(unittest.TestCase):
    def fixture(self, root: pathlib.Path):
        base = {
            "schema_version": 1,
            "model": "resnet10-detection",
            "completed_requests": 10,
            "gpu": {"name": "NVIDIA Thor MIG 2g.0gb", "multiprocessors": 12},
            "execution_environment": {
                "pid": 101,
                "cuda_visible_devices": "MIG-test",
            },
            "measurement_start_monotonic_ns": 100,
            "measurement_end_monotonic_ns": 1000,
        }
        be = dict(base, role="pressure")
        hp = json.loads(json.dumps(base))
        hp.update({
            "role": "benchmark",
            "measurement_start_monotonic_ns": 300,
            "measurement_end_monotonic_ns": 700,
        })
        hp["execution_environment"]["pid"] = 202
        (root / "be.json").write_text(json.dumps(be), encoding="utf-8")
        (root / "hp.json").write_text(json.dumps(hp), encoding="utf-8")
        (root / "server.log").write_text(
            "schedule transition pid 101 operation 1 running 1 suspended 0\n"
            "schedule transition pid 202 operation 1 running 1 suspended 0\n"
            "schedule transition pid 101 operation 2 running 0 suspended 1\n"
            "schedule transition pid 101 operation 3 running 1 suspended 0\n",
            encoding="utf-8",
        )
        (root / "be.log").write_text("using global scheduler\n", encoding="utf-8")
        (root / "hp.log").write_text("using global scheduler\n", encoding="utf-8")
        return tuple(root / name for name in (
            "be.json", "hp.json", "server.log", "be.log", "hp.log"
        ))

    def test_accepts_real_hpf_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            result = MODULE.verify(*self.fixture(pathlib.Path(directory)))
            self.assertEqual(result["measurement_overlap_ns"], 400)
            self.assertFalse(result["numeric_comparison_allowed"])

    def test_rejects_missing_suspend_and_nonoverlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            paths = self.fixture(root)
            (root / "server.log").write_text(
                "schedule transition pid 101 operation 1 running 1 suspended 0\n"
                "schedule transition pid 202 operation 1 running 1 suspended 0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "did not run HP"):
                MODULE.verify(*paths)

            hp = json.loads((root / "hp.json").read_text(encoding="utf-8"))
            hp["measurement_start_monotonic_ns"] = 1001
            hp["measurement_end_monotonic_ns"] = 1100
            (root / "hp.json").write_text(json.dumps(hp), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "did not overlap"):
                MODULE.verify(*paths)

    def test_rejects_be_that_ends_before_hp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            paths = self.fixture(root)
            be = json.loads((root / "be.json").read_text(encoding="utf-8"))
            be["measurement_end_monotonic_ns"] = 600
            (root / "be.json").write_text(json.dumps(be), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "did not continue"):
                MODULE.verify(*paths)


if __name__ == "__main__":
    unittest.main()
