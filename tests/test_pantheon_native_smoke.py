import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "pantheon_native_smoke", ROOT / "baselines/pantheon/verify_native_smoke.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


VALID_LOG = """\
[EXEC:START] HIGH_PRIORITY 1
[SCHE] 2 1 1
[EXEC:BLOCK] HIGH_PRIORITY 3 1 0 0
[EXEC:BLOCK] HIGH_PRIORITY 4 1 0 1
[EXEC:EXIT] HIGH_PRIORITY 1 10 9 1 0 1 0.9
[SCHE] 11 1 1
[EXEC:BLOCK] HIGH_PRIORITY 12 1 1 0
[EXEC:EXIT] HIGH_PRIORITY 1 15 14 1 1 0 0.7
[EXEC:STOP] HIGH_PRIORITY 16
"""


class PantheonNativeSmokeTest(unittest.TestCase):
    def fixture(self, root: pathlib.Path, log: str = VALID_LOG):
        log_path = root / "runtime.log"
        environment_path = root / "environment.json"
        log_path.write_text(log, encoding="utf-8")
        environment_path.write_text(json.dumps({
            "schema_version": 1,
            "mig_uuid": "MIG-test",
            "torch_version": "2.9.0",
            "cuda_available": True,
            "gpu": {"name": "NVIDIA Thor MIG 2g.0gb", "multiprocessors": 12},
            "gemm_checksum": 1.0,
        }), encoding="utf-8")
        return log_path, environment_path

    def test_accepts_full_and_early_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            result = MODULE.verify(*self.fixture(pathlib.Path(directory)))
            self.assertFalse(result["numeric_comparison_allowed"])
            self.assertEqual(result["early_exit_job"]["last_block"], 0)

    def test_rejects_missing_early_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = VALID_LOG.replace("1 0 0.7", "1 1 0.9")
            with self.assertRaisesRegex(ValueError, "early exit"):
                MODULE.verify(*self.fixture(pathlib.Path(directory), bad))


if __name__ == "__main__":
    unittest.main()
