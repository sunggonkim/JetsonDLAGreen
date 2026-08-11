import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PATCH = ROOT / "baselines" / "xsched" / "patches" / "thor-cuda13-tensorrt.patch"
README = ROOT / "baselines" / "xsched" / "README.md"


class XSchedPortContractTest(unittest.TestCase):
    def test_patch_preserves_xqueue_and_repairs_cuda13_lifetimes(self):
        text = PATCH.read_text(encoding="utf-8")
        self.assertIn("kCommandPropertyBlockingSubmit", text)
        self.assertIn("XSCHED_TRT_USER_STREAM_ONLY", text)
        self.assertIn("XQueueManager::AutoCreate", text)
        self.assertNotIn("SIGSTOP", text)

    def test_single_client_gate_is_not_reported_as_numeric_result(self):
        text = README.read_text(encoding="utf-8")
        self.assertIn("not a performance result", text)
        self.assertIn("two-client positive control", text)
        self.assertIn("bd494cb7a72958cd11900243a0798df00d856c6e", text)


if __name__ == "__main__":
    unittest.main()
