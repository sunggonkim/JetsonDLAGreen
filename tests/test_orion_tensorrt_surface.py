#!/usr/bin/env python3

import importlib.util
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "orion_surface", ROOT / "baselines" / "orion" / "analyze_tensorrt_surface.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class OrionTensorRtSurfaceTest(unittest.TestCase):
    def test_parses_driver_launches_without_runtime_launches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "stats.csv"
            path.write_text(
                "Time (%),Total Time (ns),Num Calls,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name\n"
                "10,100,108,1,1,1,1,0,cuLaunchKernelEx\n"
                "1,10,4,1,1,1,1,0,cudaMemcpyAsync\n",
                encoding="utf-8",
            )
            counts = MODULE.api_counts(path)
        self.assertEqual(counts["cuLaunchKernelEx"], 108)
        self.assertEqual(counts.get("cudaLaunchKernel", 0), 0)

    def test_rejects_empty_stats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "stats.csv"
            path.write_text("not,a,report\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no CUDA"):
                MODULE.api_counts(path)


if __name__ == "__main__":
    unittest.main()
