#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sota_scope", ROOT / "analysis/summarize_p9_sota_scope.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SotaScopeTest(unittest.TestCase):
    def test_requires_positive_and_negative_controls(self) -> None:
        values = {
            "boer_independent": {"system": "BOER", "status": "selected", "selected": {
                "sm_percent": 90, "complement_sm_percent": 10, "offered_rps": 500,
                "metrics": {"worst_p99_ms": 1.5},
            }},
            "boer_dependent": {"system": "BOER", "status": "no-feasible-configuration"},
            "parva_independent": {"system": "ParvaGPU", "all_slos_met": True, "services": []},
            "parva_dependent": {"system": "ParvaGPU", "feasible": False, "reason": "segments"},
            "orion": {"system": "Orion", "numeric_comparison_allowed": False, "reason": "API"},
            "quiet": {"proposed_system": "QUIET", "status": "selected", "selected_plan": {}},
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for name, value in values.items():
                path = Path(directory) / f"{name}.json"
                path.write_text(json.dumps(value))
                paths[name] = path
            result = MODULE.summarize(paths)
        self.assertEqual(result["proposed_system"], "QUIET")
        self.assertEqual(result["comparators"][0]["system"], "BOER")


if __name__ == "__main__":
    unittest.main()
