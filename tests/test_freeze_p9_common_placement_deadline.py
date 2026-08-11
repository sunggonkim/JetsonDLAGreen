import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import freeze_p9_common_placement_deadline as common


class CommonPlacementDeadlineTest(unittest.TestCase):
    def _lock(self, root: Path, placement: str, deadline: float) -> Path:
        source = root / f"{placement}.json"
        contract = {
            "workload": "resnet-control", "payload_bytes": 14720,
            "transport": "registered-shared-sysmem-direct-binding", "deadline_mode": "wall",
            "warmup": 20, "samples_per_block": 300, "slo_factor": 1.1,
            "producer_quota_percent": 100, "cpu": 13,
            "placement_variant": placement,
        }
        source.write_text(json.dumps({"kind": "fixture"}) + "\n", encoding="utf-8")
        # Build-time replay is patched in this unit test; source lock shape is tested directly.
        value = {
            "kind": "p9-dependent-pipeline-deadline-lock",
            "source_summary": str(source), "contract": contract,
            "deadline_us": deadline,
            "artifacts": {"binary": {"sha256": "a"}, "source": {"sha256": "b"}},
        }
        path = root / f"{placement}-lock.json"
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        return path

    def test_requires_both_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                with mock.patch.object(common, "build_lock", return_value={"kind": "p9-dependent-pipeline-deadline-lock"}):
                    common.build_common_lock([self._lock(root, "fixed-1g-producer-2g-consumer", 700.0)] * 2)


if __name__ == "__main__":
    unittest.main()
