import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_bless_trt_context_replica",
    ROOT / "baselines/bless/verify_trt_context_replica_smoke.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def benchmark(engine: Path) -> dict:
    return {
        "schema_version": 1,
        "model": "distilbert-sst2",
        "role": "benchmark",
        "engine": str(engine),
        "completed_requests": 20,
        "execution_environment": {"mps_active_thread_percentage": 100},
        "gpu": {"name": "NVIDIA Thor MIG 1g.0gb", "multiprocessors": 8},
        "config": {"include_transfers": True},
        "release_to_completion": {"p99_ms": 1.0},
        "measurement_start_monotonic_ns": 0,
        "measurement_end_monotonic_ns": 0,
    }


class BlessTrtContextReplicaSmokeTest(unittest.TestCase):
    def fixture(self, directory: Path) -> Path:
        engine = directory / "distilbert-sst2.engine"
        engine.write_bytes(b"engine")
        replicas = []
        for round_index in (0, 1):
            for sms in MODULE.EXPECTED_SMS:
                identity = round_index * 4 + MODULE.EXPECTED_SMS.index(sms)
                current = benchmark(engine.resolve())
                current["measurement_start_monotonic_ns"] = 1000 + identity * 100
                current["measurement_end_monotonic_ns"] = 1090 + identity * 100
                replicas.append(
                    {
                        "round": round_index,
                        "context_id": 1000 + sms,
                        "requested_sms": sms,
                        "actual_sms": sms,
                        "benchmark": current,
                    }
                )
        (directory / "result.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "bless-thor-trt-context-replica-smoke",
                    "replica_rounds": 2,
                    "replicas": replicas,
                }
            )
        )
        (directory / "stderr.txt").write_text("")
        trace = []
        for sequence, replica in enumerate(replicas):
            start = replica["benchmark"]["measurement_start_monotonic_ns"] + 10
            trace.append(
                {
                    "schema_version": 1,
                    "sequence": sequence,
                    "api": "cuLaunchKernelEx",
                    "tid": 1,
                    "start_monotonic_ns": start,
                    "end_monotonic_ns": start + 1,
                    "function": "0x1",
                    "stream": "0x2",
                    "grid": [1, 1, 1],
                    "block": [32, 1, 1],
                    "shared_mem_bytes": 0,
                    "attributes": 0,
                    "result": 0,
                }
            )
        (directory / "driver-launches.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in trace)
        )
        return engine.resolve()

    def test_accepts_reused_distinct_affinity_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            engine = self.fixture(directory)
            result = MODULE.verify(directory, engine)
        self.assertTrue(result["contexts_precreated_and_reused"])
        self.assertTrue(result["replica_launch_sequences_identical"])
        self.assertFalse(result["numeric_comparison_allowed"])

    def test_rejects_context_recreation_between_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            engine = self.fixture(directory)
            result = json.loads((directory / "result.json").read_text())
            result["replicas"][4]["context_id"] = 9999
            (directory / "result.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "not reused"):
                MODULE.verify(directory, engine)

    def test_rejects_launch_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            engine = self.fixture(directory)
            (directory / "stderr.txt").write_text("cudaErrorLaunchFailure\n")
            with self.assertRaisesRegex(ValueError, "stderr"):
                MODULE.verify(directory, engine)

    def test_rejects_replica_without_measured_driver_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            engine = self.fixture(directory)
            lines = (directory / "driver-launches.jsonl").read_text().splitlines()
            (directory / "driver-launches.jsonl").write_text(
                "\n".join(lines[:-1]) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "lacks measured"):
                MODULE.verify(directory, engine)

    def test_rejects_divergent_replica_launch_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            engine = self.fixture(directory)
            rows = [
                json.loads(line)
                for line in (directory / "driver-launches.jsonl").read_text().splitlines()
            ]
            rows[-1]["grid"] = [2, 1, 1]
            (directory / "driver-launches.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            with self.assertRaisesRegex(ValueError, "sequences differ"):
                MODULE.verify(directory, engine)


if __name__ == "__main__":
    unittest.main()
