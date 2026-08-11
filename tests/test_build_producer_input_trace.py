import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_producer_input_trace.py"
SPEC = importlib.util.spec_from_file_location("build_input_trace", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BuildProducerInputTraceTest(unittest.TestCase):
    def test_packs_dense_hashed_fixed_size_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            first = root / "a.bin"
            second = root / "b.bin"
            first.write_bytes(b"abcd")
            second.write_bytes(b"efgh")
            rows = []
            for iteration, sample in enumerate((first, second)):
                rows.append({
                    "iteration": iteration,
                    "sample_id": f"sample-{iteration}",
                    "path": str(sample),
                    "input_sha256": MODULE._sha(sample),
                })
            listing = root / "samples.jsonl"
            listing.write_text("".join(json.dumps(row) + "\n" for row in rows))
            output = root / "inputs.bin"
            result = MODULE.build(listing, output)
            raw = output.read_bytes()
        self.assertEqual(result["format"], "JDGINT1")
        self.assertEqual(result["record_count"], 2)
        self.assertEqual(raw[:8], MODULE.MAGIC)

    def test_rejects_reused_input_inode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            sample = root / "sample.bin"
            sample.write_bytes(b"abcd")
            digest = MODULE._sha(sample)
            listing = root / "samples.jsonl"
            listing.write_text("".join(json.dumps({
                "iteration": i, "sample_id": str(i), "path": str(sample),
                "input_sha256": digest,
            }) + "\n" for i in range(2)))
            with self.assertRaisesRegex(ValueError, "inode"):
                MODULE.build(listing, root / "inputs.bin")


if __name__ == "__main__":
    unittest.main()
