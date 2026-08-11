import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_application_accuracy", ROOT / "analysis/verify_application_accuracy.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _row(index: int, *, prediction: str = "cat", latency: float = 100.0) -> dict:
    return {
        "schema_version": 1,
        "request_id": f"request-{index}",
        "arrival_sequence": index,
        "input_sha256": f"{index + 1:064x}",
        "expected_label": "cat",
        "prediction": prediction,
        "correct": prediction == "cat",
        "output_sha256": f"{index + 100:064x}",
        "deadline_us": 1000.0,
        "wall_latency_us": latency,
        "deadline_miss": latency > 1000.0,
    }


def _asr_row(index: int, expected: str, prediction: str) -> dict:
    row = _row(index, prediction=prediction)
    row["expected_label"] = expected
    row["correct"] = MODULE._asr_request_correct(
        prediction, expected, max_wer=0.5,
    )
    return row


def _write_trace(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_output_trace(path: Path, count: int) -> None:
    # JDGOUT1: one fixed-size post-completion tensor per request.
    raw = bytearray(b"JDGOUT1\x00")
    raw += struct.pack("<I", 1) + struct.pack("<Q", 4)
    for iteration in range(count):
        raw += struct.pack("<I", iteration) + struct.pack("<I", iteration)
    path.write_bytes(bytes(raw))


def _output_sha(index: int) -> str:
    return hashlib.sha256(struct.pack("<I", index)).hexdigest()


def _write_pipeline(path: Path, rows: list[dict], *, warmup: int = 0) -> None:
    lines = ["request,input_sha256,wall_end_to_end_us,deadline_miss\n"]
    for index, row in enumerate(rows):
        lines.append(
            f"{warmup + index},{row['input_sha256']},{row['wall_latency_us']},"
            f"{int(row['deadline_miss'])}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


class ApplicationAccuracyGateTest(unittest.TestCase):
    def _files(self, root: Path) -> dict[str, Path]:
        files = {}
        for name, content in {
            "dataset": "".join(json.dumps({
                "schema_version": 1,
                "sample_id": f"sample-{index}",
                "input_sha256": f"{index + 1:064x}",
                "expected_label": "cat",
            }) + "\n" for index in range(4)),
            "reference_engine": "reference engine\n",
            "candidate_engine": "candidate engine\n",
        }.items():
            path = root / f"{name}.bin"
            path.write_text(content, encoding="utf-8")
            files[name] = path
        return files

    def test_passes_shared_trace_and_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = self._files(root)
            reference = root / "reference.jsonl"
            candidate = root / "candidate.jsonl"
            rows = [_row(index) for index in range(4)]
            for index, row in enumerate(rows):
                row["output_sha256"] = _output_sha(index)
            _write_trace(reference, rows)
            _write_trace(candidate, rows)
            result = MODULE.verify(
                reference, candidate, **files, workload="vision-control",
                task="classification", deadline_us=1000.0,
            )
            self.assertTrue(result["numeric_comparison_allowed"])
            self.assertEqual(result["candidate_accuracy"], 1.0)
            self.assertEqual(
                result["candidate_accuracy_by_label"],
                {"cat": {"requests": 4, "correct": 4}},
            )
            self.assertEqual(result["accuracy_delta"], 0.0)
            self.assertEqual(result["reference_trace_path"], str(reference.resolve()))
            self.assertEqual(result["candidate_trace_path"], str(candidate.resolve()))
            self.assertEqual(result["dataset_manifest_path"], str(files["dataset"].resolve()))
            self.assertEqual(
                result["reference_trace_sha256"], hashlib.sha256(reference.read_bytes()).hexdigest()
            )

    def test_rejects_input_or_label_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = self._files(root)
            reference = root / "reference.jsonl"
            candidate = root / "candidate.jsonl"
            _write_trace(reference, [_row(0)])
            changed = _row(0)
            changed["input_sha256"] = "f" * 64
            _write_trace(candidate, [changed])
            with self.assertRaisesRegex(ValueError, "input_sha256"):
                MODULE.verify(
                    reference, candidate, **files, workload="vision-control",
                    task="classification", deadline_us=1000.0,
                )

    def test_formal_gate_binds_raw_post_completion_traces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = self._files(root)
            reference = root / "reference.jsonl"
            candidate = root / "candidate.jsonl"
            reference_output = root / "reference.out"
            candidate_output = root / "candidate.out"
            rows = [_row(index) for index in range(4)]
            for index, row in enumerate(rows):
                row["output_sha256"] = _output_sha(index)
            _write_trace(reference, rows)
            _write_trace(candidate, rows)
            _write_output_trace(reference_output, len(rows))
            _write_output_trace(candidate_output, len(rows))
            result = MODULE.verify(
                reference, candidate, **files, workload="vision-control",
                task="classification", deadline_us=1000.0,
                reference_output_trace=reference_output,
                candidate_output_trace=candidate_output,
                require_output_traces=True,
            )
            self.assertTrue(result["application_output_trace_required"])
            self.assertEqual(result["reference_output_trace"]["record_count"], 4)

    def test_formal_gate_rejects_missing_raw_output_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = self._files(root)
            reference = root / "reference.jsonl"
            candidate = root / "candidate.jsonl"
            rows = [_row(0)]
            _write_trace(reference, rows)
            _write_trace(candidate, rows)
            with self.assertRaisesRegex(ValueError, "output traces"):
                MODULE.verify(
                    reference, candidate, **files, workload="vision-control",
                    task="classification", deadline_us=1000.0,
                    require_output_traces=True,
                )

    def test_formal_gate_binds_production_input_and_wall_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = self._files(root)
            reference = root / "reference.jsonl"
            candidate = root / "candidate.jsonl"
            rows = [_row(index) for index in range(2)]
            _write_trace(reference, rows)
            _write_trace(candidate, rows)
            reference_csv = root / "reference.csv"
            candidate_csv = root / "candidate.csv"
            _write_pipeline(reference_csv, rows, warmup=2)
            _write_pipeline(candidate_csv, rows, warmup=2)
            result = MODULE.verify(
                reference, candidate, **files, workload="vision-control",
                task="classification", deadline_us=1000.0,
                reference_pipeline_csv=reference_csv,
                candidate_pipeline_csv=candidate_csv,
                require_input_binding=True,
                pipeline_warmup=2,
            )
            self.assertEqual(result["application_input_binding_contract"], "passed")
            self.assertEqual(result["reference_pipeline_csv"]["measured_requests"], 2)

    def test_formal_gate_rejects_pipeline_hash_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = self._files(root)
            reference = root / "reference.jsonl"
            candidate = root / "candidate.jsonl"
            rows = [_row(0)]
            _write_trace(reference, rows)
            _write_trace(candidate, rows)
            reference_csv = root / "reference.csv"
            candidate_csv = root / "candidate.csv"
            _write_pipeline(reference_csv, rows)
            changed = dict(rows[0])
            changed["input_sha256"] = "f" * 64
            _write_pipeline(candidate_csv, [changed])
            with self.assertRaisesRegex(ValueError, "pipeline input_sha256"):
                MODULE.verify(
                    reference, candidate, **files, workload="vision-control",
                    task="classification", deadline_us=1000.0,
                    reference_pipeline_csv=reference_csv,
                    candidate_pipeline_csv=candidate_csv,
                    require_input_binding=True,
                )

    def test_formal_gate_rejects_prediction_trace_not_matching_raw_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = self._files(root)
            reference = root / "reference.jsonl"
            candidate = root / "candidate.jsonl"
            reference_output = root / "reference.out"
            candidate_output = root / "candidate.out"
            rows = [_row(0)]
            _write_trace(reference, rows)
            _write_trace(candidate, rows)
            _write_output_trace(reference_output, 1)
            _write_output_trace(candidate_output, 1)
            with self.assertRaisesRegex(ValueError, "output trace differs"):
                MODULE.verify(
                    reference, candidate, **files, workload="vision-control",
                    task="classification", deadline_us=1000.0,
                    reference_output_trace=reference_output,
                    candidate_output_trace=candidate_output,
                    require_output_traces=True,
                )

    def test_rejects_accuracy_delta_and_inconsistent_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = self._files(root)
            reference = root / "reference.jsonl"
            candidate = root / "candidate.jsonl"
            _write_trace(reference, [_row(0), _row(1)])
            _write_trace(candidate, [_row(0, prediction="dog"), _row(1, prediction="dog")])
            with self.assertRaisesRegex(ValueError, "accuracy"):
                MODULE.verify(
                    reference, candidate, **files, workload="vision-control",
                    task="classification", deadline_us=1000.0,
                )

    def test_rejects_equal_but_unacceptable_absolute_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = self._files(root)
            reference = root / "reference.jsonl"
            candidate = root / "candidate.jsonl"
            rows = [_row(0, prediction="dog"), _row(1, prediction="dog")]
            _write_trace(reference, rows)
            _write_trace(candidate, rows)
            with self.assertRaisesRegex(ValueError, "below minimum.*cat=0/2"):
                MODULE.verify(
                    reference, candidate, **files, workload="vision-control",
                    task="classification", deadline_us=1000.0,
                )

    def test_structured_detection_trace_uses_iou_metric(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = self._files(root)
            label = '{"detections":[{"box":[100,100,100,100],"class":"Person"}]}'
            prediction = '{"detections":[{"box":[110,100,100,100],"class":"Person"}]}'
            reference = root / "reference.jsonl"
            candidate = root / "candidate.jsonl"
            rows = [_row(0)]
            for row in rows:
                row["expected_label"] = label
                row["prediction"] = prediction
                row["correct"] = True
                row["input_sha256"] = "1" * 64
                row["output_sha256"] = _output_sha(0)
            files["dataset"].write_text(json.dumps({
                "schema_version": 1, "sample_id": "sample-0",
                "input_sha256": "1" * 64, "expected_label": label,
            }) + "\n", encoding="utf-8")
            _write_trace(reference, rows)
            _write_trace(candidate, rows)
            result = MODULE.verify(
                reference, candidate, **files, workload="vision-control",
                task="object-detection", deadline_us=1000.0,
            )
            self.assertTrue(result["numeric_comparison_allowed"])
            self.assertEqual(result["candidate_accuracy"], 1.0)

    def test_asr_gate_reports_word_error_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = self._files(root)
            files["dataset"].write_text("".join(json.dumps({
                "schema_version": 1,
                "sample_id": f"sample-{index}",
                "input_sha256": f"{index + 1:064x}",
                "expected_label": "hello world",
            }) + "\n" for index in range(2)), encoding="utf-8")
            reference = root / "reference.jsonl"
            candidate = root / "candidate.jsonl"
            _write_trace(reference, [
                _asr_row(0, "hello world", "hello world"),
                _asr_row(1, "hello world", "hello world"),
            ])
            _write_trace(candidate, [
                _asr_row(0, "hello world", "hello word"),
                _asr_row(1, "hello world", "hello world"),
            ])
            result = MODULE.verify(
                reference, candidate, **files, workload="whisper-control",
                task="asr", deadline_us=1000.0, asr_max_wer=0.5,
                asr_wer_tolerance=0.5,
            )
        self.assertEqual(result["reference_wer"], 0.0)
        self.assertEqual(result["candidate_wer"], 0.25)
        self.assertEqual(result["wer_delta"], 0.25)

    def test_asr_gate_rejects_candidate_wer_regression(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = self._files(root)
            files["dataset"].write_text(json.dumps({
                "schema_version": 1, "sample_id": "sample-0",
                "input_sha256": f"{1:064x}", "expected_label": "hello world",
            }) + "\n", encoding="utf-8")
            reference = root / "reference.jsonl"
            candidate = root / "candidate.jsonl"
            _write_trace(reference, [_asr_row(0, "hello world", "hello world")])
            _write_trace(candidate, [_asr_row(0, "hello world", "hello word")])
            with self.assertRaisesRegex(ValueError, "word error rate differs"):
                MODULE.verify(
                    reference, candidate, **files, workload="whisper-control",
                    task="asr", deadline_us=1000.0, asr_max_wer=0.5,
                    asr_wer_tolerance=0.1,
                )

    def test_rejects_trace_label_not_bound_to_dataset_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = self._files(root)
            reference = root / "reference.jsonl"
            candidate = root / "candidate.jsonl"
            forged = _row(0)
            forged["expected_label"] = "dog"
            forged["prediction"] = "dog"
            forged["correct"] = True
            _write_trace(reference, [forged])
            _write_trace(candidate, [forged])
            with self.assertRaisesRegex(ValueError, "dataset manifest label"):
                MODULE.verify(
                    reference, candidate, **files, workload="vision-control",
                    task="classification", deadline_us=1000.0,
                )
            bad = _row(0, latency=1001.0)
            bad["deadline_miss"] = False
            _write_trace(candidate, [bad])
            with self.assertRaisesRegex(ValueError, "classification"):
                MODULE.verify(
                    reference, candidate, **files, workload="vision-control",
                    task="classification", deadline_us=1000.0,
                )


if __name__ == "__main__":
    unittest.main()
