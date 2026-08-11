import importlib.util
import hashlib
import json
import pathlib
import struct
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis" / "build_application_prediction_trace.py"
SPEC = importlib.util.spec_from_file_location("build_prediction_trace", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_output(path: pathlib.Path, iteration: int = 2) -> None:
    value = struct.pack("<3f", 0.1, 0.9, 0.2)
    path.write_bytes(MODULE.parse.__globals__["MAGIC"] + struct.pack("<I", 1) + struct.pack("<Q", len(value)) + struct.pack("<I", iteration) + value)


class BuildApplicationPredictionTraceTest(unittest.TestCase):
    def test_repository_resnet10_output_map_matches_vendor_contract(self) -> None:
        value = json.loads((ROOT / "models" / "resnet10-output-class-map.json").read_text())
        self.assertEqual(value, {"0": "Car", "1": "RoadSign", "2": "TwoWheeler", "3": "Person"})

    def test_joins_external_label_and_wall_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            output = root / "output.bin"
            write_output(output)
            csv_path = root / "pipeline.csv"
            csv_path.write_text("request,wall_end_to_end_us,deadline_miss\n2,12.5,0\n")
            manifest = root / "requests.jsonl"
            manifest.write_text(json.dumps({
                "schema_version": 1, "iteration": 2, "request_id": "r0",
                "arrival_sequence": 0, "input_sha256": "a" * 64,
                "expected_label": "dog",
            }) + "\n")
            class_map = root / "classes.json"
            class_map.write_text(json.dumps({"1": "dog"}) + "\n")
            rows = MODULE.build(output, csv_path, manifest, class_map,
                                warmup=2, deadline_us=20.0)
        self.assertEqual(rows[0]["prediction"], "dog")
        self.assertTrue(rows[0]["correct"])
        self.assertEqual(rows[0]["wall_latency_us"], 12.5)

    def test_rejects_wall_csv_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            output = root / "output.bin"
            write_output(output)
            csv_path = root / "pipeline.csv"
            csv_path.write_text("request,wall_end_to_end_us,deadline_miss\n2,30.0,0\n")
            manifest = root / "requests.jsonl"
            manifest.write_text(json.dumps({
                "schema_version": 1, "iteration": 2, "request_id": "r0",
                "arrival_sequence": 0, "input_sha256": "a" * 64,
                "expected_label": "dog",
            }) + "\n")
            class_map = root / "classes.json"
            class_map.write_text(json.dumps({"1": "dog"}) + "\n")
            with self.assertRaisesRegex(ValueError, "classification differs"):
                MODULE.build(output, csv_path, manifest, class_map,
                             warmup=2, deadline_us=20.0)

    def test_input_binding_requires_and_matches_pipeline_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            output = root / "output.bin"
            write_output(output)
            manifest = root / "requests.jsonl"
            manifest.write_text(json.dumps({
                "schema_version": 1, "iteration": 2, "request_id": "r0",
                "arrival_sequence": 0, "input_sha256": "a" * 64,
                "expected_label": "dog",
            }) + "\n")
            class_map = root / "classes.json"
            class_map.write_text(json.dumps({"1": "dog"}) + "\n")
            pipeline = root / "pipeline.csv"
            pipeline.write_text(
                "request,input_sha256,wall_end_to_end_us,deadline_miss\n"
                + f"2,{'a' * 64},12.5,0\n"
            )
            rows = MODULE.build(
                output, pipeline, manifest, class_map,
                warmup=2, deadline_us=20.0, require_input_binding=True,
            )
        self.assertTrue(rows[0]["correct"])

    def test_resnet10_detection_mode_uses_two_tensors_and_canonical_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            output = root / "output.bin"
            cells = 40 * 23
            cov = [0.0] * (4 * cells)
            bbox = [0.0] * (16 * cells)
            payload = struct.pack("<" + "f" * len(cov), *cov) + struct.pack(
                "<" + "f" * len(bbox), *bbox
            )
            output.write_bytes(
                MODULE.parse.__globals__["MAGIC"] + struct.pack("<I", 2)
                + struct.pack("<Q", len(cov) * 4)
                + struct.pack("<Q", len(bbox) * 4)
                + struct.pack("<I", 2) + payload
            )
            csv_path = root / "pipeline.csv"
            csv_path.write_text("request,wall_end_to_end_us,deadline_miss\n2,12.5,0\n")
            manifest = root / "requests.jsonl"
            manifest.write_text(json.dumps({
                "schema_version": 1, "iteration": 2, "request_id": "r0",
                "arrival_sequence": 0, "input_sha256": "a" * 64,
                "expected_label": '{"detections":[]}',
            }) + "\n")
            class_map = root / "classes.json"
            class_map.write_text(json.dumps({
                "0": "Car", "1": "RoadSign", "2": "TwoWheeler", "3": "Person",
            }) + "\n")
            rows = MODULE.build(
                output, csv_path, manifest, class_map,
                warmup=2, deadline_us=20.0,
                prediction_mode="resnet10-detection",
            )
        self.assertEqual(rows[0]["prediction"], '{"detections":[]}')
        self.assertTrue(rows[0]["correct"])

    def test_detection_correct_uses_class_and_iou_not_json_string(self) -> None:
        expected = '{"detections":[{"box":[100,100,100,100],"class":"Person"}]}'
        shifted = '{"detections":[{"box":[110,100,100,100],"class":"Person"}]}'
        self.assertTrue(MODULE._detection_request_correct(shifted, expected, iou_threshold=0.5))
        wrong_class = '{"detections":[{"box":[110,100,100,100],"class":"Car"}]}'
        self.assertFalse(MODULE._detection_request_correct(wrong_class, expected, iou_threshold=0.5))

    def test_asr_mode_binds_external_transcript_to_output_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            output = root / "output.bin"
            write_output(output)
            output_bytes = struct.pack("<3f", 0.1, 0.9, 0.2)
            transcript_map = root / "transcripts.json"
            transcript_map.write_text(json.dumps({
                hashlib.sha256(output_bytes).hexdigest(): "hello world",
            }) + "\n")
            csv_path = root / "pipeline.csv"
            csv_path.write_text("request,wall_end_to_end_us,deadline_miss\n2,12.5,0\n")
            manifest = root / "requests.jsonl"
            manifest.write_text(json.dumps({
                "schema_version": 1, "iteration": 2, "request_id": "r0",
                "arrival_sequence": 0, "input_sha256": "a" * 64,
                "expected_label": "hello world",
            }) + "\n")
            rows = MODULE.build(
                output, csv_path, manifest, None, warmup=2, deadline_us=20.0,
                prediction_mode="asr", asr_transcript_map=transcript_map,
            )
        self.assertEqual(rows[0]["prediction"], "hello world")
        self.assertTrue(rows[0]["correct"])

    def test_resnet10_decoder_ignores_parser_excluded_person_slot(self) -> None:
        cells = 40 * 23
        cov = [0.0] * (4 * cells)
        bbox = [0.0] * (16 * cells)
        w, h = 10, 10
        cell = w + h * 40
        class_offset = 3 * cells
        cov[class_offset + cell] = 1.0
        cx = (w * 16.0 + 0.5) / 35.0
        cy = (h * 16.0 + 0.5) / 35.0
        bbox_offset = 3 * 4 * cells
        bbox[bbox_offset + cell] = cx - 100.0 / 35.0
        bbox[bbox_offset + cells + cell] = cy - 100.0 / 35.0
        bbox[bbox_offset + 2 * cells + cell] = 200.0 / 35.0 - cx
        bbox[bbox_offset + 3 * cells + cell] = 200.0 / 35.0 - cy
        outputs = [{"values": cov}, {"values": bbox}]
        prediction = MODULE._decode_resnet10_detection(
            outputs, {0: "Car", 1: "RoadSign", 2: "TwoWheeler", 3: "Person"}
        )
        self.assertEqual(prediction, '{"detections":[]}')

    def test_resnet10_decoder_rejects_category_map_without_output_slots(self) -> None:
        with self.assertRaisesRegex(ValueError, "vendor labels.txt output slots"):
            MODULE._validate_resnet10_class_map({0: "Car", 1: "RoadSign", 2: "TwoWheeler"})


if __name__ == "__main__":
    unittest.main()
