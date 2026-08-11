import struct
import tempfile
import unittest
from pathlib import Path

from baselines.pantheon.build_common_workload import build, read_arrivals


class PantheonCommonWorkloadBuilderTest(unittest.TestCase):
    def test_reads_and_encodes_dense_operational_arrivals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrival = root / "arrivals.bin"
            rows = []
            for index in range(2):
                rows.append(
                    struct.pack("<IIQ", index, index, index * 5_000_000)
                    + (f"{index + 1:064x}").encode("ascii")
                    + f"request-{index}".encode("ascii").ljust(64, b"\0")
                )
            arrival.write_bytes(
                b"JDGARR1\0" + struct.pack("<IIQ", 1, 2, 144) + b"".join(rows)
            )
            self.assertEqual(read_arrivals(arrival)[1]["release_us"], 5000)
            proto = root / "proto"
            proto.mkdir()
            (proto / "workload.proto").write_text(
                'syntax = "proto3"; message Workloads { repeated Workload workload = 1; } '
                'message Workload { string model_name = 1; uint64 release = 2; '
                'uint64 deadline = 3; int32 id = 6; }\n',
                encoding="ascii",
            )
            output = root / "workload.pb"
            result = build(
                arrival, output, proto_dir=proto,
                model_name="resnet50-imagenette", deadline_us=2224,
            )
            self.assertEqual(result["request_count"], 2)
            self.assertEqual(result["deadline_us"], 2224)
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
