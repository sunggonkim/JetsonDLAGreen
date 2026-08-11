#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib
import subprocess

import torch


MODEL_NAME = "pantheon-smoke-cnn"


class Block0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, stride=2, padding=1),
            torch.nn.ReLU(),
        )

    def forward(self, value):
        return self.net(value)


class Block1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, value):
        return self.net(value)


class Branch(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(channels, 10),
        )

    def forward(self, value):
        return self.net(value)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--proto-dir", type=pathlib.Path, required=True)
    parser.add_argument("--mig-uuid", required=True)
    args = parser.parse_args()

    model_root = args.output / "model-repository" / MODEL_NAME
    model_files = model_root / "model_files"
    model_files.mkdir(parents=True)

    torch.manual_seed(7)
    value0 = torch.randn(1, 3, 224, 224)
    block0 = Block0().eval()
    value1 = block0(value0)
    block1 = Block1().eval()
    value2 = block1(value1)
    modules = (
        ("block_00.pth", block0, value0),
        ("block_01.pth", block1, value1),
        ("branch_00.pth", Branch(16).eval(), value1),
        ("branch_01.pth", Branch(32).eval(), value2),
    )
    for name, module, example in modules:
        torch.jit.trace(module, example).save(str(model_files / name))

    (model_root / "config.pbtxt").write_text(
        'name: "pantheon-smoke-cnn"\n'
        "dims: 1\ndims: 3\ndims: 224\ndims: 224\n"
        "block_profile { id: 0 latency: 2000 }\n"
        "block_profile { id: 1 latency: 2000 }\n"
        "exit_profile { id: 0 latency: 500 accuracy: 0.70 }\n"
        "exit_profile { id: 1 latency: 500 accuracy: 0.90 }\n",
        encoding="ascii",
    )
    workload_text = args.output / "workload.textproto"
    workload_text.write_text(
        'workload { model_name: "pantheon-smoke-cnn" release: 0 '
        "deadline: 100000 id: 0 }\n"
        'workload { model_name: "pantheon-smoke-cnn" release: 0 '
        "deadline: 3000 id: 1 }\n",
        encoding="ascii",
    )
    workload_binary = args.output / "workload.pb"
    with workload_text.open("rb") as source, workload_binary.open("wb") as output:
        subprocess.run(
            [
                "protoc",
                f"--proto_path={args.proto_dir}",
                "--encode=Workloads",
                str(args.proto_dir / "workload.proto"),
            ],
            stdin=source,
            stdout=output,
            check=True,
        )

    device = torch.cuda.get_device_properties(0)
    if device.name != "NVIDIA Thor MIG 2g.0gb" or device.multi_processor_count != 12:
        raise RuntimeError("Pantheon smoke must use the fixed Thor 2g MIG instance")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Pantheon smoke must see exactly one MIG device")
    probe = torch.randn(64, 64, device="cuda")
    gemm_checksum = float((probe @ probe).sum())
    environment = {
        "schema_version": 1,
        "mig_uuid": args.mig_uuid,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": {"name": device.name, "multiprocessors": device.multi_processor_count},
        "gemm_checksum": gemm_checksum,
        "model_files": {
            path.name: sha256(path) for path in sorted(model_files.iterdir())
        },
    }
    (args.output / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
