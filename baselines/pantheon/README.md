# Pantheon Thor port

Pantheon (MobiSys 2024) is the edge-specific published competitor. The port is
pinned to upstream commit `1caa4321fe9f9902ffacb78978f11a32a7a62f64` from
`https://github.com/PantheonInfer/Pantheon`.

## Fidelity boundary

A valid row must execute these upstream mechanisms:

- offline DNN construction into block and early-exit TorchScript modules;
- per-block latency and per-exit accuracy profiles regenerated on Thor;
- the deadline scheduler that reduces exit depth when queued jobs would miss;
- the online block executor on a high-priority CUDA stream; and
- the authors' workload release/deadline format.

Using CUDA stream priority alone is NVIDIA MPS/priority, not Pantheon. Wrapping
an opaque TensorRT engine without Pantheon's block graph and scheduler is also
not a faithful Pantheon port.

## Thor native gate

The public artifact is directly relevant because it targets mobile edge GPUs
and Jetson. It now builds and executes natively on Thor using the CUDA 13 SBSA
PyTorch wheel. The port performs only mechanical compatibility work: regenerate
the checked-in protobuf output, prefer the system protobuf/fmt headers, and
link the wheel's NVPL/cuDSS dependencies. Pantheon's scheduler and block worker
remain upstream code.

`results/p9-pantheon-native-positive-20260809T103213Z` records the first native
positive control on the fixed 2g MIG instance:

- GPU: `NVIDIA Thor MIG 2g.0gb`, 12 SMs
- relaxed-deadline job: blocks 0 and 1, full exit, configured accuracy 0.9
- tight-deadline job: block 0, early exit, configured accuracy 0.7
- CUDA GEMM gate: passed
- upstream runtime exit status: 0

Build and run with `scripts/build_pantheon_thor.sh` and
`scripts/run_p9_pantheon_native_positive_control.sh`. This is a scheduler
fidelity gate using a deterministic chunked CNN, not a numeric paper result.
Numeric comparison remains disabled until the authors' processed real models
or equivalently processed common DNNs pass output/accuracy validation under the
shared arrival traces.

## Published-model recovery

Pantheon is applicable to Thor; missing packaged weights are porting work, not
an exclusion criterion.  We audited the authors' complete Zenodo artifact
(`10.5281/zenodo.11094058`, 419,394,252 bytes, 934 ZIP entries).  It contains
the offline trainer, graph partitioner, early-exit constructor, profiler, and
online runtime, but no `.pt`, `.pth`, or dataset payload.  The external
SharePoint URL named by the artifact README currently returns HTTP 404.

The numeric port therefore regenerates a reproducible CIFAR-10/ResNet50 model
with the authors' `pretrain.py`, constructs and trains every early exit with
the authors' EEN path, exports the resulting TorchScript blocks, and profiles
their latency and held-out accuracy on the fixed Thor 2g instance.  A row is
admissible only if the unsplit/full-exit prediction and accuracy match the
trained reference and the runtime reports the selected exit for every request.
Random exits, configured accuracy constants, and a stream-priority-only proxy
remain functional tests and cannot enter the numeric comparison.

The formal model-recovery artifact is
`results/p9-pantheon-cifar10-resnet50-formal-20260811/`.  It uses the pinned
upstream commit and the complete 100/100 epoch training contract, reaches
`0.9329` held-out accuracy at the final exit, and records
`full_output_max_abs_error=0.0` after EEN construction and serialization.  The
artifact is a model-recovery gate only: Pantheon's upstream model consumes
CIFAR-10 `32x32` images, whereas the current QUIET application contract is the
labelled ImageNette ResNet-50 split edge.

## Current ImageNette common-workload gate

The current contract is now exercised by the pinned online runtime with the
same ResNet-50 ImageNette split edge used by QUIET.  The adapter regenerates
CUDA TorchScript block and branch modules from the current backbone/head ONNX
artifacts, then binds the Pantheon workload protobuf to the operational
arrival trace and the current fractional deadline lock.  Pantheon's integer-
microsecond interface uses an explicit floor at the adapter boundary; the
sub-microsecond quantization is recorded in the verification artifact.

`results/p9-pantheon-resnet50-imagenette-gate100-r01-20260811/verification.json`
is the current gate.  It verifies the pinned source checkout, runtime binary,
generated modules, input and output traces, post-completion logits, labels,
and background pressure.  Reference and Pantheon accuracy are both
`0.8333333333333334` with zero delta; the observed Pantheon p99 is `4133 us`
with 2 of 90 measured requests missing the shared `2224.448116 us` lock
(effective Pantheon deadline `2224 us`).  This is a faithful published
comparator gate and is eligible for numeric DMR-goodput comparison; it is
not a claim that Pantheon meets the QUIET deadline.
