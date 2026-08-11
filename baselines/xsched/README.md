# XSched Thor port

This directory contains the source-pinned Thor adapter for XSched, the OSDI
2025 preemptive XPU scheduler. It is a published competitor, not a QUIET policy
or a renamed local baseline.

## Upstream and fidelity

- Repository: `https://github.com/XpuOS/xsched-artifacts`
- XSched submodule commit: `bd494cb7a72958cd11900243a0798df00d856c6e`
- Required live path: CUDA driver shim, XQueue, HPF scheduler, and the selected
  CUDA scheduling level.
- Patch: `patches/thor-cuda13-tensorrt.patch`

The patch preserves XSched's scheduler. It makes `cuMemAllocAsync` submission
blocking until the driver has populated the caller-owned device pointer, as
required by the CUDA API. With `XSCHED_TRT_USER_STREAM_ONLY=ON`, TensorRT's
allocator and setup streams keep native CUDA ordering while explicitly
prioritized inference streams become XQueues. This scopes XSched to the
measured scheduling unit instead of reordering engine initialization.

## Verified gate

On 2026-08-09, the patched upstream built natively on Thor/aarch64 with CUDA
13. A real TensorRT ResNet10 engine ran on the fixed 2g MIG instance for two
warm-ups and five measured requests through an HPF XQueue:

- GPU: `NVIDIA Thor MIG 2g.0gb`, 12 SMs
- requests: 5/5 completed
- p99 release-to-completion: 0.485 ms
- XQueue creation observed: yes
- CUDA errors: 0

This single-client result is a functional gate, not a performance result.

The native two-client positive control is also complete. In
`results/p9-xsched-native-positive-20260809T101042Z`, two real TensorRT
processes shared the same 2g MIG instance through XSched's global HPF server:

- BE pressure requests completed: 465
- HP requests completed: 100
- observed BE suspend transitions: 4
- observed BE resume transitions: 4
- HP/BE measurement overlap: 2.784 s
- BE measurement continued after the HP measurement ended
- CUDA errors: 0

The runner is `scripts/run_p9_xsched_native_positive_control.sh`; its verifier
rejects different GPUs, non-overlapping clients, missing HPF suspend/resume
actions, CUDA errors, or a BE interval that ends with the HP interval. Numeric
comparison remains disabled. A paper result requires the common workload,
arrival trace, output-correctness gate, and repeated measurement protocol.

## Dependent numeric smoke

The common-workload adapter adds only two mechanical compatibility changes to
the pinned runtime: defer the global agent until the first post-fork XQueue
event, and discard an already-derived operation after its client has closed.
The HPF policy, CUDA command queues, and suspend/resume mechanism are unchanged.
The patch is `patches/thor-cuda13-tensorrt.patch` and the reproducible runner is
`scripts/run_p9_xsched_dependent_smoke.sh`.

`results/p9-xsched-dependent-whisper-windowed-1000r-250rps-20260809T114154Z`
is the replay-verified numeric smoke. It creates three measured XQueues, records
four BE suspend and three resume transitions, and checksum-verifies all 1,000
2.304-MB Whisper-to-projection requests. At the independently frozen
1,703.187-us deadline, all 1,000 requests miss and p99 is 2,518.746 us. During
the 7.647-s critical window, BE arrival rate is 250.030 requests/s and completion
goodput is 160.322 requests/s. This is exploratory evidence, not a formal row.

The small common-payload runner is
`scripts/run_p9_xsched_resnet_control_smoke.sh`. It executes the TensorRT
ResNet10 `Layer7_cov` to control-MLP edge and preserves a separate checksum
trace. The verified artifact
`results/p9-xsched-resnet-control-100r-250rps-20260809T141021Z` requires all
three native XQueues and observed BE suspend/resume transitions. At the
exploratory 760-us wall deadline it records 100/100 misses and a 1,357.32-us
p99; the 117.94-ms critical window contains three completed background
requests. These are same-workload smoke measurements, not repeated paper
statistics.

The same runner also accepts the real learned ResNet10 head, rather than only
the generated control MLP:

```bash
WORKLOAD=resnet-detection-head \
CONSUMER_ENGINE="$PWD/results/p9-real-resnet-head-artifacts-20260810/resnet10-detection-head.engine" \
CONSUMER_INPUT_TENSOR=Layer6_relu_Y \
DEADLINE_LOCK="$PWD/results/<resnet-head-wall-lock>/deadline-lock.json" \
./scripts/run_p9_xsched_dependent_smoke.sh
```

The verifier binds the learned-head tensor shape, payload size, external-engine
SHA, production-wall deadline, and post-completion output trace. This path is
exploratory until its independent deadline lock and task-accuracy manifest are
available; it must not be presented as a ranked SOTA result before then.

## Labelled ResNet-50/ImageNette gate

The current native run also consumes the promoted ImageNette classification
contract used by QUIET.  The 90 measured requests use the same input trace,
operational arrival trace, 1g-producer/2g-consumer placement, TensorRT engines,
and external labels as the QUIET gate.  The reference and XSched candidate both
score `0.8333333333`, with zero accuracy delta and post-completion output
traces.  The evidence is
`results/p9-xsched-resnet50-imagenette-gate100-r03-20260811/`.

This is an application-correctness and failure-mode control, not a promoted
numeric row: the XSched run records `90/90` deadline misses at the frozen
`5722.576134 us` deadline (`DMR=1.0`, p99 `8169.06303 us`) and sets
`formal_claim_allowed=false`.  The machine-readable comparator record is in
`docs/p9-comparator-manifest.json`.
