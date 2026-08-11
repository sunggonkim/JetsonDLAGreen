# Orion Thor native port

Pinned upstream: `eth-easl/orion` at
`20f9469764fb96d94ce23a8e70615196e9ce4ba1`.

On Thor, both official shared libraries build when the CUDA 13.0 SBSA include
and library directories are supplied explicitly. The stock `compile.sh` points
at `/usr/local/cuda`, whose selected 13.2 installation lacks `cublas.h`.

Orion's original frontend creates in-process client threads and shared software
operation queues. The Thor port preserves that ownership model: the exact
TensorRT benchmark core is linked into `orion-trt-native-smoke`, and two real
TensorRT clients submit captured driver operations to a shared Orion scheduler.
The old request-level `run_thor.py` policy is retained only as an ablation and
must not be presented as Orion.

The stronger Nsight API probe confirmed that this was not only an initialization
problem. TensorRT resolves CUDA driver entry points dynamically and launches
through `cuLaunchKernelEx`, which pinned Orion did not wrap. The local
`driver_capture` substrate now interposes both direct driver symbols and launch
pointers returned through `dlsym`/`cuGetProcAddress`. A fresh six-request
ResNet10 positive control captured 126 successful `cuLaunchKernelEx` calls and
18 function handles without changing benchmark output. Evidence is in
`results/p9-orion-driver-capture-resnet-positive-20260809-v3/compatibility.json`.
The driver capture now feeds a real per-client software queue. Three independent
hardware positive controls each executed 180 successful TensorRT
`cuLaunchKernelEx` operations and made 13--14 high-priority scheduling decisions
that differed from FIFO. Both clients completed four requests without CUDA
failure. Evidence is in:

- `results/p9-orion-native-positive-deterministic-r1-20260809T093809Z`
- `results/p9-orion-native-positive-deterministic-r2-20260809T093809Z`
- `results/p9-orion-native-positive-deterministic-r3-20260809T093809Z`

This closes operation visibility, software-queue execution, and non-FIFO
scheduling. It is still a functional positive control, not a numeric Orion
comparison: the deterministic initial gate is disabled for evaluation, and
Thor-specific compute/memory interference profiles must replace it before a
performance row is enabled.

Thor-specific operation profiling is now implemented by `profile_thor.py`.
It executes each TensorRT engine in isolation and under independent compute and
memory pressure, measures every captured driver launch with CUDA events, and
derives Orion's operation class and occupancy-based SM demand. The current
strict scheduler profiles are:

- `results/p9-orion-whisper-operation-profile-20260809T110353Z`: 37 operations
  per inference; 33 compute, 3 memory, and 1 unclear.
- `results/p9-orion-distilbert-operation-profile-20260809T110353Z`: 46
  operations per inference; 27 compute, 14 memory, and 5 unclear.

The Thor scheduler strictly matches every runtime launch signature against the
profile, gives HP launches priority, and admits complementary BE launches under
the profiled SM and duration bounds. The upstream scheduler requires a
model-specific maximum aggregate BE duration. The reproducible runner derives
that value from the frozen isolated HP p99 (1,548.352 us), rather than silently
using the port's 1-us parser default. The replay-verified dependent numeric
smoke is
`results/p9-orion-dependent-whisper-faithful-1000r-20260809T1303`: 1,000
coherent Whisper-to-projection requests, 250.091 DistilBERT requests/s, DMR
57.4%, and validation-excluded p99 2,850.079 us at the frozen 1,703.187-us
deadline. `verification.json` independently replays the raw pipeline, scheduler
events, runtime TSV profiles, analysis JSON profiles, and run contract. The run
is explicitly exploratory; it is not a formal Orion headline.

The same probe was repeated on the actual Whisper producer used by the large
dependent DAG. Seven TensorRT executions issue 259 `cuLaunchKernelEx` calls
and zero Orion-interceptable CUDA-runtime/cuDNN/cuBLAS compute calls. The
Whisper-specific report and source-bound classification are in
`results/p9-orion-whisper-api-probe-20260809/compatibility.json`.

On 2026-08-10, the pinned upstream checkout was rebuilt with the SBSA CUDA
13.0 headers and libraries and then probed with the repository's TensorRT
benchmark. The libraries built successfully, but direct `LD_PRELOAD` execution
of the benchmark terminated with `SIGSEGV`; the probe classifies this as
`requires-orion-managed-client-integration`. The immutable probe record is
`results/p9-orion-upstream-sbsa-trt-probe-20260810.json`. This is an upstream
applicability boundary, not a license to replace Orion with a local scheduler;
the differential numeric gate remains closed until the official managed-client
path is connected.

## Labelled ResNet-50/ImageNette smoke

The managed-client path was also exercised on the current labelled
ResNet-50/ImageNette split. The 90 measured requests use the same input bytes,
operational arrival trace, 1g-producer/2g-consumer placement, deadline lock,
and labels as QUIET. Reference and Orion accuracy are both `0.8333333333`
with zero delta; the raw scheduler events, profile bundle, production-wall
CSV, and post-completion output trace are replay-verified by
`verify_resnet50_imagenette_smoke.py`.

Evidence is in
`results/p9-orion-resnet50-imagenette-gate100-r01-20260811/`. The run has
zero deadline misses and p99 `5537.08993 us` at the `5722.576134 us` lock, but
it remains `formal_claim_allowed=false`: the upstream differential-fidelity,
counterbalanced-session, and thermal gates are still open. This is evidence
of common-workload application correctness, not a promoted Orion frontier row.

The replayable current-run wrapper is
`scripts/run_p9_orion_resnet50_imagenette_smoke.sh`. With the default paths it
loads the frozen deadline/common-workload contracts, runs the managed client,
replays the scheduler/profile artifacts, and applies the labelled accuracy and
input/output-trace gates. Set `RESULT_DIR` to a new directory when collecting a
fresh run; never overwrite the archived gate above.
