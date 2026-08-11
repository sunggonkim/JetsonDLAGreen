# BLESS Thor reimplementation

Reference: Shulai Zhang et al., "Improving GPU Sharing Performance through
Adaptive Bubbleless Spatial-Temporal Sharing," EuroSys 2025,
DOI `10.1145/3689031.3696070`.

No public BLESS runtime artifact was found on the paper, author, or ACM pages
as of 2026-08-09. This directory is therefore an explicitly labeled
reimplementation, not an upstream port. It must never reuse a local fixed-gate
policy under the BLESS name.

The reimplementation contract follows Sections 4.2--4.5 of the paper:

1. profile every intercepted kernel at the available SM-affinity levels;
2. form squads of at most six kernels by repeatedly selecting the request with
   the smallest relative progress `Pr / Pe`;
3. enumerate strict spatial configurations and the unrestricted case;
4. select the minimum predicted squad duration using the paper's
   interference-free and workload-equivalence estimators; and
5. execute the first half of a squad in restricted contexts and the remainder
   in unrestricted contexts, preserving per-request kernel order.

`scheduler.py` fixes the paper-level decisions independently of CUDA.
`native_squad_smoke.cu` executes those decisions on Thor with real 2/4/6/8-SM
execution-affinity contexts, a six-kernel squad bound, restricted and
unrestricted phases, mapped activation state, per-request ordering, and a
checksum oracle. `verify_native_squad_smoke.py` replays the raw squad trace.
The preserved hardware artifact is
`results/p9-bless-native-squad-20260809T123619Z`.

This is a functional fidelity gate, not yet a TensorRT performance result.
`trt_context_replica_smoke.cpp` now also pre-creates and reuses 2/4/6/8-SM
contexts in one process, deserializes the same q25 TensorRT plan in every
context, and executes two complete rounds. The traced artifact
`results/p9-bless-trt-context-replica-traced-v2-20260809T1328Z` contains 9,400
successful driver launches, including 8,000 `cuLaunchKernelEx` calls, and
binds 940 identical operation signatures per measured affinity-context
replica. A second gate supplies TensorRT's 1,378,304-byte user-managed
activation block, copies
it from a restricted 2-SM context to the unrestricted 8-SM replica with
`cuMemcpyPeer`, and reproduces the exact output checksum after the handoff
(`results/p9-bless-trt-activation-replica-20260809T1335Z`).

The selected-only gate now groups each logical TensorRT launch across all four
replicas, executes one physical launch, shadow-advances the other three, and
performs a restricted-to-unrestricted activation handoff. Fixed 2-SM and 8-SM
executions reproduce the reference output. An exhaustive single-switch probe
shows that only TensorRT-safe operation boundaries preserve correctness; the
midpoint operation 23 is one such boundary. The exhaustive calibration is
frozen in `results/p9-bless-trt-safe-boundary-lock-20260809T1349`, and the
separate held-out gate is
`results/p9-bless-trt-squad-replica-heldout-20260809T1351Z`. The joined
evidence is `results/p9-bless-tensorrt-fidelity-v5-20260809T1352/summary.json`.

The same procedure was repeated for the common ResNet producer.  The q25 plan
has 18 logical launches and safe switch operations 0, 6, 9, 15, and 18.  The
independently selected midpoint operation 9 passes a held-out run with 18
physical launches, 54 shadow launches, one activation copy, and an exact output
checksum.  Synchronized 2/4/6/8-SM profiles for both the 18-launch ResNet and
47-launch DistilBERT plans feed the paper's relative-progress and configuration
estimators.  The exact common q100 ResNet plan fails inside TensorRT Myelin in
the required 2-SM execution-affinity context.  Substituting the executable q25
plan would change the workload used by the other rows, so BLESS is reported as
a measured compatibility boundary and not a numeric performance row.  A
full-inference MPS quota sweep or a local process gate is not reported as
BLESS.

Thor has 12 SMs in the 2g instance and 8 SMs in the 1g instance. Unlike the
A100 evaluation's 18 partitions, the runtime enumerates only affinity levels
that CUDA accepts on the target MIG instance and profiles every such level on
Thor. This is a hardware-domain substitution; the scheduling and estimator
algorithms are unchanged.
