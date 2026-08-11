# P9 SOTA comparison contract

## Active headline contract

The public proposed-system name is **QUIET**. The active exploratory manifest
has three executable rows: **NVIDIA MPS** (vendor baseline), **XSched (Thor
port)** (native command-queue candidate), and **QUIET**. The numeric frontier
currently contains only NVIDIA MPS and QUIET; MIG is an
isolation oracle, not a best-effort capacity comparator. **Orion (Thor port)**
and **Pantheon (Thor port)** remain published-system functional rows until
their differential and common-workload accuracy gates pass. They must not be
turned into numeric rows by substituting a local policy. This boundary is
machine-checked by `docs/p9-comparator-manifest.json` and
`analysis/compare_sota.py`.

The numeric frontier is also blocked until QUIET has a passed, byte-bound
application-accuracy gate. Current production-wall frontier files are
exploratory session evidence, not a 0.05% DMR certification.

> **Superseded for headline SOTA claims.** The fixed-action GSLICE, gpulet, and
> managed-client Orion rows below are retained only as historical mechanism
> evidence. They must be presented as quota-only provisioning,
> partition-only planning, and full-DAG quiescence. The native published-system
> contract is `docs/p9-sota-native-port-contract.md`.

The historical campaigns in this file must not be regenerated into the current
paper table. The active paper-table contract is the exact-lock manifest
`docs/p9-comparator-manifest.json`: the only currently eligible numeric rows are
`NVIDIA MPS` and `QUIET`; XSched is a native, executable candidate whose
application-accuracy, thermal, and session gates are still pending. MIG is a
capacity control, while Orion and Pantheon remain functional-only until their
native fidelity gates pass. The four-session production-wall evidence is
`results/p9-common-production-wall-frontier-repeats-commonlock-4x-20260810.json`.

## Decision

BOER, ParvaGPU, Orion, and Pantheon remain implementation-fidelity and
compatibility evidence, but they are not headline performance rows yet. A
paper table cannot substitute `unsupported` or `no feasible configuration` for
a measured competitor. The active numeric fixed-2g+1g comparison therefore
uses only rows whose gates have passed:

1. **NVIDIA MPS.** Fixed topology, explicit SM-quota sharing, and the same
   production-wall workload and deadline as QUIET.
2. **XSched (Thor port).** Pinned upstream XQueue/HPF artifact and native
   runtime verification. It remains exploratory until the same application
   accuracy and thermal/session gates are complete.
3. **QUIET.** The only proposed-system row; raw internal policy IDs are
   provenance fields and are never presented as additional schemes.

GSLICE, gpulet, BOER, and ParvaGPU remain appendix controls unless their
upstream algorithm and common-workload fidelity gates pass. Orion and Pantheon
keep their published names only in the functional comparison table, not as
executable local replacements.

The measured primitive rows remain NVIDIA MIG and NVIDIA MPS. The proposed
system row is only **QUIET**. Process-stop remains an ablation, not a system.

## Common workload

- Producer: TensorRT Whisper Tiny encoder on the 1g instance.
- Edge: the complete `[1,1500,384]` FP32 tensor, 2,304,000 bytes, with
  request-level checksum validation.
- Consumer: TensorRT projection MLP on the 2g instance.
- Pressure: TensorRT DistilBERT SST-2 offered at 500 requests/s.
- Transport: registered coherent system-memory binding across MIG instances.
- Deadline: independently frozen 1,701.316397 us, SHA-256
  `d3da4431a4f047ee133649a51dbd8ccc8716318fccfe86dc4d6ae0e34d1d8fc0`.
- Primary metrics: raw-trace DMR and one-sided exact CP95 upper bound;
  background completed requests/s is secondary.

Profile requests and evaluation requests must be disjoint. Every adapter must
record the upstream revision, algorithm inputs, selected action, common
deadline-lock SHA, engine hashes, raw traces, and the final executed action.

## GSLICE first result

The first faithful numeric port is complete. Starting from q50/q50, the
controller observes a 3.292-ms pipeline p99 and moves to q90/q10. A second
tuning round remains at q90/q10. In the disjoint 1,500-request evaluation,
GSLICE preserves 499.94 background requests/s but misses 1,500/1,500 critical
deadlines with a 2.092-ms p99. Evidence is in
`results/p9-gslice-whisper-dependent-500rps-20260809-v2/result.json`.

This is the intended failure experiment: GSLICE can right-size independent
inference functions, but its per-function latency/throughput feedback cannot
place a dependent producer's pressure into the consumer-stage slack. The
result is numeric and executable; it is not an incompatibility placeholder.

## gpulet first result

The pinned gpulet policy port profiles all five representable complementary
MPS partitions. None is schedulable at the frozen 1.701-ms deadline: even the
largest q90/q10 critical/background split has a 2.068-ms profile p99. Its
disjoint 1,500-request diagnostic execution misses 1,500/1,500 deadlines, has
a 2.090-ms p99, and sustains 499.96 background requests/s. Evidence is in
`results/p9-gpulet-whisper-dependent-500rps-20260809-v1/result.json`. The row
must report `spatial_schedule_feasible=false`; it must not present q90/q10 as
a feasible gpulet schedule.

## Orion first result

The conservative TensorRT managed-client port protects the complete
high-priority DAG request because the opaque engine does not expose Orion's
per-kernel profiles. Its first 1,500-request execution has zero misses and a
1.566-ms p99, but background goodput falls from the 500 requests/s offer to
151.3 requests/s. Evidence is in
`results/p9-orion-managed-whisper-dependent-500rps-20260809-v1/orion-result.json`.
This is the expected coarse-operation limitation. QUIET instead protects only
the producer stage that shares the 1g residual resource and resumes background
work during the dependent consumer stage; the existing balanced 6,000-request
smoke sustains 499.95 requests/s with zero misses.

## Remaining execution order

1. Freeze the three active rows and run a Williams-balanced comparison
   containing NVIDIA MPS, XSched (Thor port), and QUIET on the same application
   workload, load frontier, thermal lock, and accuracy gate.
2. Keep MIG as a separate hardware-isolation oracle and move BOER, ParvaGPU,
   GSLICE, gpulet, Orion, and Pantheon to capability/fidelity tables until
   their gates pass. Regenerate paper figures only from the active manifest.

## Edge-specific literature row

Miriam (ACM SenSys 2023) is the closest edge-GPU system. It generates elastic
CUDA kernels and coordinates mixed-critical DNNs on Jetson-class GPUs. It is
included in the mechanism table, but not relabeled as a TensorRT result:
replacing the common TensorRT engines with Miriam-generated kernels would
change the workload implementation. The public artifact and its kernel-level
design remain a required discussion point when interpreting the numeric rows.

## Integrated execution-path smoke

The first common six-system smoke completed with 300 requests per row at the
same 500 requests/s background offer. NVIDIA MIG missed 290, NVIDIA MPS 298,
GSLICE 300, and gpulet 300 requests. Orion and QUIET had zero misses. QUIET
sustained 499.83 background requests/s, while the full-DAG Orion policy port
sustained 242.99 requests/s in this short run. The longer independent Orion
execution sustains 151.3 requests/s, so Orion goodput remains preliminary
until equal-length balanced repetitions remove startup/window effects. The
machine-readable smoke is
`results/p9-six-system-numeric-smoke-300r-500rps-20260809-v1/summary.json`.

The six Williams sequences also completed at 300 requests per system. Raw
replay pools 1,800 requests per row: MIG misses 1,670, MPS 1,789, GSLICE and
gpulet 1,800, while Orion and QUIET have zero misses. Orion averages 242.31
background requests/s and QUIET 499.78 requests/s. The 2.06x ratio is only
descriptive because the zero-miss CP95 upper bound at N=1,800 is 0.166%, above
the 0.05% target. Evidence is in
`results/p9-six-system-williams-6x300-500rps-20260809-v1/aggregate.json`.

The balanced performance campaign then ran all six sequences with 1,500
requests per system, pooling 9,000 requests per row. QUIET and the Orion
managed-client port both record zero misses and an exact CP95 upper bound of
0.0333%, below the 0.05% target. QUIET sustains 499.94 background requests/s;
Orion sustains 153.57 requests/s. Across the six paired Williams runs, the
QUIET/Orion goodput ratio is 3.256x with a 95% Student-t interval of
[3.241x, 3.270x]. MIG misses 8,424/9,000, MPS 8,950/9,000, and GSLICE and gpulet miss
9,000/9,000. The campaign deliberately excludes thermal normalization and is
therefore labeled balanced performance evidence rather than thermal formal
evidence. Raw-replayed output is in
`results/p9-six-system-williams-6x1500-500rps-performance-20260809-v1/aggregate.json`.

## Structural-limit evidence

The failure rows have positive controls. BOER selects q90/q10 for two
independent services, records a 1.481-ms worst p99, and serves 499.78 and
499.63 requests/s. On the frozen dependent DAG, its best measured point is
2.080 ms and no point is feasible. ParvaGPU likewise serves its independent
ResNet and DistilBERT services at 499.89 and 499.76 requests/s with 0.434- and
0.969-ms p99, but rejects the dependent Whisper producer under its published
segment-admission rule. These controls show an abstraction mismatch, rather
than a broken port: both systems optimize independent clients or segments and
do not represent stage precedence or reclaimable producer-stage slack.

MIG does not share device memory across instances. The dependent pipeline
instead registers Thor coherent system memory into both CUDA contexts. For the
2,304,000-byte edge, cross-MIG registered transport has a 15.25-us edge p99,
versus 14.22 us in the same-instance control. Forced pinned and pageable
bounces increase end-to-end p99 from 1549.73 us to 1616.03 and 1617.35 us.
Thus device-memory isolation is real, but it need not imply a large copy cost
on this coherent edge platform. The dominant experimental failure is residual
shared-SoC interference combined with dependency-unaware scheduling.

The machine-readable, hash-bound join of these inputs is
`results/p9-structural-limit-evidence-20260809/summary.json`.

A four-sequence Williams transport ablation strengthens the transport result
to 2,000 requests per treatment. Cross-MIG registered transport records a
14.06-us edge p99. Same-instance MPS records 19.17 us and is 4.50 us slower in
the sequence-paired comparison (95% t interval [0.25, 8.74] us). Forced pinned
and pageable bounces add 98.25 us [92.71, 103.79] and 100.40 us
[96.18, 104.61], respectively. This is balanced performance evidence without
thermal normalization; its raw-replayed aggregate is
`results/p9-transport-williams-4x500-20260809/aggregate.json`.

## Slack-plan execution

The stage-DAG planner now consumes the frozen deadline lock and a measured
Whisper profile, rather than a manually copied deadline. With a predeclared
1000-us release lookahead and a 50-us reservation margin, it reserves
1674.82 us and leaves 26.50 us below the 1701.316-us deadline. The execution
runner validates the plan hash, q100/q100 placement, registered 2.304-MB edge,
producer-only protection scope, and nonnegative slack before launching.

A separate 1,500-request held-out smoke records zero misses, a 1584.51-us
pipeline p99, 499.93 background requests/s, and a 997.96-us gate p99 below the
frozen 1000-us lookahead. The plan is
`results/p9-quiet-whisper-slack-plan-20260809/plan.json`; held-out evidence is
`results/p9-quiet-whisper-slack-plan-heldout-1500r-500rps-20260809/summary.json`.
This remains performance-smoke evidence, not a confidence-qualified formal
result.
