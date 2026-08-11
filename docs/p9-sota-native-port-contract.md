# P9 native SOTA port contract

The machine-readable row and numeric-eligibility contract is frozen in
[`p9-comparator-manifest.json`](p9-comparator-manifest.json). This document
defines the required fidelity gates; it does not promote a functional gate to
numeric evidence.

`analysis/compare_sota.py` is the final promotion gate. A comparator or QUIET
row is numeric only when the summary carries the same production-wall latency,
correctness, and common-workload contracts, plus a formal evidence bundle:
`formal=true`, `thermal_normalized=true`, `ranking_allowed=true`, at least 14
independent-session/run units with paired Williams analysis, valid deadline and
thermal lock digests, and a one-sided 95% Clopper--Pearson DMR upper bound at or
below the frozen target. Exploratory runs, even with zero observed misses, are
reported descriptively and cannot enter a ranking or goodput ratio.

The paper comparison has exactly one proposed system name: **QUIET**. NVIDIA
MIG and NVIDIA MPS are vendor mechanisms. The three published competitors are
Orion (EuroSys 2024), XSched (OSDI 2025), and Pantheon (MobiSys 2024). These
are top-tier systems comparators, not placeholders selected for convenient
compatibility. A row may use one of those names only when its pinned source
runtime executes every measured request.

## Required competitors

| Row | Pinned implementation | Mechanism that must remain live on Thor | Functional positive control |
|---|---|---|---|
| Orion | `eth-easl/orion` | CUDA operation capture, per-operation compute/memory profiles, software queues, interference-aware HP/BE scheduler | Two independent TensorRT clients overlap and Orion changes at least one scheduling decision from FIFO |
| XSched | `XpuOS/xsched-artifacts` | CUDA driver shim, scheduling-unit queues, configured policy, interrupt/suspend/resume path | A long BE CUDA unit is preempted by an HP unit and resumes with correct output |
| Pantheon | `PantheonInfer/Pantheon` | online runtime queue, two-tier stream priorities, offline graph chunks, deadline-driven preemption | Two real-time chunked DNNs preempt each other on Jetson and preserve output/accuracy |

XSched and Pantheon native positive controls pass on the fixed Thor setup.
Orion's local capture path is useful functional evidence, but its native
upstream differential gate is still pending: a matching canonical decision
trace must be produced by the pinned upstream scheduler and the Thor port.
Until that gate passes, Orion remains
`numeric_comparison_allowed=false`. Pantheon also remains functional-only
until its common-workload accuracy adapter is complete. No local scheduler
reimplementation is promoted under a published competitor name.

The adapter contract is executable rather than declarative:
`analysis/verify_pantheon_common_workload.py` replays canonical reference and
Pantheon JSONL traces, requiring identical request IDs, arrivals, input hashes,
and labels. It also checks the explicit accuracy tolerance and preserves each
selected exit/block sequence. A native positive control or a hand-written
`accuracy_equivalent=true` field cannot satisfy this gate.

The verifier also requires `--upstream-source` and recomputes its SHA256 from
the pinned Pantheon source bytes. The resulting gate records the resolved
source path, digest, and `upstream_source_verified=true`; a caller-provided
digest that does not match the file is rejected. Promotion additionally
requires the source file to be tracked in a Git checkout whose `HEAD` equals
the pinned Pantheon commit, with the checkout root, commit, and relative path
recorded in the gate. `--allow-unpinned-source` is reserved for local
non-promoting fixtures. This source binding is required before
`analysis/compare_sota.py` can promote a Pantheon adapter to a numeric row. It
also requires `--runtime-binary`; the verifier recomputes and records that
binary's SHA256, and `analysis/compare_sota.py` re-hashes the bound file before
promotion. Thus the common-workload accuracy gate binds the current Pantheon
runtime binary as well as the source, training artifact, arrivals, and raw
traces.
Promotion still requires `--training-result` from the formal
CIFAR-10/ResNet50 training contract, including source, dataset, and exported
module hashes, zero full-model reconstruction error, and a passed held-out
accuracy gate. The current development artifact fails these fields and is
therefore intentionally not comparable.

The Orion differential verifier additionally requires
`--reference-source`, a tracked file in a Git checkout whose `HEAD` is exactly
the pinned upstream commit. It recomputes the source SHA256 from those bytes,
records the checkout root, commit, tracked relative path, and digest in the
gate, and rejects a caller-supplied digest that does not match. A local fixture
may be emitted only with the explicit non-promoting
`--allow-unpinned-source` flag; that output always has
`numeric_comparison_allowed=false`. This prevents a canonical trace from being
relabeled as upstream evidence after the source tree changes.
`analysis/compare_sota.py` requires the bound source provenance before allowing
an Orion row to enter a numeric table.
The differential gate also requires the current upstream runtime binary and
re-hashes it; the trace sidecar must name the same binary path and digest.
This prevents a trace from being attributed to a different runtime build after
the upstream source checkout changes.

The current replay-verified Orion smoke is
`results/p9-orion-dependent-whisper-windowed-1000r-250rps-20260809T114245Z`.
It runs 1,000 Whisper requests with the 2.304-MB coherent edge, a DistilBERT
best-effort stream offered at 250 requests/s, and the independently frozen
1,703.187-us validation-excluded deadline. Orion sustains 249.972 BE requests/s
but records 174 deadline misses (DMR 17.4%) and a 1,898.567-us p99. The raw
verifier binds the CSV, event trace, profiles, binary, and deadline lock. This
is a numeric smoke, not a counterbalanced formal result.

The matching XSched smoke is
`results/p9-xsched-dependent-whisper-windowed-1000r-250rps-20260809T114154Z`.
Its upstream HPF server owns the BE, producer, and consumer XQueues and records
four BE suspends and three resumes. Nevertheless, it records 1,000/1,000 misses
and a 2,518.746-us p99. The critical-window BE arrival and completion rates are
250.030 and 160.322 requests/s. The verifier recomputes those rates from the
absolute critical window and per-request BE completion trace. This is also a
numeric smoke, not a formal result.

For the real learned application path, XSched uses the same
`resnet10-backbone-to-learned-detection-head` workload as QUIET and NVIDIA
MPS, with producer tensor `Layer6_relu_Y`, payload size 1,884,160 bytes, and
an externally serialized detection-head engine. The dedicated runner is
`scripts/run_p9_xsched_dependent_smoke.sh` with
`WORKLOAD=resnet-detection-head`; it is eligible for a common-workload row
only after a current-binary deadline lock and external accuracy manifest are
bound. Attempts with stale locks remain invalidated rather than being reported
as XSched performance.

The Thor adaptation is an implementation task, not an applicability waiver or
an exclusion rule.
It may add `sm_110` support, wrap `cuLaunchKernelEx`, replace model loaders with
checksum-equivalent TensorRT stages, and bind processes to the fixed 2g+1g
layout. It may not replace a competitor's scheduler with a fixed MPS
percentage, process stop/resume, or QUIET's stage policy. A port failure must
first be diagnosed and repaired. If the published mechanism fundamentally
requires a capability absent from stock Thor, the paper reports that boundary
and replaces the row with another top-tier system whose original mechanism can
execute. It never creates a locally approximated row carrying the published
name.

BLESS (EuroSys 2025) and Mudi (EuroSys 2025) are mandatory mechanism-level
comparisons. No public executable artifact was found for either system as of
2026-08-09. A future implementation may be reported as a clearly labelled
paper-faithful reproduction after its policy equations and positive controls
are independently validated; until then it cannot silently replace an original
runtime or produce a fabricated numeric row.

## Evaluation rows

The final common-workload table is:

1. NVIDIA MIG isolation.
2. NVIDIA MPS spatial sharing.
3. Orion (Thor port).
4. XSched (Thor port).
5. Pantheon (Thor port).
6. QUIET.

BOER and ParvaGPU are not headline competitors: their published contribution is
offline provisioning for independent services, whereas this experiment needs
an online runtime to execute every dependent request. They may be used only as
structural provisioning controls. GSLICE, gpulet, quota-only provisioning,
partition-only planning, and full-DAG quiescence are ablations unless their
complete original runtime path is executed.

REEF and GPreempt are fallback literature candidates, not local emulations.
REEF's released reset path is tied to its ROCm runtime, while GPreempt changes
the GPU kernel/driver path. If a future stock-Thor implementation preserves
their original preemption mechanism, it can replace a failed executable row;
until then it cannot be reported as a measured REEF or GPreempt result.

## Common workload and acceptance

Every row receives the same arrival sequence, frozen deadline, TensorRT engine
hashes, the measured dependency payload (including the 14,720-byte ResNet10
Layer7 covariance control gate and the 2.304-MB Whisper activation stress),
background load, MIG layout, and request-level input/output checks. Report
critical DMR and p99/p99.9,
background goodput, GPU duty, scheduler overhead, and failed admission. A SOTA
row is numeric only after its positive control passes; build incompatibility is
reported separately and never converted into a local policy bearing that
system's name.

Primary sources:

- Orion: https://doi.org/10.1145/3627703.3629578
- XSched: https://www.usenix.org/conference/osdi25/presentation/shen-weihang
- Pantheon: https://pantheoninfer.github.io/
- BLESS: https://doi.org/10.1145/3689031.3696070
- Mudi: https://doi.org/10.1145/3689031.3696074
