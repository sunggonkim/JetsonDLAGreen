# P9 SOTA evaluation plan

## Public names

The proposed system is named **QUIET**. Internal policy identifiers are never
paper systems. Every competitor is either its published name plus `(Thor
port)`, or its published name plus `(Thor reimplementation)` when no public
artifact exists.

## Fixed comparison set

| Class | Paper label | Mechanism under test |
|---|---|---|
| Vendor baseline | NVIDIA MIG | Fixed 2g+1g physical isolation |
| Vendor baseline | NVIDIA MPS | Fixed 1g-producer/2g-consumer placement with MPS quota pressure |
| Primary published system | Orion (Thor port, EuroSys'24) | Profile-aware fine-grained kernel co-scheduling |
| Primary published system | XSched (Thor port, OSDI'25) | Preemptible command queues and HPF scheduling |
| Primary published system | Pantheon (Thor port, MobiSys'24) | Offline block/exit construction and online deadline scheduling |
| Proposed | QUIET | Dependency-aware placement, coherent cross-MIG handoff, and slack reservation |

gpulet remains a structural comparison until its upstream 3-D planner is
actually executed and reports a feasible common workload; it is not a ranked
numeric row in the fast matrix. BLESS remains a mandatory EuroSys'25 structural comparison, not a numeric row:
no public runtime was found, and the common q100 TensorRT plan fails inside
Myelin in the 2-SM context required by its quota-aware squad. Reporting the
executable q25 plan as the common q100 workload would change the experiment.
BOER remains the strongest supplemental MIG+MPS
configuration search, and ParvaGPU remains a planner feasibility control.
EdgeIso (IPDPS'20) is the Jetson-specific SoC contention comparison; its
DVFS/core-allocation boundary is evaluated separately from the GPU-runtime
ranking. Pantheon (MobiSys'24) remains the processed-DNN edge comparison and
must retain its offline model-variant/runtime semantics. Mudi is
evaluated only in an inference-plus-training experiment;
calling its cluster objective on an inference-only dependent DAG would alter
the published problem.

EdgeServing (arXiv 2026) is included as an edge-specific literature
comparator. Its time-division scheduler, batch-size selection, and early-exit
policy alter the executed model and accuracy contract. It remains outside the
numeric table until a Thor/TensorRT adapter preserves output accuracy and the
same request/deadline contract; a local scheduler must not be labeled
EdgeServing.

We also track *Performance Isolation for Inference Processes in Edge GPU
Systems* (arXiv:2601.07600) as a recent edge-specific characterization study.
It evaluates MPS, MIG, and Green Context across discrete and edge platforms;
it is useful for the CPU/SM/resource-isolation motivation axis, but it is not
an online black-box SLO governor and therefore remains literature-only until
a common Thor/TensorRT workload and accuracy contract is reproduced.

**Ev-Edge** (arXiv:2403.15717) is an additional edge-systems reference for
multi-task event-based vision execution and scheduling on Jetson Xavier. Its
event-camera runtime and model pipeline are outside the opaque TensorRT DAG
contract here, so it is a literature-only mechanism comparison rather than a
numeric row. This keeps the edge comparison broad without relabeling a local
runtime as the published system.

**GCAPS** (arXiv:2406.05221) is an additional real-time edge-GPU reference
that evaluates context-aware priority preemption on NVIDIA Jetson platforms.
It is useful for the preemption axis, but its driver/segment instrumentation
changes the execution boundary and is not equivalent to an opaque TensorRT
engine on fixed MIG instances. It is therefore literature-only unless a
native Thor adapter reproduces both its preemption semantics and application
accuracy contract.

"Different CUDA/PyTorch/GPU generation" is never an exclusion reason. A port
is admitted when it preserves the paper's scheduling state, decisions, and
actuation semantics and binds them to raw TensorRT work. If that fidelity gate
is unfinished, the row remains visibly pending rather than being replaced by a
local policy carrying the paper's name.

## Workloads

Every numeric system row must use the same fixed MIG UUIDs, TensorRT engine
hashes, request trace, producer-input trace bytes, deadline lock, correctness
oracle, and raw payload. `build_common_workload_contract.py` records the
producer trace SHA; the active learned-workload runner rejects a missing or
different trace before launching any comparator.

1. **Independent services:** ResNet10 and DistilBERT execute concurrently. This
   is the positive control for provisioning and spatial-sharing systems.
2. **Dependent small payload:** ResNet10 `Layer7_cov` output is transferred to
   a shape-compatible control MLP. This isolates dependency ordering.
3. **Dependent large payload:** Whisper Tiny produces a coherent 2.304-MiB
   tensor consumed by the 2g projection stage while DistilBERT supplies
   independent pressure. This exposes cross-MIG communication and memory-system
   contention.

The current frozen large-payload deadline is 1,701.921199 us, derived as 1.10 times the
pooled isolated p99 over 5 x 1,000 requests. The timed interval excludes the
offline correctness validation but includes producer, coherent handoff, and
consumer execution.

For the fast exploratory candidate sweep, invoke
`scripts/run_p9_common_sota_williams.py --active-only`. This executes
`NVIDIA MPS`, `XSched`, and `QUIET` in a balanced three-treatment sequence;
the output is not a numeric frontier until the manifest's application-
accuracy, thermal, and session gates pass. The historical six-treatment
runner is retained for replay but must not be used to rank structural
controls.

## Fidelity gates

BOER must preserve its six random initial probes, expected-improvement search
with `xi=0.2`, pruning, and stopping rule. The current Thor adapter applies EI
over an explicit integer candidate domain and is therefore a corrected
structural adapter, not yet a fully continuous-domain upstream execution;
regenerate its result before promoting it to a numeric comparator.
Orion must expose and profile real TensorRT driver launches. Its differential
gate must bind the canonical decision trace, pinned source SHA, common-workload
contract SHA, and an upstream-runtime provenance sidecar; matching two edited
JSONL files is not sufficient.
XSched must execute the upstream XQueue/HPF suspend-resume path over the same
TensorRT streams; a local process stop is not XSched. Its numeric verifier also
requires the byte-bound `common-workload.json` (arrival trace, dataset manifest,
tensor, payload, topology, and placement); a native smoke without that object is
functional evidence only.
BLESS must prove relative-progress squad selection, both published estimators,
restricted/unrestricted context execution, and per-request kernel order in raw
traces. A row failing its fidelity gate is reported as a functional or
structural result, never silently replaced with a local approximation.

The current BLESS gates pass the scheduler, native restricted/unrestricted
contexts, reused 2/4/6/8-SM TensorRT replicas, 9,400 traced real TensorRT
driver launches, and a checksum-verified 2-SM to 8-SM user-managed activation
handoff on Thor. The selected-only gate additionally reduces four replica
arrivals to one physical launch plus three shadow advances and switches at a
checksum-verified TensorRT boundary. It keeps
`numeric_comparison_allowed=false` until the BLESS scheduler drives the frozen,
independently profiled boundary set on the common workload. The boundary lock
and held-out replay are already preserved in the v5 fidelity bundle.
This prevents adjacent functional gates from being promoted into a performance
claim prematurely.

## Historical frozen-contract diagnostic campaign

> **Not a paper ranking.** The tables in this section are retained for raw
> provenance and failure localization only. They use superseded locks and/or
> incomplete comparator-fidelity gates. The only ranked table is generated from
> `docs/p9-comparator-manifest.json` and currently contains only the eligible
> `NVIDIA MPS` and `QUIET` rows under the production-wall contract. XSched is
> retained as a native executable candidate, but its real-application accuracy,
> thermal, and session gates are not yet passed.
>
> Formal regeneration must pass
> `analysis/summarize_p9_sota_frontier.py --require-output-traces`; a
> prediction-only accuracy JSON cannot promote a numeric row.

The current-binary six-sequence nonthermal campaign is
`results/p9-common-sota-whisper-current-nonthermal-formal-aggregate-6x1100-20260810/summary.json`.
Every row processes the same checksum-verified 2.304-MB Whisper activation at
250 offered background requests/s and the 1,701.921-us lock.

This artifact is a nonthermal diagnostic replay, not the current numeric SOTA
frontier. The comparator manifest is authoritative: Orion lacks the pinned
upstream differential gate and gpulet lacks a faithful feasible upstream
planner execution, so their numbers are retained only to localize failure
modes. They must not be ranked against QUIET until their gates pass.

| System | Numeric status | Misses / 6,600 | Pooled p99 (us) | Mean background goodput (req/s) |
|---|---|---:|---:|---:|
| NVIDIA MIG | capacity control, not ranked | 1,788 | 1,861.666 | 249.962 |
| NVIDIA MPS | executable baseline | 2,568 | 1,928.518 | 249.965 |
| Orion (Thor port) | diagnostic; differential gate pending | 3,859 | 2,786.190 | 249.886 |
| XSched (Thor port) | executable baseline | 6,600 | 2,517.895 | 144.557 |
| gpulet (Thor port) | diagnostic; faithful planner pending | 6,600 | 2,072.220 | 249.958 |
| QUIET | proposed system | 0 | 1,574.063 | 249.928 |

All 39,600 request rows are independently replayed from their raw traces.
QUIET alone certifies the 0.05% target: 0/6,600 gives a 0.04538% one-sided CP95
upper bound. This does not certify a final paper campaign: thermal
normalization, production-wall replacement, and the pending comparator gates
remain outstanding.

With the same QUIET plan and deadline left frozen, an exploratory background
load sweep delivers 499.933 req/s at 500 offered req/s and approximately 531
req/s at 600--750 offered req/s. The 375--750 points each have zero misses in
1,100 requests. One 125-req/s run has two outliers, but two immediate repeats
have none, so no monotone load frontier is claimed from this nonthermal sweep.
The raw-replayed characterization is
`results/p9-quiet-whisper-current-heldout-load-sweep-7x1100-20260810/summary.json`.

Raw stage replay localizes the failures. Producer-compute p99 is 1,487.007 us
for QUIET, versus 1,784.229 for MIG, 1,831.895 for MPS, 1,994.811 for gpulet,
2,374.851 for XSched, and 2,374.182 for Orion. Orion additionally inflates the
edge p99 to 634.551 us and consumer-compute p99 to 696.355
us. Thus the numeric rows fail for different reasons: static/spatial policies
exhaust producer slack, XSched's request-unaware queue priority cannot protect
the complete dependent response, and Orion's operation-level interleaving does
not preserve the end-to-end stage-DAG reservation.

The older 1,000-request rows below use the previous compatible lock and are
retained as development history rather than pooled with the current smoke.

`results/p9-published-sota-dependent-smoke-20260809T1353/summary.json` binds
every row below to deadline-lock SHA-256
`4b383e300d756f7da0987d0077bec2416c01588ca1a019b04c0e3e05b2b5ab48`.
These are implementation and hypothesis checks, not formal statistics.

| System | Requests | DMR | p99 (us) | Background goodput (req/s) |
|---|---:|---:|---:|---:|
| NVIDIA MIG | 1,000 | 29.1% | 1,882.39 | 249.97 |
| NVIDIA MPS | 1,000 | 38.0% | 1,927.32 | 249.95 |
| BOER (Thor port), minimum-p99 search point | 1,000 | 100% | 2,070.85 | -- |
| Orion (Thor port) | 1,000 | 57.4% | 2,850.08 | 250.09 |
| XSched (Thor port) | 1,000 | 100% | 2,518.75 | 160.32 |
| BLESS (Thor reimplementation) | -- | -- | -- | functional gate only |
| QUIET | 1,000 | 0% | 1,584.32 | 249.96 |

The smoke supports the intended stress hypothesis: systems that optimize
static capacity, same-instance kernel overlap, or generic queue preemption do
not reserve the full producer--handoff--consumer dependency path. Formal claims
still require counterbalanced repetitions and exact confidence bounds.

The current-lock provisioning evidence is now regenerated as well. BOER's
algorithm-preserving search records four hardware observations and returns no
feasible point; q90/r100 is best at 2.058-ms p99 with 100% DMR. ParvaGPU's
original segment configurator rejects the Whisper producer before placement.
Both retain successful independent-workload positive controls. The authoritative
join is `results/p9-current-whisper-structural-evidence-20260810/summary.json`.

## Intended-domain positive controls

`results/p9-published-sota-positive-controls-20260809T1353/summary.json`
separately establishes that the ports execute their published mechanisms:
BOER selects a feasible independent configuration, ParvaGPU serves its planned
independent 2g+1g allocation, Orion performs profile-aware complementary kernel
admissions, XSched suspends and resumes overlapping XQueues, and BLESS executes
ordered native kernel squads. These results are not cross-system rankings.
Together with the dependent table, they distinguish a structural workload
mismatch from a dead or bypassed implementation.

## Questions answered

1. Can each published system reproduce its intended independent-workload win
   on Thor?
2. Which mechanism first fails when a real dependency and payload cross the
   MIG boundary: admission, placement, ordering, communication, or residual
   shared-SoC contention?
3. At the same frozen deadline and correctness oracle, does QUIET recover SLO
   feasibility and more best-effort goodput than the strongest feasible
   published competitor?
