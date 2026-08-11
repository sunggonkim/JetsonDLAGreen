# QUIET P9 performance evaluation

> **Authoritative SOTA contract (2026-08-10).** The comparison inventory is
> NVIDIA MIG, NVIDIA MPS, Orion (EuroSys'24), XSched (OSDI'25), Pantheon
> (MobiSys'24), and QUIET. The current numeric frontier is only NVIDIA MPS and
> QUIET; Orion, XSched, and Pantheon are named functional candidates until
> their gates pass. BLESS (EuroSys'25), gpulet (ATC'22), BOER, and ParvaGPU remain
> literature, structural, or provisioning controls until their original
> runtime/planner and common-workload gates pass. Pantheon is the designated
> edge-specific comparator; it is functional-only until its accuracy-equivalent
> common-workload adapter produces numeric evidence.
> BOER and ParvaGPU are structural provisioning controls, not substitute
> headline systems. A source tree that does not build unchanged on Thor is a
> porting task, not an applicability waiver: each numeric row must preserve the
> paper's scheduler and control boundary and run the common TensorRT workload.
> Older sections below describing gpulet as a headline row, BOER/ParvaGPU as
> headline rows, Orion as unsupported, or a functional gate as a numeric result
> are experiment history and must not be used for the final table. The final
> machine-readable presentation contract is `headline_systems` in the schema-5
> payload comparison; the legacy `systems` array is retained only for raw
> replay compatibility.

> **Current evidence warning.** The historical numeric tables later in this
> file are retained for provenance and mechanism debugging only. They must not
> be copied into the paper headline. The latest fast evidence is the
> production-wall smoke in `docs/p9-current-status.md`; it is exploratory,
> uses 20 requests per arm, and has no thermal or application-accuracy gate.
> The active numeric frontier currently contains `NVIDIA MPS` and `QUIET`.
> XSched (Thor port) is the next native candidate, but remains out of numeric
> ranking until its shared learned-workload accuracy, thermal, and session
> gates pass. Every ranked row requires the same workload, deadline lock,
> inline correctness, and session-level SLO qualification.

The active Williams runner currently binds the forward `1g`-producer/
`2g`-consumer placement because the native XSched verifier is forward-only.
Reverse-placement QUIET/MPS measurements remain placement characterization and
are rejected from the three-system numeric comparator until XSched has an
equivalent reverse-placement contract.

## Scope and claim boundary

QUIET is the only proposed system name. The P9 study fixes one Thor GPU in a
`2g+1g` MIG layout: deadline-critical ResNet50 runs on `2g`, while best-effort
audio and language inference use the residual `1g` instance and borrow idle
time on `2g`. Temperature telemetry is retained only as a passive safety log in
the performance campaign; it is not a controller input or a paper contribution.

The legacy multimodal scenarios use the same modality sequence and deadline:

- **Independent:** audio and language requests make progress independently.
- **Dependent (incomplete):** each audio completion currently releases one
  language inference through a pipe token, but no producer tensor is moved.
  These runs measure ordering overhead only and are invalid for a cross-MIG
  dependent-pipeline claim.

Internal policy identifiers remain in raw provenance. Public tables contain one
QUIET row, executable baselines, and explicitly named ablations.

## Current implementation

The legacy scenario runner still exports dependency wait/signal file
descriptors and therefore remains control-plane only. It must not supply a
dependent result. A separate paper-valid smoke now binds the actual ResNet10
`Layer7_cov` output to registered shared system memory and consumes it in a
TensorRT control-policy network on the other MIG instance. It verifies payload
and downstream-output checksums and reports producer-to-consumer latency. The
current 100-request gate preserves both timing and per-request checksum traces,
with zero failures, four distinct payloads, four distinct policy outputs, a
6.61-us edge p99, and a 707.58-us end-to-end p99. The hash-bound artifact is
`results/p9-resnet-layer7-control-mlp-100r-traced-v2-20260809T1420Z/`.
The scenario runner freezes the
deadline from the independent calibration and reuses it unchanged for the
dependent run. The analyzer rejects mismatched traces, deadlines, worker
dependency flags, placements, or execution contracts.

The same payload-valid ResNet pipeline now has a faithful Orion operation-level
smoke. A fresh three-mode profile classifies its 18 TensorRT operations as 14
compute-sensitive, three memory-sensitive, and one unclear. Orion executes
2,900 profiled decisions with 225 complementary admissions and 1,720 reorder
events, while all payload and policy-output checksums remain valid. At the
common 770.605-us deadline it misses 70/100 requests, records an 877.865-us
p99, and retains 224.70 background requests/s. The verified inputs are
`results/p9-orion-resnet10-operation-profile-20260809T1430Z/` and
`results/p9-orion-resnet-control-common-deadline-100r-250rps-20260809T143423Z/`.
The same ResNet payload contract now runs through XSched's native CUDA XQueue
backend rather than process signals. The verified smoke creates one DistilBERT
best-effort XQueue and two pipeline-stage XQueues, and replays four BE suspend
and three resume transitions. All 100 payload and downstream-output checks pass
with four distinct values each. At the common 770.605-us deadline, XSched
misses 100/100 requests, records a 1,374.42-us p99, and completes three
background requests in the 119.61-ms critical window (25.08 requests/s). This
is smoke evidence that command-queue preemption alone does not satisfy the
dependent DAG's end-to-end SLO; it is not a formal superiority claim. The raw
and verified artifact is
`results/p9-xsched-resnet-control-common-deadline-100r-250rps-20260809T143441Z/`.

The gpulet Thor port pins the public ATC'22 artifact commit and profiles all
five representable MPS partition pairs. Each pair preserves the same ResNet
pipeline, DistilBERT 250-RPS load, and frozen deadline; the final 100 requests
are disjoint from profiling. None of the five pairs is schedulable. The
paper's fallback diagnostic q90/q10 execution misses 100/100 requests with a
949.521-us p99 while retaining 249.95 background requests/s. The verifier
replays all scheduler decisions, raw request traces, checksums, and artifact
hashes in
`results/p9-gpulet-resnet-control-5x100-eval100-20260809T142909Z/`.

The historical payload-bound, plan-enforced primary smoke table is
`results/p9-dependent-payload-primary-sota-smoke-v11-plan-enforced-20260809T144501Z/summary.json`.
Its legacy order is NVIDIA MIG, NVIDIA MPS, Orion, XSched, gpulet, and QUIET;
this artifact predates the canonical Pantheon headline contract and is not a
current SOTA ranking.
At the common 770.605-us deadline the observed miss counts are 60, 60, 70, 100,
100, and 0 out of 100, respectively. QUIET's bound plan reserves 700.031 us
and retains 70.574 us of response slack. These are smoke results, not confidence-
bounded formal claims. BLESS remains functional-only until its relative-
progress kernel-squad scheduler drives this common workload. Pantheon remains
the required edge-specific functional gate until its accuracy-equivalent
numeric adapter is complete; BOER, ParvaGPU, and gpulet are structural controls.

The common-workload execution harness now has a complete six-treatment Williams
pilot. Each system occurs once in every position and every ordered predecessor
pair occurs once. The six 100-request sequences produce the following totals:

| System | Misses / 600 | Per-run p99 range (us) | Mean BE req/s |
|---|---:|---:|---:|
| NVIDIA MIG | 363 / 600 | 888.278--915.130 | 249.932 |
| NVIDIA MPS | 374 / 600 | 887.634--915.545 | 249.909 |
| Orion | 429 / 600 | 870.540--1,014.635 | 212.515 |
| XSched | 600 / 600 | 1,359.970--1,652.657 | 24.871 |
| gpulet | 599 / 600 | 932.916--967.817 | 249.901 |
| QUIET | 0 / 600 | 702.918--713.670 | 248.381 |

The hash-verified aggregate is
`results/p9-common-sota-williams-aggregate-6x100-20260809T151333Z/summary.json`.
This establishes repeatable mechanism behavior and removes ordinal/predecessor
order as an obvious explanation. It is still a pilot: it is not thermal
normalized, and QUIET's zero-miss one-sided CP95 upper bound is 0.498%, above
the predeclared 0.05% DMR target. Same-SLO goodput superiority is therefore not
claimed from this dataset.

The follow-on formal-size nonthermal campaign increases every sequence to
1,100 requests per system. Raw replay over 6,600 requests per row yields:

| System | Misses | Pooled p99 / p99.9 (us) | CP95 DMR upper |
|---|---:|---:|---:|
| NVIDIA MIG | 4,012 | 910.892 / 928.058 | 61.7803% |
| NVIDIA MPS | 4,004 | 914.298 / 936.663 | 61.6597% |
| Orion | 4,456 | 1,047.563 / 1,390.144 | 68.4651% |
| XSched | 6,600 | 1,367.983 / 1,701.457 | 100% |
| gpulet | 6,596 | 953.180 / 974.411 | 99.9793% |
| QUIET | 0 | 720.835 / 740.146 | 0.04538% |

QUIET is the only sample-size-qualified row and its maximum observed latency is
769.890 us, below the 770.605-us deadline. The raw-replayed evidence is
`results/p9-common-sota-williams-nonthermal-formal-raw-aggregate-6x1100-20260809T153122Z/summary.json`.
This still cannot replace the final thermal-normalized campaign.

## Exploratory hardware result

The one-epoch, 80-request smoke in
`results/p9-scenario-smoke-r3-20260809` validates the execution path only. It is
not statistically sufficient for a paper claim.

| Scenario | System/control | BE goodput (req/s) | Critical misses | Critical p99 (ms) |
|---|---|---:|---:|---:|
| Independent | QUIET | 1417.55 | 0/80 | 5.208 |
| Independent | Resident-only quiescence | 555.52 | 0/80 | 5.144 |
| Independent | Static full gating | 1432.96 | 0/80 | 5.220 |
| Independent | MIG isolation | 820.46 | 10/80 | 5.791 |
| Dependent | QUIET | 901.74 | 0/80 | 5.183 |
| Dependent | Resident-only quiescence | 550.60 | 0/80 | 5.177 |
| Dependent | Static full gating | 901.50 | 0/80 | 5.228 |
| Dependent | MIG isolation | 815.40 | 9/80 | 5.673 |

Only the independent rows support preliminary performance observations.
Dependent rows are retained as control-path debugging evidence and must not be
used to claim cross-instance pipeline performance. QUIET also does not beat
static full gating in this stationary smoke, so adaptive-policy superiority is
not established.

## SOTA fidelity

| Comparator | Required fidelity before a numeric row | Current status |
|---|---|---|
| Orion (EuroSys'24) | Preserve operation profiling, compute/memory classification, and profile-aware admission/reordering over the common workload. | Same-workload, payload-valid 100-request numeric smoke complete; repeated campaign pending. |
| XSched (OSDI'25) | Preserve the native XQueue command interception and suspend/resume backend; process stop signals are not an acceptable substitute. | Native Thor functional gate and same-workload 100-request numeric smoke complete. |
| gpulet (ATC'22) | Preserve profiling, interference/rate feasibility, elastic MPS partition candidates, and best-fit selection. | Pinned upstream planner and all five Thor partitions replay-verified; no schedulable partition at the common deadline. |
| BLESS (EuroSys'25) | Preserve MPS context quotas, profiled safe switch boundaries, selected-only physical launch, and squad/shadow progress. | Functional gates only. Numeric row prohibited until the kernel-squad scheduler drives the common workload. |
| Pantheon (MobiSys'24) | Preserve offline DNN processing, nested-redundancy variants, dynamic priorities, and the paper's accuracy contract. | Official artifact inspected and native gate passes; common-workload/accuracy adapter pending. Edge-specific secondary row. |
| BOER / ParvaGPU | Preserve the original configuration search or placement planner, respectively, on the fixed topology. | Supplemental provisioning controls only; their infeasibility does not replace a runnable runtime competitor. |

`BOER (Thor port)` or `ParvaGPU (Thor port)` may appear only when their output
contains the pinned upstream commit, regenerated profile hashes, the common
contract, and measured raw traces. A local MPS quota or static gate must never be
renamed after a published system.

The published comparators now have positive controls rather than being tested
only where they fail. BOER's pinned search selects q90/q10 for two independent
TensorRT services at 500 offered requests/s each; the worst p99 is 1.481 ms
against a 3-ms per-service SLO. ParvaGPU's pinned configurator allocates the
same services to the fixed 2g and 1g segments, and measured p99 is 0.434 and
0.969 ms at approximately 500 served requests/s each. When the ResNet output
becomes an input to the 2g control stage, BOER finds no 760-us feasible
per-service MPS configuration and ParvaGPU cannot allocate both the reserved
consumer and two independent service segments. The paired, hash-bound smoke is
`results/p9-sota-workload-scope-smoke-20260809/summary.json`.

The corrected payload-valid BOER v3 smoke preserves the pinned search algorithm
and upstream complementary share rule: the ResNet producer receives searched
share `SM`, while DistilBERT receives `100-SM`. The superseded v1 run assigned
the same share to both clients and its q75/q90 launch failures are invalid BOER
evidence. V3 binds the common frozen deadline and evaluates the actual
dependent-small pipeline. Its best observed candidate q90-r200 misses 299/300
requests with a 936.665-us p99, so BOER's published search returns no feasible
configuration. The result is
`results/p9-boer-dependent-payload-search-v3-common-deadline-20260809/search.json`.
This remains a structural provisioning result rather than a runnable dependent
DAG scheduler.

The payload-valid ParvaGPU v3 replay uses freshly rebuilt isolated q100 profiles
for the ResNet producer and DistilBERT background, with the raw JSON hashes
bound into the allocation result. Both points are individually eligible, but
the original allocator requests two 1g segments after the 2g consumer has been
reserved. Only one remains, so it returns `insufficient fixed MIG segments`.
The machine-readable result is
`results/p9-parvagpu-dependent-profile-v3-common-deadline-20260809/allocation.json`.

The mechanism evidence explains why dependence is not automatically slow under
MIG. Device memory cannot be shared across instances, but both TensorRT contexts
can directly bind the same registered coherent system-memory tensor. In the
balanced 2.304-MB transport experiment, cross-MIG registered transfer has a
14.058-us edge p99, compared with 114.041 us for a pinned bounce and 115.576 us
for a pageable bounce. The paired pinned-minus-registered difference is 98.248
us with a t95 interval [92.710, 103.786] us. The failure mode of BOER and
ParvaGPU is therefore missing precedence/slack and fixed segment capacity, not
an unavoidable payload copy.

QUIET's stage-scope ablation separates DAG-aware scheduling from coarse time
division. Both producer-only and full-pipeline quiescence have zero misses over
1,500 requests, while producer-only quiescence preserves 499.961 background
requests/s versus 152.666 requests/s for full-pipeline quiescence (3.27x). The
raw-replayed combined evidence is
`results/p9-resnet-dependent-structural-limit-evidence-v2-20260809T144929Z/summary.json`.

Orion's incompatibility is now traced rather than inferred from the failed
`LD_PRELOAD` probe. Nsight Systems records 108 `cuLaunchKernelEx` calls for six
ResNet10 executions and zero calls to `cudaLaunchKernel`, cuDNN, or cuBLAS
compute APIs intercepted by the pinned Orion artifact. Its source has no
driver-launch wrapper. A managed frontend would initialize queues but still
could not schedule TensorRT compute. The source/report-bound result is
`results/p9-orion-tensorrt-api-probe-20260809/compatibility.json`.

The public functional comparison is assembled in
`results/p9-dependent-payload-six-system-smoke-20260809/summary.json`. At 250
offered background RPS, NVIDIA MIG misses 46/100 requests and NVIDIA MPS misses
42/100; QUIET misses 0/100 with 722.8-us post-release p99 and 248.7 background
req/s. BOER has no search-feasible point, ParvaGPU is allocation-infeasible,
and Orion lacks the required interception surface. Missing SOTA numbers remain
null rather than being replaced with local look-alike policies.

QUIET's first measured quota sweep evaluates producer/background MPS shares
25/75, 50/50, 75/25, 90/10, and 100/100 on the same 14,720-byte dependent
pipeline at 250 offered background requests/s. Only q100/q100 satisfies the
exploratory 760-us deadline (717.9-us p99, 0/100 misses). Lower producer shares
remain fixed while the background is paused and therefore cannot reclaim the
critical stage's withheld SM capacity. The stage-DAG planner consequently
selects q100/q100 and uses temporal handback for background work. The bound
candidate spec and selected plan are in
`results/p9-quiet-quota-selection-250rps-v2-20260809/`.

The second payload-valid DAG uses Whisper Tiny `last_hidden_state`
(`1x1500x384` FP32, 2.304 MiB) and a shape-compatible TensorRT projection
consumer. Registered cross-MIG binding has 7.95-us edge p99, versus 102.7 us
for pinned bounce and 104.1 us for pageable bounce. Validation-excluded full
p99 is 1.547, 1.616, and 1.617 ms. Same-instance MPS records 8.32-us edge p99,
so cross-MIG placement is not measurably slower in this smoke. The mechanism is
coherent system-memory mapping, not cross-instance CUDA device-memory sharing.
The hash-bound ablation is
`results/p9-mig-trt-whisper-pipeline-smoke-20260809/summary.json`.

The corrected large-edge interference smoke adds a 250-request/s DistilBERT
tenant to the producer's 1g instance and excludes all three checksum-only
validation intervals from the deadline metric. Across 1,000 requests, NVIDIA
MIG, NVIDIA MPS, and process-stop miss 550, 586, and 125 requests; their p99s
are 2.226, 2.244, and 2.061 ms. QUIET records 0/1,000 misses at 1.574-ms p99
and retains 249.93 background requests/s. Three independent QUIET smoke
repetitions likewise record 0/3,000 misses with 1.565--1.582-ms p99.
Full-pipeline gating is not selected: it reduces background goodput to
165--167 requests/s and still observes 1/3,000 misses. The raw request traces
are under `results/p9-dependent-whisper-corrected-four-system-1000r-250rps-20260809-v1`
and `results/p9-quiet-whisper-producer-scope-corrected-1000r-250rps-r*-20260809`.
These remain mechanism repetitions, not a confidence-qualified SLO claim.

The same-contract SOTA result is now complete at smoke fidelity. BOER's pinned
Bayesian search, ParvaGPU's pinned segment configurator, and Orion's pinned
interceptor are evaluated rather than relabeled local policies. BOER has no
feasible complementary-MPS point, ParvaGPU has no SLO-feasible 1g Whisper
profile, and Orion sees 259 unsupported driver launches with zero visible
compute calls. The unified six-system artifact is
`results/p9-dependent-whisper-six-system-smoke-20260809/summary.json`.

Across three 1,000-request repetitions, QUIET records 1/3,000 misses and
1.570-ms pooled p99 while preserving 249.93 background requests/s. NVIDIA MIG
and NVIDIA MPS miss 1,654 and 1,772 requests; process-stop misses 397. The
single QUIET miss is a producer-compute outlier, not tensor transport. This
establishes the mechanism and failure attribution; the next campaign must
freeze the deadline independently and increase the request count before making
an SLO-confidence claim.

The next-stage smoke no longer uses the exploratory deadline. Five independent
1,000-request isolated pipeline blocks produce a pooled p99 of 1,546.651 us;
the fixed 1.10 interference budget yields a 1,701.316-us lock. With that lock,
three repetitions record 0/3,000 misses for QUIET, versus 1,431 for NVIDIA MIG,
1,630 for NVIDIA MPS, and 320 for process-stop. QUIET pooled p99/max are
1,576.9/1,609.9 us and background goodput remains 249.90 requests/s. BOER,
ParvaGPU, and the public comparison artifact are independently bound to the
same lock. The artifacts are
`results/p9-whisper-pipeline-deadline-calibration-5x1000-20260809/`,
`results/p9-dependent-whisper-frozen-repeated-3000r-250rps-20260809/`, and
`results/p9-dependent-whisper-frozen-six-system-smoke-20260809/`.

The initial locked offered-load frontier is also complete. QUIET serves
249.93/499.94/532.19 background requests/s at 250/500/800 offered RPS with
zero misses in each 1,000-request smoke. At 800 RPS, MIG and MPS serve the
offered load but miss 1000/1000 critical requests; process-stop serves 509.22
RPS and misses 553/1000. Separate 550/600-RPS probes place QUIET saturation at
approximately 535 RPS. The next repetition campaign should therefore focus on
500 and 600 offered RPS rather than spending equal samples on clearly
infeasible MIG/MPS saturation points.

That balanced repetition campaign is complete. Four runs at each load use a
four-treatment Williams order over NVIDIA MIG, NVIDIA MPS, the process-stop
ablation, and QUIET. Every system therefore appears in every ordinal position
once; the analyzer rejects any other four-run order and replays all request
traces before aggregation.

| Offered load | System | Misses / requests | Pooled p99 (us) | Mean background goodput (RPS) | One-sided CP95 DMR upper |
|---:|---|---:|---:|---:|---:|
| 500 | NVIDIA MIG | 5,562 / 6,000 | 2,216.54 | 499.96 | 93.2453% |
| 500 | NVIDIA MPS | 5,925 / 6,000 | 2,196.26 | 499.96 | 98.9765% |
| 500 | Process-stop ablation | 3,079 / 6,000 | 2,091.32 | 496.97 | 52.3857% |
| 500 | QUIET | **0 / 6,000** | **1,572.94** | **499.95** | **0.049916%** |
| 600 | NVIDIA MIG | 3,998 / 4,000 | 2,210.39 | 599.95 | 99.9911% |
| 600 | NVIDIA MPS | 4,000 / 4,000 | 2,240.89 | 599.95 | 100.0000% |
| 600 | Process-stop ablation | 2,055 / 4,000 | 2,098.79 | 506.51 | 52.6863% |
| 600 | QUIET | **1 / 4,000** | **1,578.55** | **533.58** | **0.1185%** |

The 600-RPS QUIET miss is a producer-compute outlier (1,898.63 us), not an
edge-transport outlier. The fresh 500-RPS campaign is confidence-qualified in
the smoke condition: 0/6,000 has a 0.049916% exact upper bound against the
0.05% target. The 600-RPS point is not qualified, and neither result replaces
the final thermal/formal campaign. Evidence is in
`results/p9-dependent-whisper-frozen-williams-500rps-4x1500-20260809/` and
`results/p9-dependent-whisper-frozen-williams-600rps-4x1000-20260809/`.

## Actual dependent-small stress smoke

One independent DistilBERT tenant was added to the ResNet10-to-policy DAG. MIG
isolation alone missed 1000/1000 deadlines because it isolates the 2g consumer
but not the 1g producer from another 1g tenant. Same-instance MPS also missed
1000/1000. Plain process-stop gating missed 866/1000 because submitted kernels
continued executing. QUIET's inference-boundary cooperative drain restored
0/1000 misses with 713.8-us p99, while retaining 727.1 DistilBERT req/s versus
1064.1 req/s unprotected. Its pre-release drain p99 was 880.2 us and is reported
separately. These are single-run mechanism results, not statistical claims.

The newer stage-resolved regression in
`results/p9-dependent-stage-smoke-20260809` uses the rebuilt instrumented
pipeline for 200 requests per configuration. It confirms the same mechanism
and attributes the failure: MIG isolation and MPS spatial sharing inflate the
1g producer-compute p99 to 1,003.6 and 969.2 us, respectively, while the actual
14,720-byte cross-MIG edge remains 49.0 and 45.4 us. QUIET records 601.5-us
producer p99, 51.2-us edge p99, 711.0-us full p99, and 0/200 misses. These
numbers supersede the earlier stress smoke for stage attribution, but remain a
single-run exploratory result.

A short offered-load check also found and fixed an idle-worker handback defect.
The cooperative worker now acknowledges a pause signal while sleeping between
requests instead of waiting for the next release. With a declared 1-ms
critical lookahead, QUIET is arrival-bound feasible at 100 and 250 offered RPS:
it records 0/100 misses, 721.3/722.8-us post-release p99, and
879.4/952.3-us drain p99. NVIDIA MIG misses 14/100 and 46/100; NVIDIA MPS misses
15/100 and 42/100. The two-point v2 artifact is
`results/p9-dependent-frontier-idle-ack-v2-20260809`. It is a functional
frontier check, not a confidence-qualified result.

## Required experiments

1. Run QUIET, MIG isolation, resident-only quiescence, and static full gating on
   both scenarios with the same frozen deadline and balanced order.
2. Profile and run BOER on the same `2g+1g` layout. This is the primary SOTA
   comparison because it jointly uses MIG and MPS and targets inference QoS.
3. Port and run ParvaGPU as a secondary spatial planner comparison. Report any
   infeasible allocation rather than silently changing the topology.
4. Attempt Orion once with a recorded build/interposition probe. Keep it as
   literature-only or incompatible if TensorRT hides the required operations.
5. Report critical DMR with a one-sided exact 95% bound, p99/p99.9, total and
   per-modality goodput, and paired run-level goodput ratios. A zero-miss smoke
   is not an SLO certificate.
6. Separate the borrowing claim from the adaptive claim. QUIET versus
   resident-only isolates borrowing value; QUIET versus static full gating
   isolates feedback value. If the latter has no positive paired confidence
   interval, describe the controller as protection logic rather than a
throughput contribution.

## Same-contract SOTA port smoke

The BOER adapter preserves the pinned artifact's Bayesian search schedule,
static and dynamic pruning, and normalized two-service objective. Thor capacity
profiles selected `q25/200 RPS` for the independent scenario and `q50/200 RPS`
for the dependent scenario. The selected BOER point and QUIET were then replayed
for 800 critical requests with the same 5.607159734-ms deadline and 200 offered
RPS per pressure tenant. QUIET used the smallest tested q100 handback guard that
completed successfully (5 ms; 3 and 4 ms failed closed).

| Scenario | System | Pressure goodput (req/s) | Critical misses | DMR | Critical p99 (ms) |
|---|---|---:|---:|---:|---:|
| Independent | QUIET | 397.97 | 1/800 | 0.125% | 5.268 |
| Independent | BOER (Thor port) | 399.69 | 6/800 | 0.750% | 5.600 |
| Dependent | QUIET | 397.95 | 0/800 | 0% | 5.204 |
| Dependent | BOER (Thor port) | 399.62 | 35/800 | 4.375% | 5.678 |

At this offered load, both systems saturate the demand and therefore have
nearly identical goodput. BOER's native p99 feasibility test does not enforce
QUIET's 0.05% DMR objective: its independent p99 remains just below the
deadline while six requests still miss. QUIET substantially reduces misses,
but its independent run also exceeds the target. Consequently, these results
motivate a predeclared offered-load sweep whose primary metric is maximum
goodput satisfying the DMR confidence bound; they are not a formal headline.

The ParvaGPU Thor configurator was run with fresh isolated `1g/q100` profiles.
It requests one 1g segment for audio and one for language. Because the fixed
layout reserves 2g for the critical service and leaves only one 1g segment,
the original allocation model reports `infeasible`. Orion's pinned interceptor
and scheduler both build on Thor with CUDA 13.0, but direct injection into the
opaque TensorRT benchmark terminates with SIGSEGV before Orion's managed-client
queues are initialized. Orion is therefore compatibility-only in this study.

The formal paper headline must wait for the same-contract BOER run and repeated
QUIET measurements. Until then, the defensible result is implementation
validation plus exploratory characterization.

## Fast comparison matrix

Before another formal campaign, the executable surface is checked with one
80-request epoch at 200 offered RPS per pressure tenant. The public comparison
rows are exactly NVIDIA MIG isolation, NVIDIA MPS spatial sharing, BOER,
ParvaGPU, Orion, and QUIET. Internal fixed-gating policies remain ablations and
do not appear as additional systems.

| Scenario | System | Status | Pressure goodput (req/s) | Misses | Critical p99 (ms) |
|---|---|---|---:|---:|---:|
| Independent | NVIDIA MIG isolation | measured smoke | 395.14 | 0/80 | 5.508 |
| Independent | NVIDIA MPS spatial sharing | measured smoke | 396.84 | 20/80 | 7.728 |
| Independent | BOER (Thor port) | measured smoke | 396.80 | 2/80 | 5.622 |
| Independent | ParvaGPU (Thor port) | fixed-layout infeasible | -- | -- | -- |
| Independent | Orion (Thor probe) | managed client required | -- | -- | -- |
| Independent | QUIET | measured smoke | 370.27 | 0/80 | 5.206 |
| Dependent | NVIDIA MIG isolation | measured smoke | 393.15 | 0/80 | 5.467 |
| Dependent | NVIDIA MPS spatial sharing | measured smoke | 394.88 | 20/80 | 7.610 |
| Dependent | BOER (Thor port) | measured smoke | 392.88 | 1/80 | 5.604 |
| Dependent | ParvaGPU (Thor port) | fixed-layout infeasible | -- | -- | -- |
| Dependent | Orion (Thor probe) | managed client required | -- | -- | -- |
| Dependent | QUIET | measured smoke | 365.26 | 0/80 | 5.242 |

The workload is ResNet-50 vision as the 2g critical service plus Whisper audio
and DistilBERT language pressure. In the independent scenario, audio and
language execute concurrently without an edge. In the current dependent
scenario, each paired language inference consumes a completion token from
audio, but the token carries no tensor payload. The reported dependent numbers
therefore verify scheduling mechanics only.

This table verifies integration only. It also exposes the next research
question: MIG isolation already has zero misses in this short run and higher
goodput than QUIET. The formal evaluation therefore must increase offered
tenant demand or report the SLO-goodput frontier; it must not claim a QUIET
speedup from this smoke.
