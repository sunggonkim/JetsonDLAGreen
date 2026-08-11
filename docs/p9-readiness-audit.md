# QUIET P9 Readiness Audit

This is the current claim boundary for the production-wall dependent-DAG
extension. It is an audit of evidence, not a performance table.

| Area | Current assessment | Publication blocker | Promotion evidence |
|---|---|---|---|
| QUIET mechanism | Working two-stage prototype plus a separate three-slot external-process data-plane ring with sequence ownership, backpressure, timeout, and stale-reclaim telemetry | General multi-DAG and multi-inflight claims are not implemented; the ring is not yet the TensorRT production path | Three-stage/fan-in/fan-out workload, ring-vs-pipe control overhead, and application-level promotion |
| Motivation | Real ResNet dependent-head and Whisper producer/projection smokes exist; independent/dependent controls are exploratory | Causal effect is not yet thermal/session formal | Same workload and SLO, dependency-only toggle, payload and placement factorial, repeated sessions |
| Production latency | Completion-before-validation wall contract is implemented and trace-bound | Current rows are exploratory and not application-accuracy certified | Frozen production binary, common deadline lock, post-completion output trace, accuracy gate |
| Vendor controls | MIG isolation and MPS controls are represented with explicit topology | MIG is a capacity/isolation control, not a matched numeric BE baseline | Same workload, same arrival trace, same capacity disclosure, SLO-goodput frontier |
| Orion | Pinned managed-client Thor path now passes the shared labelled ImageNette gate (`0.8333` vs. `0.8333`, zero delta) | Local scheduler is not yet upstream-differential equivalent | Upstream-vs-port decision trace with zero mismatches, then repeated sessions and thermal normalization |
| XSched | Native Thor XQueue/suspend-resume path now passes the shared labelled ImageNette application gate (`0.8333` vs. `0.8333`, zero delta) | Current run is SLO-infeasible (`90/90` misses, `DMR=1.0`) and nonthermal | Repeated sessions, thermal normalization, and a feasible frontier point |
| Pantheon | Native positive control and formal pinned CIFAR-10/ResNet50 model-recovery artifact pass (`0.9329` final-exit held-out accuracy; exact full-output equivalence) | Its upstream model is `32x32` CIFAR-10, not the current labelled ImageNette ResNet-50 split edge; common-workload accuracy-equivalent adapter is missing | Upstream block/exit runtime on the exact current input/arrival/deadline contract, with bound accuracy/output traces |
| Edge literature | Miriam, EdgeServing, Ev-Edge, and *Performance Isolation for Inference Processes in Edge GPU Systems* are cited as mechanism/context comparators | They are not executable numeric rows on Thor | Reproduce the published runtime contract or keep them literature-only |
| Statistical evidence | Williams ordering, raw replay, SHA binding, and session-level smoke summaries exist | No thermal-normalized formal frontier; exploratory CP bounds exceed the 0.05% target | Independent sessions/reboots, paired/block bootstrap, exact miss bound, frozen formal campaign |
| Paper | Claim boundaries and generated evidence paths are tracked | Full-length evaluation/design expansion and final thermal tables remain | Regenerate figures/tables only from replay-verified formal aggregates |

## Fixed comparison names

The only proposed-system name is **QUIET**. The measured matrix is:

1. NVIDIA MIG isolation oracle/control;
2. NVIDIA MPS baseline;
3. Orion (EuroSys 2024 Thor port, only after differential fidelity);
4. XSched (OSDI 2025 Thor port);
5. Pantheon (MobiSys 2024 edge port, only after common-workload accuracy); and
6. QUIET.

Local policy IDs such as `mig-governor`, `fixed-full-gate`, or
`resident-full-gate` are internal baselines/ablations, never proposed-system
names. BOER, ParvaGPU, gpulet, BLESS, Miriam, EdgeServing, and the 2026 edge
isolation preprint remain structural or literature evidence until their exact
runtime and accuracy contracts are reproduced.

## Current evidence snapshot

- Python tests: `701/701` (including campaign preflight, split-DAG shape,
  transport-ablation, and parser-supported-label contract tests).
- CTest: `121/121` in the current `build-r39` tree (including the registered/direct-pageable transport contract
  and three-slot external-process ring contract, alongside preflight,
  split-DAG equivalence, active-frontier launcher, static full-gate smoke,
  KITTI conversion, application-trace decoding, and fast-pair shell).
- The active Williams input path now rejects validation-excluded or legacy
  rows before aggregation; the XSched verifier emits and checks `wall`,
  `production-wall-arrival-to-completion`, and `post-completion` metadata.
- Completion audit: `17` requirements verified, `6` deferred. The learned
  causal requirement now uses a current activation-replay aggregate; formal
  comparator, upstream fidelity, thermal, frontier, and paper gates remain
  open.
- Current learned ResNet10 causal replay uses three paired sessions for both
  QUIET and NVIDIA MPS. Each arm used the same 25-record `JDGACT1` replay
  (5 warmup plus 20 measured), the same `JDGARR1` release schedule, fixed
  1g-producer/2g-consumer placement, and post-completion correctness. The
  aggregate is at
  `results/p9-real-resnet-head-causal-current-wall-replay-20260811/`; it is
  current-wall, nonthermal, exploratory evidence rather than an SLO or ranking
  result.
- The current ImageNette ResNet-50 Williams aggregate has three SHA-bound
  sequences (`270` requests per system), zero observed misses for MPS, XSched,
  and QUIET, and descriptive p99 means of `1982.899/3780.742/2046.144` us.
  It is stored at
  `results/p9-active-resnet50-imagenette-frontier-r03-20260811/frontier.json`
  with `formal=false` and `ranking_allowed=false`; its CP95 upper DMR is
  `0.011034`, so it is not an SLO qualification.
- The active learned-workload launcher now regenerates each row's application
  prediction trace and post-completion/input-bound accuracy gate directly from
  the recorded raw evidence. It rechecks declared output/pipeline hashes and
  fails closed on missing reference assets; the three-sequence aggregate above
  was replayed through this path without rerunning hardware.
- Current real learned workload: ResNet10 backbone to learned detection head.
  The parser-supported detector labels are `Car`, `RoadSign`, and
  `TwoWheeler`; person-only COCO8 input is rejected as an invalid workload
  rather than being treated as a failed SOTA result.
- Full-model versus split producer/head graph equivalence now passes on the
  actual ResNet10 input shape `[1,3,368,640]` with zero numerical error; the
  record is `results/p9-real-resnet-head-graph-equivalence-20260810/equivalence.json`.
  This is not a task-accuracy gate and does not promote the workload.
- A ResNet-50 classification split is now executable: 1g/q100 backbone to a
  2g classification head (`1x1024x14x14`, 802,816-byte edge). The external
  ImageNet-labelled production-wall smoke is archived at
  `results/p9-resnet50-imagenet-mini10-paired-smoke-20260811/` (59 measured
  requests per arm, QUIET 2,016.274 us p99 with 1 miss and MPS 2,301.523 us
  p99 with 17 misses; both 42/59 top-1). The frozen 0.80 accuracy gate
  rejected that historical six-synset smoke, so it remains correctness/failure
  evidence only. The current standard ImageNette gate passes on 90 measured
  requests (`0.8333/0.8333`, zero delta) with bound input/output traces; it is
  an application gate, not an SLO or numeric frontier row. The producer-input
  trace SHA is bound in the common-workload contract; active learned-workload
  runners reject a missing or mismatched trace.
- The native XSched path accepts the same ResNet-50 split contract. Its
  promoted 90-request ImageNette run uses the same input/arrival/data contract
  as QUIET and matches the `0.8333` accuracy gate with zero delta, but measures
  `8169.063 us` p99 with 90/90 misses at the `5722.576 us` deadline. It is
  therefore a labelled correctness/failure-mode control
  (`formal_claim_allowed=false`), not a feasible SLO row.
- The first externally labelled KITTI smoke uses the official sample and
  explicit flip augmentations, but both QUIET and NVIDIA MPS score `0/20` at
  the fixed `0.90` detector gate.  The result is recorded as rejected negative
  evidence; lowering the threshold or relabeling model output is prohibited.
- Current large dependent transport workload: Whisper-Tiny projection edge.
- The real Whisper Tiny ASR path passes the official LibriSpeech `dev-clean`
  gate with `0.90/0.90` accuracy, equal `0.11266` mean WER, and byte-identical
  output traces. Together with the ImageNette gate, this establishes two real
  application gates; the existing comparator and thermal requirements remain
  open.
- Current numeric frontier: **none**. Existing MPS/XSched/QUIET rows are
  production-wall exploratory evidence with `ranking_allowed=false`.
- Fast exploratory performance snapshot (3 paired sessions, 300 requests per
  system, nonthermal): NVIDIA MPS `861.22 us` mean p99 / `249.88 rps` / `130`
  misses; XSched `1313.76 us` / `33.18 rps` / `300` misses; QUIET `655.40 us`
  / `247.87 rps` / `0` misses. This is directional only, not a paper ranking.
- A fresh learned-head fast pair (20 requests, same 1g producer/2g consumer,
  250-rps offer, inline correctness) at
  `results/p9-real-resnet-head-fast-current-20260811T180113/` completed with
  zero misses: QUIET p99 `699.973 us`, `247.917 rps`; NVIDIA MPS p99
  `3136.075 us`, `248.805 rps`.  The large change from the preceding short MPS
  run is retained as variance evidence; this exploratory command-line run has
  no common contract and is not a frontier point.
- Latest learned-head offered-load sweep is recorded at
  `results/p9-real-resnet-head-load-sweep-20260811/frontier.json`. It pairs
  QUIET and NVIDIA MPS at 125/250/375 requests/s with the same trained
  ResNet10 detection-head engine, wall deadline, placement, and inline output
  checks. All six 100-request points have zero observed misses; descriptive
  goodput reaches 362.45 rps for QUIET and 374.26 rps for NVIDIA MPS at the
  375-rps offer. The one-sided CP95 upper DMR is 2.95% at every point, so this
  is an exploratory same-contract frontier only, not a 0.05% SLO result or a
  SOTA ranking.
- Comparator contract now requires `common_workload` equality for workload ID,
  topology, placement, input tensor, payload bytes, arrival-trace SHA, and
  dataset-manifest SHA; old summaries lacking this object are not promotable.
- Numeric SOTA promotion additionally requires the shared formal evidence
  bundle: thermal normalization, paired session-level statistics, valid lock
  digests, `ranking_allowed=true`, and a one-sided 95% Clopper--Pearson DMR
  upper bound no greater than the frozen target. The latest load sweep remains
  exploratory because it has only 100 requests per point and its bound is
  2.95%, not 0.05%.
- The six-sequence Williams replayer now preserves old raw results but forces
  `numeric_comparison_allowed=false` when the run-level contract is absent;
  legacy SLO-qualified rows therefore cannot silently become a frontier.
- The independent-repeat and offered-load frontier aggregators apply the same
  rule, so no legacy load sweep can be promoted merely because its per-point
  DMR is low.
- A read-only `analysis/preflight_p9_campaign.py` now fails fast on missing or
  stale common-workload, accuracy, deadline, and active-boundary thermal
  inputs. It also requires the passed accuracy gate to match the common
  workload's workload ID, request count, and dataset manifest path/SHA before
  any long run starts. It never synthesizes labels or arrival records.
- The active MPS/XSched/QUIET wrapper performs deadline-lock verification before
  launching the first arm, forwards learned-workload producer traces and
  explicit warmup, and checks JDGINT1 record count against `warmup + requests`.
  Stale artifact hashes or trace-count mismatches therefore fail before a
  comparator consumes hardware time.
- A native XSched learned-head attempt was executed with 100 production-wall
  requests and inline correctness, but its verifier rejected the inherited
  deadline lock because calibration hashes no longer matched the current
  binary/source. The partial output is explicitly marked invalidated and is
  not included in any frontier; a fresh current-binary lock is required.
- The P1 transport characterization now covers registered-direct, direct
  pageable capability control, pinned bounce, pageable bounce, managed/UVM
  control, host-materialize control, same-instance MPS, and an explicit
  cross-MIG CUDA P2P/IPC negative control. The superseded pre-cache short
  artifact `results/p1-transport-short-20260811-r04/` is retained only for
  historical debugging; current evidence is in the expanded characterization
  artifacts below. The direct registered path is named the
  `full-coherent registered system-memory activation edge`.
- The current 100-request directional transport run is archived at
  `results/p1-transport-directional-20260811-r01/`. Registered-direct,
  pinned-bounce, pageable-bounce, and same-instance MPS all completed with
  zero checksum failures. This is a directional transport gate, not a
  thermal-normalized or application-accuracy result.
- The expanded transport characterization is archived at
  `results/p1-mig-sysmem-characterization-20260811-r05/`, with both
  `small-to-big` and `big-to-small` directions, warm/cold cache controls,
  four payload sizes at queue depth 1, and a registered-direct three-slot
  queue-depth-3 control. The explicit P2P/IPC negative control is included.
  The compute and memory-pressure subset is at
  `results/p1-mig-sysmem-characterization-pressure-20260811-r03/` and covers
  both queue depths. All executable cases passed the full
  producer-write/visibility/consumer-read checksum contract. These remain
  characterization artifacts.
- The three-slot data plane is exercised in
  `results/p1-ring-smoke-20260811-r06/`: normal wraparound and delayed
  consumer cases complete all requests with zero mismatches; the timeout
  control reports a bounded timeout; and the consumer-death case returns
  `fault-ok` with stale slot reclamation. This is an implementation/control
  artifact, not a promoted application result.
- QUIET plan actuation now binds UUID, transport, producer/background/consumer
  quotas, protection scope, and dependent admission into a pre-launch
  manifest. CLI transport mismatches fail closed before a CUDA process is
  started. A stale inherited deadline lock also fails closed and remains
  invalid promotion evidence.
- Planner profiles now carry an explicit tail model: a measured joint request
  p99 or a named risk budget is promotable; legacy component-p99 sums remain
  selectable only as visibly non-promotable characterization plans.
- The structural DAG contract now validates precedence, cycles, and
  three-stage/fan-out/fan-in topology. Current TensorRT execution remains
  two-stage; the contract therefore keeps general-DAG promotion disabled until
  a real multi-stage workload is executed.
- After the pipeline transport-description update, the current-binary causal
  replay/schema gate was refreshed at
  `results/p9-mig-trt-causal-contract-current-20260811-r05/`; dependent and
  independent arms consumed the same JDGACT1 activation bytes and JDGARR1
  operational schedule with identical output/checksum traces.

## MIG dependency interpretation

The dependent edge is not expected to be slow merely because its stages occupy
different MIG instances. QUIET's full-coherent registered system-memory
activation edge maps the same payload and avoids an explicit pageable/pinned
device-to-host and host-to-device bounce. The measured exploratory decomposition therefore
separates `edge_transport` from `transport_ready`: the former can remain small
while producer completion, notification, and consumer admission dominate the
dependent tail. The motivation experiment must hold model, arrival trace, SLO,
placement, and payload fixed while toggling only the dependency, then sweep
registered-direct, pinned-bounce, and pageable-bounce transports. It must report
payload-transfer, readiness/queue, consumer-start, and end-to-end wall p99
separately; a single end-to-end number cannot establish a MIG-memory claim.

## Next execution order

1. Run the preflight with the accepted externally labelled dataset/arrival
   manifest and producer-input trace, then run the active MPS/XSched/QUIET
   Williams matrix with the real downstream engine. Reject any comparator
   lacking its upstream differential or accuracy gate; do not reuse the
   rejected ResNet-50 gate as a promotion artifact.
2. Execute the dependency-only causal factorial before thermal normalization.
3. Collect the same-SLO offered-load/goodput frontier for every feasible row,
   including a conservative full-gate baseline.
4. Attach real labels and task-accuracy traces to the learned DAGs.
5. Freeze thermal/session conditions and rerun only the promoted matrix.
6. Regenerate the paper from the final replay-verified aggregate.
