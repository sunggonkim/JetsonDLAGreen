# SOTA scope and baseline matrix

## Naming contract

The proposed system has one public name: **QUIET**. `joint-governor`,
`mig-governor`, and `jdg-governor` are raw experiment identifiers, not scheme
names. Descriptive controls such as fixed gating and resident-only execution are
baselines or QUIET ablations. They must not be presented as separate systems.

## Current reporting rule

Current reporting separates comparability from partial artifact evidence:

1. **Fixed common gate:** QUIET, NVIDIA MIG, NVIDIA MPS, XSched, Orion, and
   Pantheon repeat in this exact order in every end-to-end table and graph.
   Each row uses the same 90 labelled ImageNette inputs, operational arrival
   trace, artifacts, and 2,224.448-us deadline. This directional gate exposes
   measured SLO feasibility and is not a formal rank.
2. **Formal thermal subset:** QUIET, NVIDIA MPS, and native XSched share the
   six-session campaign. The other three roster names remain visible with
   dashes and an explicit promotion blocker.
3. **Executed partial evidence:** BLESS, GSLICE, gpulet, BOER, ParvaGPU, and
   DeepPlan report their successful positive control, selected plan, historical
   diagnostic, or measured incompatibility in a separate table.
4. **Reason-only literature scope:** systems without a current end-to-end row
   appear in the first README table with the precise blocker; no synthetic
   number is assigned.

The distinction is machine-readable in `docs/p9-comparator-manifest.json`.
`fixed_numeric_roster` and `executed_result_order` freeze the six-system order,
`partial_evidence_order` freezes the separate artifact table,
`direct_ranking_order` identifies the formal subset, and the older
`numeric_frontier_order` remains for historical exact-lock analysis scripts.
The generated counterparts are
`paper/eurosys27/generated/p9-six-system-imagenette-gate.json` and
`paper/eurosys27/generated/p9-current-results.tex`.

Systems for which the published implementation never ran, or whose design
requires changing the opaque TensorRT workload/control boundary, remain in the
non-numeric literature table. No local MPS, priority, or gating heuristic is
reported under another paper's name.

BLESS illustrates the distinction. Its scheduler, estimators, 2/4/6/8-SM
contexts, 9,400 traced launches, and q25 switching path execute, so BLESS is in
the ledger. The exact q100 TensorRT plan fails inside Myelin in the required
2-SM replica, so the row reports that measured compatibility boundary instead
of fabricating a common-workload latency. BOER and ParvaGPU likewise retain
their independent-service positive controls and their measured dependent-DAG
failures. DeepPlan retains its direct-host plan-selection result even though it
does not schedule the dependency.

## Historical design-space context

The validated P8 results below evaluate QUIET without MIG. P9 retains the QUIET
name while extending its isolation boundary to a fixed `2g+1g` layout. Earlier
smoke and nonthermal campaigns remain provenance; they are not pooled with the
current thermal ImageNette campaign.

The earlier common-workload hardware campaign used the real 14,720-byte
ResNet10 Layer7 covariance tensor, a shape-compatible TensorRT control stage,
and a 770.605-us wall deadline. Its six Williams sequences and 6,600-request
rows are historical nonthermal evidence. The still-earlier 600-request
campaign remains smoke evidence only.

The P9 workload contract now has two modes: `independent` multimodal tenants
run concurrently, while `dependent` tenants form an audio-completion to
language-inference chain. Both use the same fixed `2g+1g` layout, TensorRT
engines, critical trace, and deadline; only the dependency edges differ.

For the dependent path, device-local allocations remain isolated by MIG, but
the data plane uses registered coherent system memory bound directly by both
TensorRT contexts. The historical control-tensor ablation measures 14.058-us
cross-MIG registered edge p99 versus 114.041 us for a pinned bounce. The
current learned ResNet10 split is a necessary second check: its 1.884-MB
registered edge p99 is 2,195.51 us, with notification/precedence accounting for
the tail; pinned and pageable controls measure 1,863.77 and 2,283.20 us.
Consequently the paper attributes BOER/ParvaGPU failures to missing DAG/slack
modeling and fixed segment capacity, while explicitly avoiding the stronger
claim that cross-MIG dependence is always cheap or always copy-bound.

## Historical P8 bottom line

The validated P8 contribution is **not** another MPS/MIG/Green Context
characterization. QUIET targets opaque TensorRT clients and closes a specific
gap left by MPS quota and CUDA priority: work already issued by a client.

QUIET stops synchronous best-effort submitters, waits for their one-in-flight
work to drain, executes a critical burst, and resumes them. Its novelty boundary
is narrow:

1. unmodified serialized TensorRT engines and ordinary client processes;
2. no graph slicing, elastic-kernel generation, framework patch, driver patch,
   or MIG requirement;
3. modality-specific drain envelopes plus online guard/admission feedback; and
4. a no-MIG multimodal edge evaluation with cross-MIG used only as an oracle.

The current data show that black-box quiescence wins against MPS, priority, and
conservative time division in the near-zero-deadline-miss region. The adaptive
governor is statistically indistinguishable from the fixed profiled baseline,
and an offline-tuned 1.5-ms fixed guard is faster in the stationary sweep. The
data do **not** establish adaptive optimality or a direct performance win over
systems evaluated on other hardware and runtimes.

## Closest work

| Work | Venue | Scheduling boundary | What it requires | Honest relation to QUIET |
|---|---|---|---|---|
| [BOER](https://doi.org/10.1145/3712285.3759857) | SC 2025 | Joint MIG+MPS configuration | Discrete-GPU-oriented profiler, Bayesian optimizer, and serving cluster | Structural provisioning control only; not an online dependent-runtime competitor. |
| [ParvaGPU](https://doi.org/10.1109/SC41406.2024.00048) | SC 2024 | MIG segments with MPS replicas | Discrete-GPU topology, per-model profiling, and deployment planner | Structural provisioning control only; not a headline runtime row. |
| [BLESS](https://doi.org/10.1145/3689031.3696070) | EuroSys 2025 | Adaptive spatial-temporal bubble sharing | Kernel-level integration and MPS context control | The paper algorithm is reimplemented because no public runtime was found. q25 ResNet profiling and held-out switching pass, but the common q100 TensorRT plan fails in the required 2-SM replica, so this is a measured compatibility boundary rather than a numeric row. |
| [DeepPlan](https://doi.org/10.1145/3552326.3567508) | EuroSys 2023 | Layer-wise direct host access and parallel model transfer | Patched PyTorch 1.9 and multiple discrete GPUs | Data-plane comparator. Its upstream plan rule is applicable to coherent host access, but it has no dependent-DAG admission or slack policy. |
| [Mudi](https://chenwenyan.github.io/assets/pdf/mudi.pdf) | EuroSys 2025 | SLO-aware inference/training spatial multiplexing | Cluster profiler, interference model, adaptive batching, and resource scaling | Direct top-tier literature comparison; no public runtime artifact was found, so no numeric row is fabricated. |
| [GSLICE](https://doi.org/10.1145/3419111.3421284) | SoCC 2020 | MPS share and batching | Server inference controller | Direct MPS quota primitive; does not close already-issued queues. |
| [gpulet](https://www.usenix.org/conference/atc22/presentation/choi-seungbeom) | USENIX ATC 2022 | Spatio-temporal placement | Multi-GPU server and interference model | Direct spatial primitive, different placement scope. |
| [REEF](https://www.usenix.org/conference/osdi22/presentation/han) | OSDI 2022 | Kernel preemption/padding | GPU runtime support; mostly idempotent kernels | Much finer preemption. Not directly portable to the closed Jetson stack. |
| [Miriam](https://doi.org/10.1145/3625687.3625789) | SenSys 2023 | Elastic CUDA kernels | Source transformation and custom runtime | Strong edge baseline, but cannot accept an opaque TensorRT plan. |
| [HaX-CoNN](https://doi.org/10.1145/3627535.3638502) | PPoPP 2024 | Per-layer accelerator mapping | Heterogeneous GPU/DLA/NPU-style SoC | Explicitly models shared memory; Thor and Orin Nano paths here have no usable DLA. |
| [Pantheon](https://doi.org/10.1145/3643832.3661878) | MobiSys 2024 | Processed DNN scheduling and preemption | Offline DNN processing and online runtime | The pinned native runtime is in the fixed 90-input common gate; its protobuf API floors the shared 2,224.448-us deadline to 2,224 integer microseconds, and it is not yet in the thermal campaign. |
| [EdgeIso](https://doi.org/10.1109/IPDPS47924.2020.00039) | IPDPS 2020 | Dynamic CPU/GPU contention isolation on Jetson | User-level monitoring, DVFS, and incremental core allocation | Direct edge-SoC comparison for shared-memory contention. It is not relabeled as a GPU command scheduler and is evaluated on a separate CPU/EMC stress axis. |
| [DARIS](https://arxiv.org/abs/2504.08795) | 2025 preprint | MPS, streams, synchronized stages | Modified LibTorch path and segmented DNN stages | Closest concept. QUIET uses whole synchronous TensorRT processes; no direct cross-platform win is claimed. |
| [EdgeServing](https://arxiv.org/abs/2605.05527) | 2026 preprint | Deadline-aware time division and early exit | Model changes, batching, and edge scheduler | Direct edge scheduling context; not a drop-in comparator because QUIET preserves opaque TensorRT engines and does not alter model exits. |
| [Ev-Edge](https://arxiv.org/abs/2403.15717) | 2024 preprint | Multi-task event-based vision scheduling on Jetson Xavier | Event-camera pipelines and Xavier-specific runtime | Additional edge reference; its model/runtime contract differs from opaque TensorRT on Thor, so it remains literature-only. |
| [GCAPS](https://arxiv.org/abs/2406.05221) | 2024 preprint | GPU context-aware preemptive priority scheduling for real-time Jetson tasks | Driver/segment instrumentation and context preemption | Additional edge real-time reference; not a numeric comparator until its preemption semantics and accuracy contract are reproduced on fixed Thor MIG/TensorRT. |
| [GPreempt](https://www.usenix.org/conference/atc25/presentation/fan) | USENIX ATC 2025 | Timeslice-based GPU yield | Runtime/kernel-level preemption mechanism | More general and finer, but outside an unmodified TensorRT deployment. |
| [XSched](https://www.usenix.org/conference/osdi25/presentation/shen-weihang) | OSDI 2025 | Preemptible XPU command queues | Interposed XQueue abstraction and device backends | Native Thor suspend/resume gate and replay-verified common-workload numeric smoke passed. |
| [Performance Isolation for Inference Processes in Edge GPU Systems](https://arxiv.org/abs/2601.07600) | 2026 preprint | MPS/MIG/Green Context characterization | Cross-platform discrete/edge experiments | Closest mechanism overlap; no black-box SLO governor or Thor full-GPU multimodal campaign. |
The arXiv rows are preprints and must be identified as such in the paper.

## Executable baseline taxonomy

| Paper label | Exact implementation | Claim level |
|---|---|---|
| Isolated | One preloaded ResNet50 TensorRT context | Direct baseline |
| MPS q5/q25 | Per-worker `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` | Direct spatial baseline |
| Priority q25 | Low-priority pressure and high-priority critical CUDA streams | Direct primitive, **not Pantheon** |
| Conservative guard | Stop workers 6 ms before every critical burst | Direct fixed time-division baseline |
| Profiled guard | Fixed 1.5/2-ms modality envelope plus 0.5 ms | Direct offline baseline; closest DARIS-style staging primitive |
| QUIET | Profile plus adaptive slack and admission | Proposed system |
| Same-MIG | Critical and pressure workers in one instance | Separate ResNet10 oracle suite |
| Cross-MIG | Critical on 2g and pressure on 1g | Hardware isolation oracle, not the proposed system |
| Green Context | Explicit CUDA SM partition in the CUDA microbenchmark | Mechanism-only microbenchmark; never merged with DNN headlines |

## Historical P8 required and completed evaluation

- [x] Six counterbalanced full-GPU repetitions with 95% confidence intervals.
- [x] Real TensorRT ResNet50, DistilBERT, and Whisper execution with transfers.
- [x] MPS quota, CUDA priority, fixed temporal, profiled temporal, and adaptive
  governor baselines under identical traces.
- [x] One, two, four, and six offered-tenant scalability.
- [x] Per-request release-to-completion traces and deadline misses.
- [x] Gate overhead, board power, temperature, and device utilization proxy.
- [x] Separate same-MIG/cross-MIG TensorRT isolation oracle.
- [x] Balanced 0--6-ms guard sweep exposing a 1.5-ms stationary oracle.
- [ ] Additional critical architectures and non-NVIDIA edge GPU validation.
- [ ] Asynchronous clients with outstanding depth greater than one.
- [ ] Real sensor data and task-accuracy validation; current tensors are
  deterministic and shape-correct for timing evaluation.

## Claim rules

1. Do not call priority-q25 Pantheon, profiled gating DARIS, or MPS control
   GSLICE/gpulet. They are only directly executable primitives.
2. Do not combine CUDA microbenchmark and TensorRT numbers into one result.
3. Do not claim LPDDR/EMC bandwidth measurement: R39.2 telemetry records board
   power, temperature, SM activity, and a memory-utilization proxy.
4. Do not claim hard real-time preemption. QUIET waits for bounded work to
   drain and depends on the one-in-flight worker contract.
5. Do not claim direct SOTA throughput superiority across different hardware,
   models, or runtime modifications.
