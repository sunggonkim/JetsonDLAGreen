# Current P9 status

This file records the evidence promoted by the current QUIET manuscript.  It
supersedes earlier P8, nonthermal, smoke, and token/control summaries as a
statement of the public claim.  Historical artifacts remain useful for design
traceability but do not widen this scope.

## System contract

QUIET's promoted vision path executes a low-inflight two-stage TensorRT chain
on one Jetson AGX Thor.  Its non-promoted Whisper motivation path uses a
bounded three-slot credit window to overlap adjacent requests.
The producer runs in a 1g MIG instance and the trained consumer head runs in a
2g instance.  Both processes register the same page-aligned system-memory
mapping in their own CUDA contexts and bind context-local pointers directly to
TensorRT.  The paper calls this the **full-coherent registered system-memory
activation edge**.

For each request, protection begins before producer release and ends when the
complete activation becomes visible to the consumer.  Best-effort work resumes
while the consumer runs.  Production-wall latency begins at the operational
arrival and ends at consumer completion; post-completion checksums and output
validation remain observable but do not extend that metric.

## Promoted thermal campaign

The formal workload is a ResNet-50 ImageNette classifier split at a
`[1,1024,14,14]` FP32 activation (802,816 bytes).  DistilBERT-SST2 supplies
open-loop best-effort load on the producer partition.  The independently
calibrated production-wall p99 is 2,050.439 us; the frozen 1.10 factor gives a
2,255.483-us deadline.

The campaign contains six counterbalanced sessions and 1,100 measured
requests per system per session:

| System | Requests | Misses | CP95 DMR | p99 (us) | p99.9 (us) | BE req/s |
|---|---:|---:|---:|---:|---:|---:|
| **QUIET** | 6,600 | 0 | **0.0454%** | **1,902.987** | 2,025.643 | 249.909 |
| NVIDIA MPS | 6,600 | 2 | 0.0954% | 2,045.606 | -- | 249.941 |
| XSched | 6,600 | 6,600 | 100.0000% | 4,351.332 | -- | 97.845 |

QUIET alone qualifies for the frozen 0.05% DMR target.  Across the six paired
sessions, QUIET minus MPS p99 is -138.508 us with a 95% interval of
[-233.217, -43.798] us.  The paired goodput effect is -0.0317 requests/s with
an interval of [-0.0988, 0.0354], so the evidence does not establish a
directional goodput difference.

Every thermal session passes the frozen sensor/range gate.  Each contains
430--434 telemetry samples, includes the VDD_GPU rail, has at most 3.907 C
within-session `soc012` range and 2.938 C `tj` range, and has at most 0.910 C
cross-session mean drift.  Temperature qualifies a session; it is not used to
rescale latency.

## Application gates

- **ImageNette:** 90 measured labelled inputs; reference and cross-partition
  accuracy are both 0.8333, with zero delta and a frozen 0.80 minimum.
- **LibriSpeech/Whisper-Tiny:** ten measured utterances; reference and
  candidate exact-match accuracy are both 0.9000, WER is 0.1127 for both, and
  complete output traces are byte-identical.

These gates establish that the runtime carries complete application
activations rather than tokens or constant payloads.

## Published comparators

- **XSched** is pinned at commit `bd494cb7a72958cd11900243a0798df00d856c6e`.
  Its CUDA shim, scheduling-unit queues, and policy execute the complete formal
  workload.  It is a faithful but deadline-infeasible row.
- **Pantheon** is pinned at commit
  `1caa4321fe9f9902ffacb78978f11a32a7a62f64`.  Its online runtime, priority
  tiers, model repository, and chunked modules execute a separate 90-request
  ImageNette gate with 0.8333 accuracy, two misses, and 4,133-us p99.  Its
  integer-microsecond adapter contract prevents pooling it with the thermal
  table.
- **Orion** remains outside numeric claims because the upstream differential
  decision gate is incomplete.  Other systems remain structural comparisons
  unless their original runtime executes the same model, input, release,
  output, and deadline contract.

Only QUIET is a proposed system name.

## Non-promoted controls

- A three-session nonthermal Whisper-Tiny crossover uses 19 requests/s, a
  250-ms internal deadline, and 300 requests per system.  NVIDIA MIG misses
  167/300 requests, static NVIDIA MPS misses 64/300, and QUIET misses 0/300.
  All outputs are byte-identical.  This is workload-real but arrival-trace
  partial: twelve labelled windows are cyclically replayed, so it is motivation
  evidence rather than a formal SLO or accuracy expansion.
- The three-load frontier has 3,300 requests per point and is not thermal
  normalized.  Its CP95 bounds are descriptive and no point is used for
  ranking.
- Same-activation causal replay and transport controls are exploratory.  They
  support the mechanism claim that precedence changes shared-SoC contention;
  they do not establish a formal SLO or universal transport ranking.
- The validated three-slot external-process ring is not yet integrated with
  the formal vision TensorRT path.  Whisper uses a narrower three-slot credit
  window without the standalone ring's timeout and stale-owner recovery.
- The planner can validate larger DAG schemas, but the promoted application is
  a two-stage chain and does not support an arbitrary-DAG performance claim.

## Artifact map

- Paper source and PDF: `paper/eurosys27/p9-main.tex` and `p9-main.pdf`
- Generated tables: `paper/eurosys27/generated/p9-current-results.tex`
- Figure/input hashes:
  `paper/eurosys27/generated/p9-figure-provenance.json`
- Figure generator: `analysis/generate_p9_current_figures.py`
- ASR crossover generator and compact evidence:
  `analysis/generate_p9_whisper_asr_crossover_figure.py` and
  `paper/eurosys27/generated/p9-whisper-asr-mig-crossover.json`
- Machine audit: `analysis/audit_p9_goal_completion.py`
- Real-application runbook: `docs/p9-real-application-runbook.md`
- Native-port contract: `docs/p9-sota-native-port-contract.md`

The raw `results/` tree and downloaded/compiled model products remain on the
measurement host and are intentionally excluded from Git because they occupy
approximately 34 GB.
