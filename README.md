# QUIET on Jetson AGX Thor

QUIET is a dependency-aware runtime for two-stage TensorRT inference across
isolated edge-GPU partitions. It carries a complete intermediate activation
between separate Thor MIG CUDA contexts through registered coherent system
memory, protects the producer from best-effort interference, and returns that
capacity immediately after publication while the consumer executes.

The implementation treats the activation edge, publication event, and
remaining end-to-end slack as one scheduling contract. It does not modify
TensorRT engines, CUDA kernels, or the NVIDIA driver.

<p align="center">
  <img src="paper/eurosys27/figures/p9-quiet-overview.png" width="100%" alt="QUIET profiles and locks the workload, places the two stages and coherent activation edge, executes according to publication state, and promotes only replay-verified evidence.">
</p>

## Results at a glance

### Why does the headline table have only three rows?

The repository contains two NVIDIA controls and implementations or Thor
adapters for **nine published systems**: XSched, Pantheon, Orion, BLESS,
GSLICE, gpulet, BOER, ParvaGPU, and DeepPlan. The three-row headline table is
not the full comparison inventory. It is the subset that executed the exact
formal ImageNette contract with the same model split, inputs, arrival trace,
output oracle, frozen deadline, current binary/engine hashes, six independent
sessions, and thermal gate.

Comparison evidence is therefore reported at four levels:

1. **Formal, directly ranked:** QUIET, NVIDIA MPS, and native XSched.
2. **Measured native gate, separately scoped:** Pantheon.
3. **Executed fidelity, compatibility, or structural result:** NVIDIA MIG,
   Orion, BLESS, GSLICE, gpulet, BOER, ParvaGPU, and DeepPlan.
4. **Literature-only mechanism comparison:** systems whose original runtime
   cannot execute the same opaque TensorRT dependency contract on Thor.

This separation prevents a local approximation, a different deadline, or an
infeasible planner output from being presented as a same-condition SOTA win.

### Formal thermal-normalized ImageNette campaign

The promoted experiment uses a fixed 1g-producer/2g-consumer placement, a
802,816-byte ResNet-50 activation, open-loop DistilBERT best-effort pressure,
and a frozen 2,255.483-us arrival-to-completion deadline. Each system executes
1,100 measured requests in each of six counterbalanced sessions.

| System | Requests | Misses | Observed DMR | one-sided CP95 DMR | p50 (us) | p99 (us) | p99.9 (us) | BE requests/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **QUIET** | 6,600 | **0** | **0.0000%** | **0.0454%** | 863.886 | **1,902.987** | **2,025.643** | 249.909 |
| NVIDIA MPS | 6,600 | 2 | 0.0303% | 0.0954% | **739.887** | 2,045.606 | 2,086.694 | **249.941** |
| XSched (native Thor port) | 6,600 | 6,600 | 100.0000% | 100.0000% | 3,484.202 | 4,351.332 | 4,741.721 | 97.845 |

QUIET is the only row whose exact confidence bound qualifies for the frozen
0.05% DMR target. Across the six paired sessions, QUIET reduces p99 relative
to MPS by 138.508 us, with a 95% interval of [-233.217, -43.798] us. The
paired background-goodput effect is -0.0317 requests/s with an interval of
[-0.0988, 0.0354], so the supported claim is lower tail latency at
statistically unresolved goodput difference, not a throughput win.

<p align="center">
  <img src="paper/eurosys27/figures/p9-latency-cdf.png" width="100%" alt="Full latency CDF and logarithmic tail for QUIET, NVIDIA MPS, and XSched in the formal campaign.">
</p>

### Application-semantic gates

| Workload | Measured inputs | Reference | Candidate | Delta | Promotion scope |
|---|---:|---:|---:|---:|---|
| ImageNette / ResNet-50 | 90 | 0.8333 accuracy | 0.8333 accuracy | 0.0000 | Labelled split-DAG gate passed |
| LibriSpeech / Whisper-Tiny | 10 | 0.9000 exact match | 0.9000 exact match | 0.0000 | Byte-identical output traces |
| LibriSpeech / Whisper-Tiny | 10 | 0.1127 WER | 0.1127 WER | 0.0000 | Below frozen 0.20 maximum |
| Formal ImageNette campaign | 6,600/system | 0.8345 accuracy | 0.8345 for every system | 0.0000 | Bound to every formal session |

## Complete SOTA comparison coverage

The table below accounts for every published system with code, a Thor port,
or measured compatibility evidence in this repository. Missing formal metrics
mean that reporting a number under the published name would violate the
common-workload or fidelity contract.

| System | Venue / role | What was actually executed | Measured outcome | Public status |
|---|---|---|---|---|
| [XSched](baselines/xsched/) | OSDI 2025 | Pinned native XQueue CUDA path, suspend/resume transitions, full formal ImageNette workload | 6,600/6,600 misses; p99 4,351.332 us; accuracy 0.8345 | **Formal comparator; SLO-infeasible** |
| [Pantheon](baselines/pantheon/) | MobiSys 2024 | Pinned online runtime, priority tiers, model repository, and chunked modules on the labelled ImageNette gate | 90 requests; accuracy 0.8333; 2 misses; p99 4,133 us; BE 249.785 requests/s | **Faithful native gate**, separate because its adapter uses an integer-us 2,224-us contract |
| [Orion](baselines/orion/) | EuroSys 2024 | Managed-client Thor path and the same 90-input ImageNette semantic gate | Accuracy 0.8333; 0/90 misses at its distinct 5,722.576-us diagnostic deadline; p99 5,537.090 us | Executed diagnostic only; upstream differential-decision trace remains open |
| [BLESS](baselines/bless/) | EuroSys 2025 | Paper scheduler and estimators, 2/4/6/8-SM contexts, 9,400 traced TensorRT launches, activation handoff, held-out switch boundaries | q25 gate passes; exact common q100 plan fails inside Myelin in the required 2-SM replica | Measured compatibility boundary; no fabricated numeric row |
| [GSLICE](baselines/gslice/) | SoCC 2020 | Algorithm port for quota selection on the earlier dependent Whisper contract | Historical 9,000-request run: 9,000 misses; p99 2,094.460 us | Superseded nonthermal mechanism evidence |
| [gpulet](baselines/gpulet/) | USENIX ATC 2022 | Pinned placement decision over all five representable complementary partitions | No SLO-feasible dependent plan; diagnostic run has 9,000/9,000 misses and p99 2,099.367 us | Structural planner limit, not a feasible numeric competitor |
| [BOER](baselines/boer/) | SC 2025 | Pinned Bayesian MIG+MPS optimizer on independent and dependent contracts | Independent positive control: worst p99 1.481 ms; dependent best p99 2.080 ms, no feasible point | Structural abstraction result |
| [ParvaGPU](baselines/parvagpu/) | SC 2024 | Pinned segment allocator plus independent-service execution | Independent p99 0.434/0.969 ms at about 500 requests/s; dependent producer rejected by admission rule | Structural abstraction result |
| [DeepPlan](baselines/deepplan/) | EuroSys 2023 | Published plan-selection rule applied to Thor coherent host access | Direct-host data-plane choice is applicable; no stage-precedence or end-to-end slack scheduler | Data-plane comparison only |

NVIDIA MIG is also measured as a physical-isolation/capacity oracle, but it
has no matched best-effort slice and is not a same-capacity formal row. NVIDIA
MPS is the matched vendor baseline in the formal table.

### Historical six-system campaign

This earlier campaign explains why the repository contains substantially more
comparison code than the three formal rows. It uses a different Whisper-based
2.304-MB dependency, a 1,701.316-us deadline, six Williams sequences, 9,000
requests per treatment, and no thermal normalization. It is raw-replayed
evidence, but it is **superseded for headline ranking** because the current
ImageNette application, binary lock, deadline, fidelity rules, and thermal
contract are different.

| Historical treatment | Requests | Misses | CP95 DMR | p99 (us) | BE requests/s | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| NVIDIA MIG isolation | 9,000 | 8,424 | 94.0194% | 2,215.748 | 499.963 | Capacity/isolation oracle |
| NVIDIA MPS | 9,000 | 8,950 | 99.5668% | 2,180.974 | 499.969 | Vendor spatial control |
| GSLICE port | 9,000 | 9,000 | 100.0000% | 2,094.460 | 499.955 | Quota-only policy misses dependency slack |
| gpulet port | 9,000 | 9,000 | 100.0000% | 2,099.367 | 499.959 | No feasible dependent placement |
| Orion managed-client port | 9,000 | 0 | 0.0333% | **1,569.738** | 153.571 | Full-DAG protection; fidelity gate incomplete |
| **QUIET** | 9,000 | **0** | **0.0333%** | 1,578.293 | **499.943** | Producer-only protection and consumer-stage handback |

In this historical contract, QUIET/Orion background goodput is 3.255x with a
95% paired-session interval of [3.241x, 3.270x]. This remains historical
mechanism evidence rather than a current formal SOTA claim. The tracked
explanation is in [`docs/p9-sota-reselection.md`](docs/p9-sota-reselection.md).

### Broader literature matrix

The project scope and related-work audit also compare the scheduling boundary
with systems that do not have an accuracy-equivalent opaque TensorRT execution
path on Thor. They remain literature comparisons; their names are never
attached to local heuristics.

<details>
<summary>Show literature-only and mechanism-level comparisons</summary>

| Work | Venue | Compared mechanism | Why it has no current numeric row |
|---|---|---|---|
| [Mudi](https://chenwenyan.github.io/assets/pdf/mudi.pdf) | EuroSys 2025 | SLO-aware inference/training spatial multiplexing | Cluster scaling and training objective differ; no public common-workload runtime artifact |
| [MIGER](https://doi.org/10.1145/3673038.3673089) | ICPP 2024 | Joint MIG and MPS allocation for deep-learning clusters | Independent cluster allocation does not preserve the request-specific activation edge or current Thor contract |
| [FluidFaaS](https://research.csc.ncsu.edu/picture/publications/papers/hpdc2025.pdf) | HPDC 2025 | Dynamic pipelines for strongly isolated serverless GPU functions | A100/serverless materialization semantics differ; retained as a host-memory materialize/copy concept, not a runtime row |
| [REEF](https://www.usenix.org/conference/osdi22/presentation/han) | OSDI 2022 | Kernel preemption and padding | Released reset path targets a different runtime/backend |
| [Miriam](https://doi.org/10.1145/3625687.3625789) | SenSys 2023 | Elastic CUDA kernels for edge inference | Requires generated kernels instead of opaque TensorRT plans |
| [HaX-CoNN](https://doi.org/10.1145/3627535.3638502) | PPoPP 2024 | Per-layer heterogeneous accelerator mapping | The evaluated Thor path has no equivalent usable DLA target |
| [EdgeIso](https://doi.org/10.1109/IPDPS47924.2020.00039) | IPDPS 2020 | CPU/GPU shared-resource isolation on Jetson | Relevant SoC contention control, not a cross-MIG command scheduler |
| [DARIS](https://arxiv.org/abs/2504.08795) | 2025 preprint | MPS, streams, and synchronized model stages | Requires a modified LibTorch segmented path |
| [EdgeServing](https://arxiv.org/abs/2605.05527) | 2026 preprint | Time division, batching, and early exits | Changes model/execution semantics and therefore needs a separate accuracy-equivalent adapter |
| [Ev-Edge](https://arxiv.org/abs/2403.15717) | 2024 preprint | Multi-task event-vision scheduling | Different event-camera workload and Xavier runtime |
| [GCAPS](https://arxiv.org/abs/2406.05221) | 2024 preprint | Context-aware preemptive priority | Requires driver/task-segment instrumentation not present in the opaque TensorRT contract |
| [GPreempt](https://www.usenix.org/conference/atc25/presentation/fan) | USENIX ATC 2025 | Timeslice-based GPU yield | Runtime/kernel-level preemption is outside the unmodified TensorRT boundary |
| [Edge-GPU process isolation study](https://arxiv.org/abs/2601.07600) | 2026 preprint | MPS/MIG/Green-Context characterization | No common Thor dependency, application gate, or QUIET SLO contract |

</details>

The historical design-space discussion is in
[`docs/sota-matrix.md`](docs/sota-matrix.md), and the current public claim
boundary is in [`docs/p9-current-status.md`](docs/p9-current-status.md).

## Additional measured results

### Same-activation causal replay

The independent and dependent arms use the same pre-captured activation bytes,
models, request IDs, placement, arrivals, pressure, and output oracle. Only
precedence changes. Each point below is a 20-request arm; the interval uses
three sequential session pairs and is exploratory rather than thermal formal
evidence.

| System | Mean dependent - independent p99 | 95% paired-session interval | Interpretation |
|---|---:|---:|---|
| **QUIET** | **-2,081.286 us** | **[-2,394.953, -1,767.619] us** | Dependency selects a lower-contention schedule in all three pairs |
| NVIDIA MPS | -514.579 us | [-4,779.753, 3,750.595] us | Direction is unresolved and varies across pairs |

<p align="center">
  <img src="paper/eurosys27/figures/p9-causal-pairs.png" width="100%" alt="Three paired same-activation independent and dependent p99 measurements for QUIET and NVIDIA MPS.">
</p>

### Transport control

The 20-request learned ResNet10 control carries the same 1,884,160-byte tensor
and emits the same output trace for every transport. It rejects the assumption
that registered direct binding is universally fastest; QUIET profiles the
joint stage/edge tail instead of assuming a copy constant.

| Transport | Edge p99 (us) | Production-wall p99 (us) | Scope |
|---|---:|---:|---|
| Registered coherent system memory, direct TensorRT binding | 2,195.508 | 2,908.268 | Current QUIET data path |
| Pinned D2H/H2D bounce | **1,863.770** | **2,538.617** | Explicit-copy control |
| Pageable bounce | 2,283.200 | 2,975.838 | Explicit-copy control |

These short nonthermal measurements characterize the mechanism; they are not
a universal transport ranking.

### Offered-load frontier

Each hollow point contains three sessions and 3,300 requests. The points are
descriptive because they are not thermal normalized and cannot by themselves
certify a 0.05% DMR target. The formal QUIET anchor is a separate six-session
point.

| System | Offered BE requests/s | Requests | Misses | CP95 DMR | p99 (us) | Completed BE requests/s |
|---|---:|---:|---:|---:|---:|---:|
| NVIDIA MPS | 125 | 3,300 | 0 | 0.0907% | 1,940.769 | 124.950 |
| NVIDIA MPS | 250 | 3,300 | 0 | 0.0907% | 1,990.308 | 249.888 |
| NVIDIA MPS | 375 | 3,300 | 0 | 0.0907% | 2,052.077 | 374.868 |
| QUIET | 125 | 3,300 | 1 | 0.1437% | 1,851.749 | 124.941 |
| QUIET | 250 | 3,300 | 4 | 0.2772% | 1,912.867 | 249.908 |
| QUIET | 375 | 3,300 | 29 | 1.1963% | 2,240.971 | 374.897 |
| **QUIET formal anchor** | 250 | 6,600 | **0** | **0.0454%** | **1,902.987** | 249.909 |

<p align="center">
  <img src="paper/eurosys27/figures/p9-load-frontier.png" width="100%" alt="Confidence-bounded deadline-miss ratio and completed background requests per second over three offered loads, plus the formal QUIET anchor.">
</p>

### Session and thermal stability

All six formal sessions pass the frozen thermal gate. Each has 430--434
telemetry samples and the required VDD_GPU rail; the largest within-session
`soc012` range is 3.907 °C, the largest `tj` range is 2.938 °C, and the largest
cross-session mean drift is 0.910 °C. Temperature admits or rejects a session;
it is not used to rescale latency.

<p align="center">
  <img src="paper/eurosys27/figures/p9-session-stability.png" width="100%" alt="Per-session p99 latency and frozen thermal envelope across six counterbalanced sessions.">
</p>

## Runtime boundary

QUIET begins protection before producer release and ends it immediately after
payload publication/visibility. Best-effort work resumes while the consumer
runs. Production-wall latency ends at consumer completion; checksums and
output validation happen afterward.

<p align="center">
  <img src="paper/eurosys27/figures/p9-stage-timeline.png" width="100%" alt="QUIET request timeline from declared arrival through pause, producer publication, best-effort resume, consumer completion, and post-completion validation.">
</p>

These six embedded images are all figures used by the current manuscript. PDF
and PNG versions, generated tables, and their SHA-256 provenance are under
[`paper/eurosys27/`](paper/eurosys27/).

## Repository layout

```text
.
├── analysis/       Evidence replay, statistics, audits, and paper generation
├── baselines/      Nine published-system ports/adapters and fidelity checks
├── benchmarks/     C++/CUDA microbenchmarks and TensorRT pipeline binaries
├── configs/        Sanitized configuration examples
├── docs/           Platform notes, contracts, runbooks, and comparison scope
├── include/        Shared C++ headers
├── models/         Download manifest and public label/class metadata
├── paper/eurosys27 Current LaTeX source, all figures, tables, provenance, PDF
├── runtime/        QUIET controllers, telemetry, and aggregation logic
├── scripts/        Model preparation and experiment orchestration
├── src/            Platform probe implementation
└── tests/          Python, shell-contract, and C++ unit tests
```

Build products, virtual environments, downloaded models, TensorRT engines,
machine-local MIG state, and raw experiment directories are intentionally not
tracked. This keeps the public repository source-oriented while preserving the
34-GB measurement corpus on the experiment host.

## Build and test

The native runtime requires Jetson Linux R39.2, CUDA 13.2, TensorRT 10.16.2,
CMake 3.25 or newer, Ninja, and a C++20 compiler.

```bash
cmake -S . -B build-r39 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=110
cmake --build build-r39 --parallel
ctest --test-dir build-r39 --output-on-failure
```

Project targets compile with strict host warnings, including `-Wall`,
`-Wextra`, `-Wpedantic`, and `-Werror` where applicable. The analysis and test
tools use Python 3 together with NumPy, SciPy, Matplotlib, Pillow, ONNX/ONNX
Runtime, PyTorch, and OpenCV for the relevant application preparation paths.

```bash
python3 -m pytest -q
```

## Platform configuration

`scripts/configure_thor_mig.sh` creates the fixed 1g+2g layout and writes a
machine-local environment file. The variable schema is shown in
[`configs/mig.env.example`](configs/mig.env.example); real UUIDs are never
committed.

Experiment runners validate MIG UUIDs, model and input hashes, operational
release traces, thermal sensors, and runtime binaries before admitting a run.
System-level scripts may stop the display manager, pin clocks, or manage a
private MPS daemon and therefore require appropriate local privileges.

## Paper and figure reproduction

The current manuscript entry point is
[`paper/eurosys27/p9-main.tex`](paper/eurosys27/p9-main.tex), and the compiled
ten-page paper is [`paper/eurosys27/p9-main.pdf`](paper/eurosys27/p9-main.pdf).

```bash
cd paper/eurosys27
pdflatex -interaction=nonstopmode -halt-on-error p9-main.tex
bibtex p9-main
pdflatex -interaction=nonstopmode -halt-on-error p9-main.tex
pdflatex -interaction=nonstopmode -halt-on-error p9-main.tex
```

On the measurement host, the current figures and tables can be regenerated
from the bound raw corpus with:

```bash
python3 analysis/generate_p9_current_figures.py
```

The generator fails closed on input SHA-256, request count, deadline,
application gate, thermal gate, miss count, or replayed-p99 mismatch.

## Evidence policy and scope

- `results/` is the local raw evidence store and is excluded from Git because
  it is approximately 30 GB.
- `models/cache/` and `models/engines/` are reproducible downloads/builds and
  are excluded because they are approximately 4 GB.
- `models/manifest.json` records public sources and expected hashes.
- `paper/eurosys27/generated/p9-figure-provenance.json` binds every promoted
  figure/table input and output.
- A published-system name is used numerically only when its pinned scheduler
  and runtime execute the measured request. Local approximations retain their
  own control or diagnostic labels.

The formal claim is limited to one Jetson AGX Thor, one fixed 1g+2g placement,
one thermal envelope, and a low-inflight two-stage ImageNette path. The
validated external-process ring and larger-DAG planner schema are not yet the
production TensorRT data path, and the current result does not imply a general
multi-inflight or arbitrary-DAG performance guarantee.
