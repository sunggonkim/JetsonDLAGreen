# QUIET on Jetson AGX Thor

QUIET is a dependency-aware runtime for two-stage TensorRT inference across
isolated edge-GPU partitions.  It carries a complete intermediate activation
between separate Thor MIG CUDA contexts through registered coherent system
memory, protects the producer from best-effort interference, and returns that
capacity immediately after publication while the consumer executes.

The implementation treats the activation edge, publication event, and
remaining end-to-end slack as one scheduling contract.  It does not modify
TensorRT engines, CUDA kernels, or the NVIDIA driver.

## Current result

The promoted experiment is a six-session, counterbalanced,
thermal-normalized ResNet-50/ImageNette campaign.  Every system executes the
same operational arrival trace, 1g-to-2g placement, model split, output oracle,
and frozen 2,255.483-us arrival-to-completion deadline.

| System | Requests | Misses | one-sided CP95 DMR | p99 (us) | BE requests/s |
|---|---:|---:|---:|---:|---:|
| **QUIET** | 6,600 | 0 | **0.0454%** | **1,902.987** | 249.909 |
| NVIDIA MPS | 6,600 | 2 | 0.0954% | 2,045.606 | 249.941 |
| XSched (native Thor port) | 6,600 | 6,600 | 100.0000% | 4,351.332 | 97.845 |

QUIET is the only row whose exact confidence bound qualifies for the frozen
0.05% deadline-miss target.  Across sessions, its p99 reduction relative to
MPS is 138.508 us; the paired background-goodput interval spans zero.  The
real ImageNette and LibriSpeech/Whisper application gates preserve their
reference metrics.  Causal replay, transport controls, and the three-load
frontier remain explicitly exploratory or descriptive.

The complete claim scope is documented in
[`docs/p9-current-status.md`](docs/p9-current-status.md) and in the
[compiled paper](paper/eurosys27/p9-main.pdf).

## Repository layout

```text
.
├── analysis/       Evidence replay, statistics, audits, and paper generation
├── baselines/      Native comparator ports and Thor compatibility adapters
├── benchmarks/     C++/CUDA microbenchmarks and TensorRT pipeline binaries
├── configs/        Sanitized configuration examples
├── docs/           Platform notes, contracts, runbooks, and design history
├── include/        Shared C++ headers
├── models/         Download manifest and public label/class metadata
├── paper/eurosys27 Current LaTeX source, figures, tables, provenance, and PDF
├── runtime/        QUIET controllers, telemetry, and aggregation logic
├── scripts/        Model preparation and experiment orchestration
├── src/            Platform probe implementation
└── tests/          Python, shell-contract, and C++ unit tests
```

Build products, virtual environments, downloaded models, TensorRT engines,
machine-local MIG state, and raw experiment directories are intentionally not
tracked.  This keeps the public repository source-oriented while preserving
the 34-GB measurement corpus on the experiment host.

## Build

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
`-Wextra`, `-Wpedantic`, and `-Werror` where applicable.

The analysis and test tools use Python 3 together with NumPy, SciPy,
Matplotlib, Pillow, ONNX/ONNX Runtime, PyTorch, and OpenCV for the relevant
application preparation paths.

```bash
python3 -m pytest -q
```

## Platform configuration

`scripts/configure_thor_mig.sh` creates the fixed 1g+2g layout and writes a
machine-local environment file.  The variable schema is shown in
[`configs/mig.env.example`](configs/mig.env.example); real UUIDs are never
committed.

Experiment runners validate MIG UUIDs, model and input hashes, operational
release traces, thermal sensors, and runtime binaries before admitting a run.
System-level scripts may stop the display manager, pin clocks, or manage a
private MPS daemon and therefore require appropriate local privileges.

## Paper and figures

The current manuscript entry point is
[`paper/eurosys27/p9-main.tex`](paper/eurosys27/p9-main.tex).  Compile it with:

```bash
cd paper/eurosys27
pdflatex -interaction=nonstopmode -halt-on-error p9-main.tex
bibtex p9-main
pdflatex -interaction=nonstopmode -halt-on-error p9-main.tex
pdflatex -interaction=nonstopmode -halt-on-error p9-main.tex
```

The repository includes the compiled ten-page PDF, six publication figures in
PDF and PNG formats, generated tables, and output hashes.  On the measurement
host, figures can be regenerated from the bound raw corpus with:

```bash
python3 analysis/generate_p9_current_figures.py
```

The generator fails closed on input SHA-256, request-count, deadline,
application-gate, thermal-gate, miss-count, or replayed-p99 mismatches.

## Evidence policy

- `results/` is the local raw evidence store and is excluded from Git because
  it is approximately 30 GB.
- `models/cache/` and `models/engines/` are reproducible downloads/builds and
  are excluded because they are approximately 4 GB.
- `models/manifest.json` records public sources and expected hashes.
- `paper/eurosys27/generated/p9-figure-provenance.json` binds every promoted
  figure/table input and output.
- A published-system name is used numerically only when its pinned scheduler
  and runtime execute the measured request.  Local approximations retain
  their own control or diagnostic labels.

## Scope

The formal claim is limited to one Jetson AGX Thor, one fixed 1g+2g placement,
one thermal envelope, and a low-inflight two-stage ImageNette path.  The
validated external-process ring and larger-DAG planner schema are not yet the
production TensorRT data path, and the current result does not imply a general
multi-inflight or arbitrary-DAG performance guarantee.
