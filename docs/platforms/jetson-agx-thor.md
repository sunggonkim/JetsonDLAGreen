# NVIDIA Jetson AGX Thor Validation

## Platform

This platform record was collected on 2026-08-06 with `SAMPLES=1000 WARMUP=100 ./scripts/run_p0.sh` after enabling `jetson_clocks`.

| Property | Observed value |
|---|---|
| Board | NVIDIA Jetson AGX Thor Developer Kit, 128 GB |
| L4T | R39.2.0 (upgraded in place from R38.4.0) |
| Power mode | MAXN |
| CUDA | Driver/runtime/compiler 13.2; compute capability 11.0 |
| GPU | NVIDIA Thor, 20 SMs |
| TensorRT | 10.16.2.10 |
| TensorRT-visible DLA cores | 0 |
| Green Context resource query | Supported |
| Green Context lifecycle probe | Supported |
| Green Context minimum SM partition | 8 SMs |
| Green Context co-scheduling alignment | 8 SMs |
| MPS daemon and CUDA-client lifecycle | Supported |
| MIG capability | Profiles 83 (`2g+gfx`) and 78 (`1g+me`) validated |
| Current main-experiment mode | Full GPU, MIG disabled |
| MIG instance SMs reported by CUDA | 12 SM and 8 SM |

The DLA device-tree nodes are absent and TensorRT reports zero DLA cores. Consequently, this board cannot produce the DLA-GPU placement results proposed in the main design. Thor experiments must use GPU partitioning, priority, MPS, and shared-resource control. DLA baselines require a separate DLA-equipped Jetson platform.

The system was upgraded to R39.2 using NVIDIA's documented APT minor-release
procedure. The package audit, dependency check, R39.2 kernel boot, MIG mode, and
both recommended instances were verified after reboot.

R39.2 has an important context-order hazard: after the graphics-capable 2g
instance owns an active context, a new context on the 1g instance can hang in
`cudaSetDevice`. The failure was reproduced locally, including the poisoned 1g
state after terminating the hung client. `configure_thor_mig.sh` recreates the
instances and starts a persistent MPS server on the 1g instance before starting
GDM on the 2g instance. Experiments must preserve this ordering and record the
workaround as part of the platform configuration.

## P0 Microbenchmark

The initial default-priority CUDA baseline produced the following release-to-completion latency. These values validate the harness; they are not DNN results and must not be used as paper evaluation data.

| Background | p50 (ms) | p99 (ms) | p99 / isolated |
|---|---:|---:|---:|
| None | 0.0293 | 0.0393 | 1.00x |
| Compute pressure | 0.0324 | 0.4113 | 10.48x |
| Memory pressure | 0.0848 | 6.9081 | 175.99x |

The CUDA event service-time distribution rises with the wall-clock distribution, so the observed tail is accelerator scheduling and service delay rather than only host timing overhead. The memory-pressure kernel is intentionally coarse and produces a naive-concurrency stress baseline. Subsequent experiments must compare high-priority streams and actual Green Context partitions before drawing an isolation conclusion.

The corresponding local artifacts are under `results/p0-20260806T094117Z/`. The directory is intentionally ignored by Git because future experiment runs should be managed as versioned artifact data rather than source code.

## P1 Cross-instance MIG Baseline

The first R39.2 baseline used 100,000 measured requests after 1,000 warm-up
requests. The latency-critical kernel ran on the 12-SM `2g` instance. A
persistent MPS server submitted pressure kernels to the 8-SM `1g` instance.
MAXN and maximum GPU/EMC clocks were active.

| Best-effort pressure | p50 (ms) | p99 (ms) | p99.9 (ms) | max (ms) | p99 / isolated |
|---|---:|---:|---:|---:|---:|
| None | 0.0417 | 0.0501 | 0.0565 | 0.0647 | 1.000x |
| Compute | 0.0416 | 0.0501 | 0.0558 | 0.1350 | 1.000x |
| Memory | 0.0427 | 0.0533 | 0.0598 | 0.2628 | 1.062x |

The result supports a narrow claim: cross-instance MIG removes the measured SM
service-time interference for this microbenchmark, while memory pressure still
inflates p99 and produces a 4.06x isolated-relative maximum. It is not yet a
multimodal DNN result. Repeated trials, telemetry, TensorRT workloads, deadline
misses, and a governor comparison are required before making a paper claim.

The local artifacts are under `results/p1-mig-20260806T115243Z/`. Reproduce the
measurement with `SAMPLES=100000 WARMUP=1000 PRESSURE_SECONDS=8
./scripts/run_p1_mig.sh` after exporting `SUDO_PASSWORD`.

## P8 Full-GPU QUIET Campaign

After returning the board to full-GPU mode, the formal campaign ran six cyclic
policy orders with ResNet50-v2 critical bursts and DistilBERT/Whisper pressure.
Every policy processed 57,600 measured critical requests. The isolated p99 was
`4.7981 +/- 0.0176 ms`, and the per-run 1.10x deadline averaged
`5.3091 +/- 0.0285 ms`.

| Policy | Deadline miss rate | Pressure goodput (requests/s) |
|---|---:|---:|
| MPS q5 | 0.292031 | 576.68 +/- 1.70 |
| MPS q25 | 0.480469 | 928.04 +/- 3.64 |
| Priority q25 | 0.384983 | 763.30 +/- 4.96 |
| Conservative guard | 0.000052 | 302.94 +/- 3.75 |
| Profiled guard | 0.000052 | 567.78 +/- 4.03 |
| QUIET | 0.000087 | 578.15 +/- 12.96 |

Intervals are Student-t 95% confidence intervals over six complete runs. The
source is `results/p8-campaign-quiet-final2-20260808/summary.json`. A separate
balanced sweep places the fixed-guard knee at 1.5 ms (zero observed misses,
599.60 +/- 7.70 requests/s) and shows that the current controller fails its target
at an 8-ms critical period. P0/P1 CUDA measurements and P5/P6 ResNet10 MIG
measurements remain separate mechanism and isolation-oracle suites.
