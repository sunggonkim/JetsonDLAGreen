# QUIET cross-MIG dependent-inference redesign

## Invalidated assumption

The current dependent mode transports one byte of control state. It does not
move a TensorRT output into the downstream process. Its near-equal independent
and dependent latency is therefore expected and cannot support a cross-MIG
communication claim.

## Research gap

BOER and ParvaGPU optimize placement, MIG size, MPS share, and replica count
from per-model profiles. Their unit of optimization is an independently served
model, not a dataflow edge whose producer and consumer occupy isolated GPU
instances. Orion exposes operator-level scheduling through CUDA-library
interposition, but that boundary neither supplies cross-GPU-instance memory
sharing nor works for the current opaque TensorRT process.

The fixed `2g+1g` setting therefore exposes a concrete choice that these
systems do not make: co-locate a dependent edge and lose isolation, or separate
the stages and pay an explicit communication cost. QUIET should optimize this
choice, not present another quota controller.

## Proposed QUIET mechanism

1. Represent each workload as a stage DAG. Every edge records payload size,
   production deadline, consumer deadline, and whether a lossy transform is
   allowed.
2. Use a cross-MIG shared-memory ring backed by registered host memory. Thor
   places CPU and iGPU allocations in SoC DRAM. The capability probe now shows
   that a 1g producer and 2g consumer can independently register the same
   shared pages and perform checksum-correct GPU writes and reads. This is
   system-memory transport, not unsupported CUDA IPC between GPU instances.
3. Write the producer's actual output bytes into the ring and verify a payload
   sequence number plus checksum before the consumer launches. A control token
   alone is insufficient.
4. Reserve both communication time and downstream compute time in the slack
   admission decision. Co-locate edges whose measured transfer cost exceeds
   available slack; place small edges across instances to retain isolation.
5. Gate borrowers only when the critical stage or its communication reservation
   begins. The scheduler must report compute latency, handoff latency, and full
   producer-to-consumer latency separately.

The first implementation should compare three data paths rather than assume a
winner: pageable host bounce (`D2H+H2D`), pinned host bounce, and the
full-coherent registered system-memory activation edge. GPUDirect RDMA is not
the primary Thor path because embedded Tegra does not expose it as a supported
CUDA data path.

## Workloads

- **Independent:** camera vision, audio encoding, and language inference have
  no dataflow edges. This measures isolation and aggregate goodput.
- **Dependent-small:** ResNet output logits or features feed a small control
  policy. The first implementation is now complete: ResNet10 `Layer7_cov`
  (`1x4x23x40` FP32, 14,720 bytes) is bound directly to a TensorRT reduction +
  linear + sigmoid policy on the other MIG instance.
- **Dependent-large:** Whisper Tiny encoder `last_hidden_state`
  (`1x1500x384` FP32, 2,304,000 bytes) feeds a shape-compatible TensorRT
  sequence-average plus 384x16 projection and sigmoid consumer. DistilBERT is
  not used as an incompatible byte sink.

## Evaluation order

1. Capability probe: **complete on the current Thor**. Across 100 measured
   transfers per size, 500 total payloads had zero mismatched bytes. Handoff
   p99 was 65.4 us for 4 KiB, 101.1 us for 1 MiB, and 347.7 us for 8 MiB. The
   raw artifact is `results/p9-mig-sysmem-probe-20260809/result.json`.
2. Transport microbenchmark: 64 B through 64 MiB, p50/p99 latency, bandwidth,
   CPU utilization, and EMC pressure for all three paths.
3. End-to-end smoke: **complete for the small edge with per-request replay**.
   In 100 requests there were zero payload checksum failures, four distinct
   request payloads, and four distinct downstream policy outputs. The direct
   coherent edge p99 was 6.61 us and full ResNet-to-policy p99 was 707.58 us.
   The verifier binds the active 1g producer and 2g consumer UUIDs, tensor
   names and shape, binary/source/engine hashes, timing trace, and separate
   request checksum trace. Evidence is in
   `results/p9-resnet-layer7-control-mlp-100r-traced-v2-20260809T1420Z/`.
   The same binary now exposes `--dependency-mode independent`; its consumer
   runs concurrently on a separate deterministic input mapping, while the
   dependent arm consumes the producer's actual output mapping. Three paired
   real-edge repeats show an exploratory dependent-minus-independent wall-p99
   increase of 85.793--101.240 us (mean 95.882 us) with zero checksum
   failures. The paired-session descriptive 95% interval is 74.163--117.602 us
   (n=3); it is not a formal confidence claim. The causal artifact is
   `results/p9-causal-real-edge-20260810/repeats/summary.json`; the earlier
   pre-contract JSON is superseded.
4. Short frontier: MIG isolation with host bounce, same-instance MPS, BOER,
   and QUIET at increasing payload sizes and offered rates.
5. Only after the mechanism works, run repetitions and thermal controls for the
   paper.

The payload gate is also bound into the public small-payload comparison at
`results/p9-dependent-payload-six-system-smoke-v4-20260809T1422Z/summary.json`;
token-only dependent runs cannot satisfy that input contract.

The stage-resolved 200-request regression is now complete. With an independent
DistilBERT tenant sharing the producer's 1g instance, NVIDIA MIG isolation had
1,003.6-us producer-compute p99 and 1,101.3-us pipeline p99. NVIDIA MPS spatial
sharing had 969.2-us producer-compute p99 and 1,087.9-us pipeline p99. In both
cases all 200 requests missed the exploratory 760-us deadline. The
cross-instance edge itself remained 45--49 us p99, so this experiment does not
support the hypothesis that cross-MIG payload movement is the dominant cost.
The dominant cost is interference at the producer's shared 1g execution
domain. QUIET's cooperative handback reduced producer-compute p99 to 601.5 us
and pipeline p99 to 711.0 us with zero misses, while retaining 889.3
DistilBERT req/s. Raw evidence is in
`results/p9-dependent-stage-smoke-20260809`.

The communication-aware planner now consumes five measured quota candidates at
250 offered background requests/s. Producer/background shares of 25/75, 50/50,
75/25, and 90/10 produce pipeline p99 values of 1,760, 993, 839, and 834 us,
respectively, and violate the exploratory 760-us deadline. Pausing the
background does not restore SM capacity withheld by the producer's fixed MPS
percentage. The q100/q100 candidate instead records 717.9-us p99, zero misses,
and 247.9 background requests/s. Its conservative sum-of-stage reservation is
747.6 us, leaving 12.4 us. The selected-plan artifact is
`results/p9-quiet-quota-selection-250rps-v2-20260809/plan.json`. This is still a
100-request mechanism sweep, not a statistical result, but it establishes that
QUIET must keep critical contexts at full quota and control background work by
admission and temporal handback rather than by permanently reducing the
critical MPS quota.

Release lead-time is not free. The planner now requires an explicit critical
lookahead and charges any uncovered part of the drain to arrival-to-completion
latency. An initial 100-RPS run exposed a protocol bug: a sleeping pressure
worker deferred its cooperative ACK until its next release, producing 9.4-ms
drain p99. The worker now acknowledges SIGUSR1 immediately while idle and then
continues sleeping after resume. In the fresh v2 runs, drain p99 is 879.4 us at
100 RPS and 952.3 us at 250 RPS. With the declared 1,000-us lookahead, QUIET is
arrival-bound feasible with 0/100 misses at both loads; MIG and MPS miss 14--46
requests. Unpredictable arrivals still must include the full drain in their
deadline and cannot use this scheduled-release result. Evidence is in
`results/p9-dependent-frontier-idle-ack-v2-20260809`.

The first same-workload transport smoke is also complete. For the 14.7-KiB
edge, registered direct binding reduced end-to-end p99 from 728.5 us
(pageable bounce) and 734.0 us (pinned bounce) to 690.0 us. Its handoff p99 was
114.9 us, so no tail-latency superiority claim is made. Same-instance MPS was
697.4 us end-to-end p99 without competing load. These values only establish
the comparison harness; the next experiment adds controlled background demand.

The large-edge smoke moves the actual 2.304-MiB Whisper tensor for 100 requests
with zero checksum failures, four distinct payload checksums, and four distinct
projection outputs. Full per-request checksums cost approximately 2.7 ms on
each side and are reported separately from the data path. Excluding that
validation instrumentation, registered direct binding records 1,547-us
pipeline p99 and 7.95-us edge p99. Pinned and pageable bounce record 1,616 and
1,617 us pipeline p99, with 102.7 and 104.1 us edge p99. Same-instance MPS is
not faster than cross-MIG registered binding in this smoke (8.32-us edge p99,
1,556-us pipeline p99). The result explains why MIG dependency need not be
slow on Thor: device allocations remain isolated, but each MIG context maps
the same coherent SoC system-memory pages, avoiding explicit D2H+H2D copies.
Evidence is in `results/p9-mig-trt-whisper-pipeline-smoke-20260809/summary.json`.

At an exploratory 1,620-us deadline, the stage-DAG planner selects only the
registered cross-MIG candidate. Its conservative p99 stage sum is 1,556 us;
the pinned and pageable candidates reserve 1,640 and 1,646 us. The selection
artifact is `results/p9-quiet-whisper-transport-selection-20260809/plan.json`.
The deadline is smoke-only and must be independently frozen before a paper
claim.

With a 250-request/s DistilBERT tenant on the producer's 1g instance, the same
large DAG exposes the execution-interference limit. The original 100-request
smoke incorrectly left output-checksum time inside its otherwise
validation-excluded deadline. The corrected metric excludes producer-payload,
consumer-payload, and output checksum instrumentation while preserving wall
latency separately. In a fresh 1,000-request comparison, NVIDIA MIG misses
550 requests (2,226-us p99), NVIDIA MPS misses 586 (2,244-us p99), and plain
process-stop misses 125 (2,061-us p99). QUIET misses 0 with 1,574-us p99 while
retaining 249.93 background requests/s. Evidence is in
`results/p9-dependent-whisper-corrected-four-system-1000r-250rps-20260809-v1`.

Request-level traces also resolve the protection scope. Three fresh
producer-only QUIET repetitions record 0/3,000 misses, 1,565--1,582-us p99,
and 249.87--249.93 background requests/s. Holding the tenant quiet through the
consumer stage records 1/3,000 misses and only 165.27--167.14 background
requests/s. The measured DAG therefore selects producer-only protection for
this workload; full-pipeline protection remains a fallback when a future
profile cannot reserve the consumer-stage slack. These are smoke repetitions,
not a confidence-qualified SLO result.

The full same-contract SOTA smoke is hash-bound in
`results/p9-published-sota-dependent-smoke-manifest-bound-20260810/summary.json`;
its comparator manifest SHA is recorded in the output. BOER and BLESS are
structural-only, Orion is differential-pending, XSched is native-runtime
verified, and only rows marked numeric-eligible can enter a frontier.
BOER's pinned Bayesian search measures no feasible complementary-MPS point;
its q90 producer candidate still has 2.067--2.080-ms p99. ParvaGPU's pinned
configurator rejects the isolated 1g Whisper profile under its native
headroom rule. Orion observes 259 TensorRT `cuLaunchKernelEx` calls and zero
compute calls visible to its interceptor, so no numeric Orion result is
fabricated. These negative results are paired with their independent-workload
positive controls to distinguish a faithful unsupported case from a broken
port.

Three complete four-policy repetitions were replayed from request CSVs. The
pooled p99/observed DMR are 2,214.6 us/55.13% for NVIDIA MIG, 2,250.0 us/59.07%
for NVIDIA MPS, 2,059.2 us/13.23% for process-stop, and 1,569.6 us/0.033% for
QUIET. QUIET has one miss in 3,000 requests, caused by a single 1,683.9-us
producer execution rather than the 15.5-us tensor edge. This observed DMR is
below 0.05%, but the sample is explicitly not confidence-qualified. The
trace-replayed aggregate is
`results/p9-dependent-whisper-repeated-3000r-250rps-20260809/summary.json`.

The exploratory threshold has now been replaced for subsequent runs. A
separate background-free calibration collected five 1,000-request blocks,
replayed every request trace, and froze `1.10 x pooled p99`: 1,546.651 us x
1.10 = **1,701.316 us**. The source- and engine-bound lock is
`results/p9-whisper-pipeline-deadline-calibration-5x1000-20260809/deadline-lock.json`.
At this locked deadline, three new repetitions give NVIDIA MIG 1,431/3,000
misses (2,199.0-us pooled p99), NVIDIA MPS 1,630/3,000 (2,237.7 us),
process-stop 320/3,000 (2,064.4 us), and QUIET 0/3,000 (1,576.9 us, 1,609.9-us
maximum). All retain approximately 249.9 background requests/s.

BOER and ParvaGPU consume the same lock SHA. BOER's pinned search still has no
feasible point (q75/q90 candidates are approximately 2.08 ms), while ParvaGPU
still has no native-admission-feasible 1g Whisper profile. The frozen
six-system smoke and trace-replayed repetition aggregate are
`results/p9-dependent-whisper-frozen-six-system-smoke-20260809/summary.json`
and
`results/p9-dependent-whisper-frozen-repeated-3000r-250rps-20260809/summary.json`.
This removes post-hoc deadline selection but is still a smoke campaign: five
calibration blocks and three evaluation repetitions do not certify the 0.05%
DMR target with a confidence bound.

The first frozen offered-load frontier uses a declared 1,000-us release
lookahead and 250, 500, and 800 offered background requests/s. QUIET is
arrival-bound feasible at all three points: it serves 249.93, 499.94, and
532.19 background requests/s with zero misses in each 1,000-request smoke.
At 800 offered RPS, NVIDIA MIG and NVIDIA MPS serve approximately 800 RPS but
miss every critical request; process-stop serves 509.22 RPS and misses 553.
Thus 800 RPS is not a valid higher-goodput baseline point under the SLO. The
hash-bound frontier is
`results/p9-dependent-whisper-frozen-frontier-250-500-800rps-20260809/summary.json`.
Additional QUIET-only probes at 550 and 600 offered RPS both serve about 535
RPS with zero misses, locating the current handback-limited saturation point.

The two useful frontier points were then repeated with the four-treatment
Williams order so that every measured system occupies every ordinal position
once. A fresh 4 x 1,500-request campaign at 500 offered RPS gives NVIDIA MIG,
NVIDIA MPS, and the process-stop ablation 5,562, 5,925, and 3,079 misses.
QUIET misses 0/6,000, records a 1,572.94-us pooled p99, and serves 499.95
background requests/s. At 600 offered
RPS, the corresponding miss counts are 3,998, 4,000, 2,055, and 1; QUIET serves
533.58 background requests/s with a 1,578.55-us pooled p99. Its sole miss is a
1,898.63-us producer-compute outlier rather than an edge-transfer failure.

The 500-RPS smoke now meets the exact-binomial statistical criterion: its
one-sided 95% Clopper--Pearson DMR upper bound is 0.049916%, just below the
0.05% target. The 600-RPS row remains exploratory because its 1/4,000 result
has a 0.1185% upper bound. This is still not the final thermal/formal claim.
The trace-replayed aggregates are
`results/p9-dependent-whisper-frozen-williams-500rps-4x1500-20260809/summary.json`
and
`results/p9-dependent-whisper-frozen-williams-600rps-4x1000-20260809/summary.json`.

## Defensible paper claim

The target claim is not that QUIET invents MIG, MPS, or process gating. It is:

> QUIET provides isolation-preserving dependent inference across fixed MIG
> instances by combining a measured cross-instance data plane with
> communication-aware slack placement, while existing spatial schedulers model
> services as independent workloads and opaque-runtime schedulers cannot expose
> the required tensor edge.

This claim is valid only if the actual payload path improves the SLO-goodput
frontier over host bounce and same-instance sharing. If the Thor coherent path
does not outperform those baselines, the MIG-dependent contribution must be
dropped rather than hidden behind token-only results.
