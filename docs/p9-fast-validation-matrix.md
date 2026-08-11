# QUIET Fast Validation Matrix

This is the short feedback loop before any thermal-normalized formal campaign.
It is deliberately small enough to rerun after every binary or workload change.

## Names and rows

The only proposed-system name is **QUIET**. The comparison inventory contains
published functional rows, vendor controls, and one proposed row:

| Class | Row | Role |
|---|---|---|
| Vendor | NVIDIA MIG | fixed 2g+1g physical isolation; no BE goodput when both slices are reserved |
| Vendor | NVIDIA MPS | fixed topology with explicit SM-quota sharing |
| SOTA | Orion (Thor port) | upstream operation-aware scheduling, only after differential gate |
| SOTA | XSched (Thor port) | upstream XQueue/HPF suspend-resume, only after native gate |
| SOTA | Pantheon (Thor port) | offline block/exit plus online deadline runtime, only after accuracy gate |
| SOTA context | EdgeServing (literature-only) | time division, batching, and early exit; numeric port requires accuracy-equivalent adapter |
| Proposed | QUIET | dependent-DAG placement, reservation, and cooperative quiescence |

The **active exploratory smoke matrix** is intentionally smaller:
`NVIDIA MPS`, `XSched (Thor port)`, and `QUIET`. MIG is a capacity/isolation
oracle. XSched is not currently numeric-eligible: its native path is verified,
but the shared learned-workload accuracy, thermal, and session gates still
remain. Orion and Pantheon are retained in the inventory but remain
functional-only until their native differential and application-accuracy gates
pass. A row can be named in the inventory without being eligible for numeric
ranking.

For learned workloads, the active launcher fails before creating a result
directory or starting a GPU child unless `APPLICATION_ACCURACY_GATE` is a
passed gate at the frozen accuracy floor and its dataset manifest matches
`COMMON_WORKLOAD_CONTRACT`. Checksum-only or stale accuracy evidence cannot
start a comparison.

gpulet, BOER, ParvaGPU, BLESS, and EdgeIso remain structural or intended-domain
controls until their common-workload fidelity gates pass. They must not be
shown as ordinary numeric competitors when a planner is infeasible or a local
reimplementation is being used.

## Three paired workloads

Every row uses the same active MIG UUIDs, engine hashes, request timestamps,
deadline lock, and correctness oracle:

1. **Independent:** ResNet10 and DistilBERT run concurrently with no edge.
2. **Dependent-small:** ResNet10 `Layer7_cov` (14,720 bytes) feeds the control
   TensorRT stage across the 1g to 2g boundary.
3. **Dependent-large:** Whisper Tiny `last_hidden_state` (2,304,000 bytes)
   feeds the 2g projection stage; DistilBERT is the independent pressure tenant.

The independent/dependent comparison changes only the edge and precedence
contract. It does not change model, offered load, placement, deadline, or
arrival trace. This is the causal motivation experiment; changing all of these
at once is not a valid dependency claim.

## Two-speed execution

The dev loop runs 100--300 requests per row, with production-wall latency and
inline checksum validation in a paired correctness run. A second performance
run uses the same binary, engines, and trace with checksum instrumentation off;
it is marked performance-only and cannot enter a formal aggregate alone.

For each point record:

- wall arrival-to-completion p50/p99/p99.9 and deadline misses;
- checksum mode, unique payload/output checksums, and correctness status;
- topology and `numeric_comparison_allowed` contract;
- best-effort goodput and offered load;
- stage compute, edge transport, gate/drain, and output-verification costs.

Only a row with correctness, a faithful-port gate, and CP95 DMR qualification
can enter the same-SLO goodput frontier. A pure MIG capacity control and an
infeasible gpulet planner remain useful negative evidence but are excluded from
that frontier.

## Immediate decision gates

1. Re-run the three workloads with QUIET, NVIDIA MPS, and one faithful SOTA
   port before starting a six-sequence Williams campaign.
2. Require at least two measured QUIET quota/placement candidates before using
   words such as “planner search” or “communication-aware selection”; one
   candidate is reported as characterization only.
3. If a SOTA port fails its native/differential or accuracy gate, report
   `not-comparable` and retain its positive control instead of substituting a
   local policy under the paper's name.
4. Freeze the current wall binary, engines, topology, and request traces only
   after this matrix passes. Then run session-balanced Williams repetitions and
   thermal normalization.

The first current three-arm aggregate is
`results/p9-fast-active-comparator-20260810/summary.json`. It is deliberately
not ranked: all three rows share the lock and correctness contract, but the
100-request smoke has no session-level confidence, application-accuracy gate,
or thermal normalization. It is the required input sanity check before the
load frontier, not the final SOTA result.
