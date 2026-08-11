# GSLICE Thor port

Pinned paper contract: GSLICE, ACM SoCC 2020, DOI
`10.1145/3419111.3421284`. GSLICE has no public implementation artifact, so
this port reimplements Algorithm 1 from the paper rather than relabeling a
static MPS quota.

The port preserves the paper's 5% deadband, proportional resource update, and
MAX-MIN allocation across inference functions. MPS percentages are immutable
after CUDA context creation, so each adjustment uses a prewarmed process
restart, matching GSLICE's shadow-IF switchover at an epoch boundary. The
common experiment uses the frozen p99 deadline as the latency SLO and the
offered background request rate as the throughput demand.

`run_thor.py` records every tuning round and then runs a disjoint evaluation at
the converged allocation. It invokes the common payload-valid TensorRT
pipeline; the public result label is `GSLICE`.
