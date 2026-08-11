# ParvaGPU Thor adapter

Pinned upstream: `MunQ-Lee/ParvaGPU_SC24` at
`5f3de1e18582b4c81896a1c3eb0e2915238dfee6` (MIT).

The adapter preserves the segment configurator's throughput-per-GPC choice,
`SLO/2 * 0.9` latency constraint, demand matching, one-to-three MPS processes,
and size-descending allocation. Profiles must be regenerated on Thor.

In the fixed comparison, the critical service reserves `2g`, leaving one `1g`
segment. ParvaGPU does not co-locate different models in one segment, so a
simultaneous audio+language workload will normally be reported as infeasible.
That limitation is an experimental result, not a reason to alter the topology.

The payload-valid spec reserves 2g for the dependent consumer and asks the
original configurator to place the ResNet producer and independent DistilBERT
tenant in the one remaining 1g segment. Their profiles are individually
SLO-feasible, but ParvaGPU does not co-locate different models in one segment,
so the fixed-layout allocator must report `insufficient fixed MIG segments`.

The v2 result rebuilds both TensorRT engines on the current 1g/q100 device and
binds each raw benchmark SHA plus the generated profile CSV SHA into the
allocator output at
`results/p9-parvagpu-dependent-profile-v2-20260809/allocation.json`.

The independent positive control profiles both 1g and 2g instances, lets the
pinned configurator allocate the fixed 2g+1g layout, and executes that
allocation at 500 offered requests/s per service. ResNet10 and DistilBERT
record 0.434-ms and 0.969-ms p99 while meeting both SLOs. Evidence is in
`results/p9-parvagpu-independent-profile-v2-20260809/` and
`results/p9-parvagpu-independent-execution-v2-20260809/`. The dependent
allocation failure is therefore a DAG/layout limitation, not a blanket claim
that ParvaGPU cannot run on Thor.

The large-edge contract is independently profiled rather than inferred from
the ResNet result. On the remaining 1g instance, the q100 Whisper producer
records 1.494-ms isolated p99. ParvaGPU's original 0.9-headroom half-SLO
admission rule therefore finds no SLO-feasible producer profile for the
1.620-ms end-to-end deadline, before it attempts to place the independent
DistilBERT tenant. The profile, manifest, and allocator decision are in
`results/p9-parvagpu-dependent-whisper-profile-20260809/`.

Re-evaluation with the final independently frozen 1,703.187-us windowed
pipeline lock reaches the same decision. The lock-bound spec and decision are
`baselines/parvagpu/specs/p9-dependent-whisper-frozen-smoke.json` and
`results/p9-parvagpu-dependent-whisper-profile-20260809/windowed-allocation.json`.

The same profile evidence has now been rebound to the current 1,701.921199-us
lock. The original configurator again returns
`no SLO-feasible ParvaGPU profile for whisper-producer`; see
`results/p9-parvagpu-dependent-whisper-current-lock-20260810/allocation.json`.
