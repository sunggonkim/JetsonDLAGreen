# BOER Thor adapter

Pinned upstream: `TsingYiPainter/SC25_BOER` at
`df54815de3b1c9059f873a17c13f7d5203eedd3e` (Apache-2.0).

The adapter preserves six random initial probes, at most 20 observations, the
five-no-improvement stopping rule, static linear capacity pruning, dominated
point dynamic pruning, upstream expected-improvement acquisition
(`xi=0.2`), and BOER's normalized two-service objective. It replaces only
A100-specific paths, remote addresses,
MPS PIDs, and profile files with an evaluator subprocess contract.

The selected configuration must still be replayed using the complete QUIET
scenario runner. This adapter alone is not a measured baseline.

`evaluate_dependent_pipeline.py` is the payload-valid evaluator. It applies a
candidate's searched MPS percentage to the actual 1g ResNet producer, assigns
the independent DistilBERT tenant the upstream algorithm's complementary
`100-SM` share, and runs the 2g TensorRT policy consumer. Engine plans and MPS
client limits are selected separately for those two shares. The superseded
first payload smoke used the same share for both clients and is not valid BOER
evidence. The older
`evaluate_candidate.py --scenario dependent` path uses the invalid token-only
workload and must not be used for a dependent-inference result.

The independent positive control uses the same pinned search and complementary
share rule with concurrently measured TensorRT ResNet10 and DistilBERT
services. It selects q90/q10 at 500 offered requests/s per service with
1.481-ms worst p99 under a 3-ms SLO. Evidence is in
`results/p9-boer-independent-payload-search-v1-20260809/search.json`. This
positive control is required when interpreting the dependent no-feasible
result: BOER works for its intended independent-service abstraction.
The large dependent workload now uses the same TensorRT Whisper Tiny producer,
2.304-MiB payload, projection consumer, DistilBERT background tenant, and
1.620-ms validation-excluded deadline as QUIET. Fresh q10/q25/q50/q75/q90
capacity profiles are hash-bound in
`baselines/boer/specs/p9-dependent-whisper-smoke.json`. The pinned Bayesian
search evaluates hardware points with complementary MPS shares and returns no
feasible configuration: even q90/q10 records 2.067--2.080-ms p99 and 100%
deadline misses. The result is
`results/p9-boer-dependent-whisper-search-20260809-v1/search.json`.

The search was repeated against the final independently frozen 1,703.187-us
windowed pipeline deadline, with the lock path and SHA embedded in the BOER
contract. It remains infeasible. Fresh q50/q75/q90 hardware observations have
3.063/2.077/2.071-ms p99 and 100% deadline misses; the search therefore returns
no feasible configuration rather than substituting a local policy. See
`results/p9-boer-dependent-whisper-windowed-search-20260809T1238/search.json`.
Each measured observation binds the raw pipeline trace, pipeline summary, and
background result by SHA-256.

The current pipeline binary was recalibrated to 1,701.921199 us and BOER was
rebound to that exact lock. The pinned search again returns no feasible
configuration. Its best measured point is q90/r100 at 2.057633-ms p99 and 100%
DMR; q75/r200 and q50/r100 reach 2.074245 and 3.070453 ms. Evidence is in
`results/p9-boer-dependent-whisper-current-lock-search-20260810/search.json`.
