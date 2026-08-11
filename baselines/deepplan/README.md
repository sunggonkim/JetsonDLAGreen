# DeepPlan Thor adapter

This directory ports the plan-selection algorithm from DeepPlan, EuroSys 2023,
upstream commit `ceb324428184bb46987fba235c7c893a0e6a48f1`. It is not a
renamed registered-memory baseline.

The adapter preserves the upstream naive, static, and dynamic plan rules. A
profile row records device execution, weight/data load, and direct-host-access
execution times. The dynamic planner begins with load-then-execute layers,
replays pipeline execution and load stalls, and converts a layer to direct host
access only when the performance gap fits the current stall and does not exceed
the paper artifact's `device + 1.5 * load` overload guard.

On Thor, the existing 2.304-MB transport profile is a positive control for the
data-plane mechanism: registered coherent access has 14.058-us p99 versus
114.041 us for a pinned bounce, so the source-pinned rule selects direct host
access. This does not make DeepPlan dependency-aware. Its planner concerns
model-layer residency and transfer; it has no stage-DAG precedence, critical
path, or end-to-end slack admission. A numeric `DeepPlan` row is allowed only
after this planner drives the common TensorRT path without QUIET gating.
