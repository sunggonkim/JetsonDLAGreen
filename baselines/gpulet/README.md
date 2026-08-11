# gpulet Thor policy port

This directory ports gpulet's elastic-partitioning decision to the fixed
2g+1g Thor topology. It does not rename a local static policy as gpulet.

The adapter pins the public artifact commit, profiles every representable MPS
partition pair with the common TensorRT workload, applies gpulet's
latency/throughput/interference feasibility test and best-fit ordering, and
then evaluates the selected action on a disjoint request set. If no spatial
partition is schedulable, the result records that decision and still executes
the largest critical partition as diagnostic evidence; that execution is not
reported as a gpulet-feasible allocation.

Upstream: <https://github.com/casys-kaist/glet>, commit
`3c1c2aad3b33edcef20e549d5093c43af497e6ae` (USENIX ATC 2022).

The current ResNet-control smoke uses a frozen 770.605-us wall deadline, five
100-request profile sets, and a disjoint 100-request evaluation. No partition
is schedulable; q90/q10 is retained only as the paper-defined diagnostic and
misses 100/100 requests at 949.521-us p99. The raw and replay-verified artifact
is `results/p9-gpulet-resnet-control-5x100-eval100-20260809T142909Z/`.
