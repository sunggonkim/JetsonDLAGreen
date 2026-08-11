# Published-system ports

This directory is reserved for algorithm-preserving ports of published systems.
Local QUIET controls and ablations do not belong here and must never be renamed
after a paper.

| System | Pinned upstream commit | Port requirement |
|---|---|---|
| BOER | `df54815de3b1c9059f873a17c13f7d5203eedd3e` | Preserve Bayesian optimization, static/dynamic pruning, and objective feedback; replace A100 paths/topology and regenerate observations on Thor. |
| ParvaGPU | `5f3de1e18582b4c81896a1c3eb0e2915238dfee6` | Preserve segment configuration, demand matching, and allocation; regenerate Thor profiles and restrict candidate sizes to the fixed layout. |
| Orion | `20f9469764fb96d94ce23a8e70615196e9ce4ba1` | Use the native CUDA/cuDNN/cuBLAS interceptor and scheduler; first prove TensorRT operation visibility on Thor. |
| XSched | `bd494cb7a72958cd11900243a0798df00d856c6e` | Preserve the CUDA shim, XQueue, HPF server, and suspend/resume path; scope automatic XQueues to the measured TensorRT stream. |
| Pantheon | `1caa4321fe9f9902ffacb78978f11a32a7a62f64` | Preserve offline block/exit construction, deadline scheduler, and online block execution using the authors' TorchScript representation; retrain and accuracy-validate public-model exits because the official archive does not package weights. |

`analysis/compare_sota.py` accepts a numeric result only when it binds the
pinned commit, declares the required fidelity class, contains hashes of
regenerated Thor profiles, and exactly matches the QUIET workload contract.
Until such an output exists, the comparison status is `not-run`.

The headline runtime comparison contains NVIDIA MIG, NVIDIA MPS, Orion,
XSched, Pantheon, and QUIET. BOER and ParvaGPU are provisioning controls, not
substitute names for an online runtime scheduler.
