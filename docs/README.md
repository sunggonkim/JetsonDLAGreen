# Documentation map

The current public artifact is the dependency-aware P9 system described by
the QUIET paper.  Start with the following files:

- [`p9-current-status.md`](p9-current-status.md): promoted results and claim
  boundaries.
- [`p9-real-application-runbook.md`](p9-real-application-runbook.md): real
  ImageNette and LibriSpeech application preparation and validation.
- [`p9-sota-native-port-contract.md`](p9-sota-native-port-contract.md): rules
  for admitting a published system as an executable comparator.
- [`p9-comparator-manifest.json`](p9-comparator-manifest.json): pinned
  comparator source and artifact metadata consumed by the audit tools.
- [`platforms/jetson-agx-thor.md`](platforms/jetson-agx-thor.md): platform
  assumptions and device setup.

The remaining P9 plans and audits record design history.  They are retained
for traceability, but a historical smoke or nonthermal campaign does not
override the promoted scope in `p9-current-status.md` or the paper.

Raw experiment directories and downloaded model/engine binaries are omitted
from Git because they occupy roughly 34 GB and contain machine-specific
artifacts.  The paper ships generated tables, figures, and a SHA-256
provenance manifest under `paper/eurosys27/`.
