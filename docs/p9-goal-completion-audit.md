# P9 goal-completion audit

Status: **complete for the bounded QUIET claim described in the paper**.

The authoritative machine audit is
`analysis/audit_p9_goal_completion.py`.  On the measurement host it verifies
the raw artifact graph, native comparator pins, application outputs, thermal
admission, paper contract, and compiled PDF.  The public repository retains
the audit implementation and compact publication outputs; the 34-GB raw/model
corpus remains local.

## Completed gates

- A single public proposed-system name: QUIET.
- Full activation payloads for the promoted ImageNette path and the independent
  Whisper-Tiny application gate.
- Operational, non-acknowledgement-paced arrivals and an
  arrival-to-consumer-completion production wall.
- Post-completion output validation that does not inflate service latency.
- Request-indexed same-activation replay for the causal control.
- A fixed hash-bound 1g producer / 2g consumer plan with joint-tail admission.
- Native XSched execution in the complete thermal campaign.
- A separate native Pantheon application/fidelity gate.
- Six counterbalanced, thermally admitted ImageNette sessions.
- Exact one-sided Clopper--Pearson DMR qualification from 6,600 requests per
  system.
- Request-level Type-7 percentile replay and figure/table SHA-256 provenance.
- A warning-free ten-page ACM paper with no unresolved citations or
  references.

## Formal outcome

At the frozen 2,255.483-us deadline, QUIET completes 6,600/6,600 requests with
zero misses, 1,902.987-us p99, and a 0.0454% one-sided 95% DMR upper bound.
NVIDIA MPS records two misses and a 0.0954% bound; XSched misses all requests.
QUIET's paired p99 reduction versus MPS is 138.508 us, while the paired
goodput interval spans zero.

## Required boundaries

Completion does not promote every local artifact.  The offered-load sweep is
descriptive, the causal and transport controls are exploratory, Pantheon uses
a separate adapter contract, Orion is nonnumeric, and neither the external
ring nor larger-DAG schema is a production multi-inflight/DAG claim.

## Repository checks

The publication commit is accepted only after all of the following succeed:

```bash
python3 analysis/generate_p9_current_figures.py
python3 -m pytest -q
ctest --test-dir build-r39 --output-on-failure
```

The paper is built with one BibTeX and three pdfLaTeX passes.  Build logs are
checked for warnings, errors, undefined citations/references, and overfull or
underfull boxes.  Generated and raw evidence remains fail-closed on SHA-256,
request-count, miss-count, deadline, application, and thermal mismatches.
