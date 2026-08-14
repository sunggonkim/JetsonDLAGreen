# EuroSys 2027 Draft Workspace

This folder contains the ACM acmart review draft for QUIET.  The current
dependency-aware Thor manuscript is p9-main.tex; main.tex is the older no-MIG
manuscript and must not be used as the current result.

## Files

- p9-main.tex: current top-level manuscript.
- p9-sections/*.tex: current section files.
- refs.bib and p9-refs.bib: checked bibliographies.
- figures/p9-*.pdf and figures/p9-*.png: generated design and evaluation
  figures.
- generated/p9-six-system-imagenette-gate.json: fixed-roster common-gate
  metrics.
- generated/p9-current-results.tex: evidence-backed application, formal,
  common-comparator, and partial-evidence tables.
- generated/p9-figure-provenance.json: input and output SHA-256 bindings.

## Generate figures

From the repository root, generate every current figure and table with:

    python3 analysis/generate_p9_six_system_figure.py
    python3 analysis/generate_p9_current_figures.py

The generator fails if a recorded input hash, request count, miss count,
application gate, thermal gate, or replayed p99 is inconsistent.

## Compile the current draft

From this directory:

    pdflatex p9-main.tex
    bibtex p9-main
    pdflatex p9-main.tex
    pdflatex p9-main.tex

The public system name is QUIET.  The current formal headline is limited to
the thermal-normalized ImageNette campaign.  Causal and load-sweep figures
retain their exploratory/descriptive labels.  Every end-to-end table and graph
repeats QUIET, NVIDIA MIG, NVIDIA MPS, XSched, Orion, and Pantheon in that
order.  Executed mechanism-only or incompatible artifacts remain visible in a
separate partial-evidence table.
