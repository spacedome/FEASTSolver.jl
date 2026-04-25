# Experiments

This directory is for research scripts, historical reproductions, and problem
setups that are useful but are not maintained automated tests.

The `legacy_tests/` directory contains scripts that previously lived under
`test/`. Many of them encode important nonlinear FEAST experiments or paper
figures, but they need curation before they can become deterministic tests or
benchmarks. Preserve them when reorganizing the repo; convert them deliberately
into one of:

- `test/runtests.jl` cases when they have clear pass/fail criteria.
- `benchmark/` scripts when the point is timing or scaling.
- documented examples when the point is explaining an algorithm or problem.

