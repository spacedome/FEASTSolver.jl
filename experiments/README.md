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

The `nleigs_comparison/` directory contains a current research experiment for
comparing nonlinear FEAST against NEP-PACK's NLEIGS implementation on problems
from `article.tex`. It is not an automated benchmark because the comparison
depends on algorithm-specific parameters and target-region choices.

The `moment_rii/` directory tracks the higher-moment NLFEAST question from
`article.tex`: how to apply RII to SS/Beyn-Hankel moment expansions without
letting the active subspace grow by a factor of the moment count each iteration.

Research orientation:

- `src/nlfeast.jl` contains the canonical nonlinear FEAST-Beyn hybrid prototype.
- `src/nlfeast_experimental.jl` contains higher-moment experiments related to
  Beyn and Sakurai-Sugiura style methods.
- `src/feast_experimental.jl` contains unfinished IFEAST/inexact-FEAST work.
- `article.tex` is the main local reference for the nonlinear contour-method
  research framing.
