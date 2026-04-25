# FEASTSolver

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://spacedome.github.io/FEASTSolver.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://spacedome.github.io/FEASTSolver.jl/dev)
[![Build Status](https://travis-ci.com/spacedome/FEASTSolver.jl.svg?branch=master)](https://travis-ci.com/spacedome/FEASTSolver.jl)
[![Codecov](https://codecov.io/gh/spacedome/FEASTSolver.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/spacedome/FEASTSolver.jl)
[![Coveralls](https://coveralls.io/repos/github/spacedome/FEASTSolver.jl/badge.svg?branch=master)](https://coveralls.io/github/spacedome/FEASTSolver.jl?branch=master)

-----

This is a Julia research implementation of FEAST and related contour/subspace
eigensolvers. It is not meant to replace the reference FORTRAN implementation:
the primary goal is to keep the algorithms easy to inspect, modify, and compare
while preserving the performance-critical structure needed for serious numerical
experiments.

The maintained automated tests live in `test/runtests.jl`. Historical research
scripts and old experiment drivers live under `experiments/legacy_tests/` until
they are curated into tests, benchmarks, or examples.

Tests use `TestItems.jl`/`TestItemRunner.jl` under Julia's normal `Pkg.test()`
entrypoint. Run `just test REGEX` to execute only matching test item names, and
`just test-slow` to include slow tagged items such as the gun cavity NEP case.

## API shape

The first-class dense serial solvers are `feast!`, `gen_feast!`,
`dual_gen_feast!`, and the canonical nonlinear prototype `nlfeast!`. Pass a
`DenseFeastStats()` object as `stats=` when iteration diagnostics and phase
timings are wanted.

Contour-parallel dense FEAST is deliberately explicit: use
`distributed_feast!`, `distributed_gen_feast!`, or
`distributed_dual_gen_feast!` for one-shot runs. Use the corresponding plan type
when workers should retain fixed contour-node ownership and cached workspace
across runs.

Experimental routines remain in-tree for research, but are not part of the
normal exported interface; call them as qualified `FEASTSolver.*` names if you
need to inspect or resurrect them.
