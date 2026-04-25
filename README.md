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
