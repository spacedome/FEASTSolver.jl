# FEASTSolver

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://spacedome.github.io/FEASTSolver.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://spacedome.github.io/FEASTSolver.jl/dev)
[![Build Status](https://travis-ci.com/spacedome/FEASTSolver.jl.svg?branch=master)](https://travis-ci.com/spacedome/FEASTSolver.jl)
[![Codecov](https://codecov.io/gh/spacedome/FEASTSolver.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/spacedome/FEASTSolver.jl)
[![Coveralls](https://coveralls.io/repos/github/spacedome/FEASTSolver.jl/badge.svg?branch=master)](https://coveralls.io/github/spacedome/FEASTSolver.jl?branch=master)

-----

This is an implementation of the FEAST eigensolver in Julia, and is not meant to replace the reference FORTRAN implementation. 
Despite this, it does aim to be performant, and should work well for large dense and sparse problems where spectral slicing is needed.

For Julia bindings to the FORTRAN implementation see FEAST.jl (coming soon).
