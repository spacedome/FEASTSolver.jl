# FEASTSolver.jl

FEASTSolver.jl is a research-oriented implementation of FEAST and related
contour/subspace eigensolvers. The dense serial implementations are intended to
be readable references as well as useful numerical kernels; distributed variants
make the contour-node parallelism explicit.

## Public API Boundaries

The dense serial reference family is `feast!`, `gen_feast!`, and
`dual_gen_feast!`. These share the same contour-first interface and accept
`stats=DenseFeastStats()` when iteration counts, convergence counts, residuals,
and phase timings should be recorded.

The nonlinear implementation currently exported as first-class is `nlfeast!`.
It accepts the same `stats=` diagnostics hook. Moment-expanded and
inexact/iterative nonlinear experiments are kept in-tree, but are intentionally
not exported as the normal user-facing interface.

The contour-parallel family is explicit: `distributed_feast!`,
`distributed_gen_feast!`, `distributed_dual_gen_feast!`, and their corresponding
plan types expose worker ownership, BLAS-thread policy, cached workspace, and
distributed timing statistics instead of hiding those concerns behind keywords
on the serial solvers.

```@index
```

```@autodocs
Modules = [FEASTSolver]
```
