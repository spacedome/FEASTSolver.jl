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

Sparse standard FEAST is exposed by the same `feast!` name through dispatch on
`A::AbstractSparseMatrix`. The default `SparseDirectSolver()` uses Julia's
SuiteSparse/UMFPACK path via sparse `lu`; `SparseBiCGSTABSolver()` is available
for experiments but is not yet the tuned workspace-oriented path.

The nonlinear implementation currently exported as first-class is `nlfeast!`.
It accepts FEAST-native nonlinear operators with `operator_prototype(op)`,
`materialize!(M, op, z)`, and `mul!(Y, op, z, V)` methods, plus the same
`stats=` diagnostics hook. Moment-expanded and inexact/iterative nonlinear
experiments are kept in-tree, but are intentionally not exported as the normal
user-facing interface.

Contour handling is explicit but keeps the simple path simple. Solver wrappers
default to a circular trapezoidal contour from `c`, `r`, and `nodes`, while the
expert path accepts a `Contour` object directly. `CustomContour(nodes, weights;
inside=z -> ...)` is available for experimental quadrature rules; the `inside`
predicate is required whenever the solver must classify returned eigenvalues.

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
