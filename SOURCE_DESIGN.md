# FEASTSolver Source Design

This note describes a cleanup direction for the main `src/` tree after the
moment-RII experiment. The goal is not to make the Julia package compete with
the Fortran FEAST implementation. The goal is to keep research implementations
clear, correct, and fast enough to run meaningful experiments without turning
the library into a maze of allocation workarounds.

## Current State

The finalized FEAST variants are mostly present and working:

- `feast.jl`: dense standard, generalized, and dual generalized FEAST.
- `sparse_feast.jl`: sparse standard and generalized FEAST, with direct and
  first-pass iterative solver policies.
- `nlfeast.jl`: canonical nonlinear FEAST/Beyn hybrid, including sparse and
  action-style operator paths.
- `distributed_feast.jl` and `distributed/`: process-parallel dense contour
  variants with persistent worker ownership.
- `fastlapack.jl`: the practical non-allocating LAPACK bridge needed because
  Julia builtins allocate too much in the hot path.
- `stats.jl` and `distributed/stats.jl`: lightweight timing and convergence
  diagnostics.
- `moment_rii.jl`, `feast_experimental.jl`, and `nlfeast_experimental.jl`:
  research code that should not drive the finalized FEAST source layout.

The main issue is that the FEAST loop is repeated in several files. Each copy
contains the same conceptual stages, but with local differences for dense,
sparse, nonlinear, and distributed solves. That makes the algorithms harder to
read and makes performance plumbing look like algorithmic complexity.

## Design Principle

Use three implementation layers:

1. **Reference layer**: direct pseudocode-style algorithms using Julia builtins
   such as `\`, `qr`, `eigen`, and materialized matrices. This layer optimizes
   for correctness, readability, and documentation.
2. **Optimized layer**: allocation-conscious production research
   implementations using explicit workspaces, `mul!`, FastLapackInterface, and
   sparse solver policies. This is the default public path.
3. **Parallel layer**: explicit process-parallel contour orchestration. This is
   optional, plan-based, and allowed to be more verbose because Julia worker
   ownership, serialization, and JIT behavior need to be visible.

The Rust experiment in `~/Code/FEAST.rs` used explicit "kernels" and stage
traits. Julia does not need to copy that structure wholesale. The useful idea is
the stage boundary, not the Rust-style trait hierarchy.

## Algorithm Stages

Every finalized FEAST variant should read as an orchestration of these stages:

1. Validate dimensions and normalize the contour/options.
2. Build or reuse workspaces.
3. Optionally pre-factor contour shifts.
4. Extract Ritz data from the current subspace.
5. Form residuals and convergence diagnostics.
6. If not converged, apply the contour filter or RII update.
7. Return inside-contour Ritz values, vectors, and residuals.

For nonlinear FEAST the first filter is Beyn-style moment extraction and later
updates use residual inverse iteration over the same contour nodes. For dual
FEAST, the extraction and update happen for paired right/left spaces. For sparse
FEAST, the reduced problems remain dense, while shifted solves use sparse
policies.

## Public API Boundary

Keep the current public algorithm names:

- `feast!`
- `gen_feast!`
- `dual_gen_feast!`
- `nlfeast!`
- `distributed_feast!`
- `distributed_gen_feast!`
- `distributed_dual_gen_feast!`
- `distributed_nlfeast!`

Add explicit reference names instead of overloading the main optimized path:

- `reference_feast!`
- `reference_gen_feast!`
- `reference_dual_gen_feast!`
- `reference_nlfeast!`

The reference variants should be documented as correctness and teaching tools.
They should be used by tests to cross-check small cases. They should not accept
every expert keyword from the optimized variants.

## Proposed Source Layout

One possible final layout:

```text
src/
  FEASTSolver.jl

  contours.jl
  stats.jl
  utils.jl
  operators/
    gallery.jl
    materialization.jl

  linalg/
    dense_lapack.jl
    sparse_shifts.jl
    sparse_solvers.jl
    residuals.jl
    reduced.jl

  reference/
    linear.jl
    nonlinear.jl

  optimized/
    linear_standard.jl
    linear_generalized.jl
    linear_dual_generalized.jl
    nonlinear.jl
    sparse_standard.jl
    sparse_generalized.jl

  parallel/
    plans.jl
    workers.jl
    linear_standard.jl
    linear_generalized.jl
    linear_dual_generalized.jl
    nonlinear.jl
    stats.jl

  experimental/
    inexact_feast.jl
    nlfeast_moments.jl
```

This exact tree is negotiable. The important boundary is that linear algebra
plumbing moves out of algorithm files, reference algorithms stay readable, and
parallel plans stay isolated from serial code.

## Naming Rules

Do not adopt BLAS/LAPACK routine names as package-level vocabulary. Use FEAST
and numerical-analysis words in algorithm code:

- `factor_shift!`, not `getrf_shift!`
- `solve_shifted!`, not `getrs!`
- `orthogonalize_subspace!`, not `geqrf!`
- `reduced_eigenproblem!`, not `geev!`
- `biorthogonalize_pair!`, not `gesdd!`

LAPACK-specific names can remain inside `linalg/dense_lapack.jl`, where they
are wrappers around FastLapackInterface and Base LAPACK calls.

## Workspaces

The optimized layer should allocate once at the top of each call, or once in an
explicit plan object. Workspaces should be named by algorithm role, not by LAPACK
implementation detail:

- `StandardFeastWorkspace`
- `GeneralizedFeastWorkspace`
- `DualGeneralizedFeastWorkspace`
- `NonlinearFeastWorkspace`
- `SparseShiftWorkspace`

These do not have to be exposed immediately. A good first step is to introduce
internal workspace constructors and pass them to small stage helpers. Public
expert workspace APIs can wait until the shape stabilizes.

The current `fastlapack.jl` helpers are useful, but they mix three concerns:
shift materialization, LAPACK workspaces, and small reduced linear algebra. Split
them eventually into:

- dense shifted-system materialization and solve helpers;
- dense reduced-problem helpers;
- generic column/residual kernels.

## Reference Layer

Reference implementations should be deliberately simple. For example, standard
linear FEAST should look roughly like:

```julia
for iteration in 0:iter
    Q = Matrix(qr(Q).Q)
    Ared = Q' * A * Q
    F = eigen(Ared)
    X = Q * F.vectors
    R = A * X - X * Diagonal(F.values)
    converged && break
    Q = zero(Q)
    for (z, w) in contour
        Q += w * (X - (z * I - A) \ R) * Diagonal(1 ./ (z .- F.values))
    end
end
```

This layer can allocate freely. It is a guardrail against optimized code
becoming unreadable. It is also the right place to document mathematical steps
and produce small correctness examples.

## Optimized Layer

The optimized serial layer should preserve clear stage boundaries without
forcing a heavyweight kernel abstraction. A practical pattern is:

```julia
workspace = StandardFeastWorkspace(A, X, contour; store, mixed_precision)
for iteration in 0:iter
    orthogonalize_subspace!(workspace)
    reduced_standard_eigenproblem!(workspace)
    update_standard_residuals!(workspace)
    record_iteration!(stats, workspace)
    converged(workspace) && break
    apply_standard_rii_filter!(workspace)
end
retain_inside(workspace)
```

Each stage can dispatch on workspace type when needed. This gives us most of the
Rust staged design without introducing Rust-style traits into Julia.

Dense optimized variants should share:

- contour-node and weight access;
- shifted matrix materialization;
- factor/solve policy;
- residual update helpers;
- stats recording;
- inside-contour selection.

Sparse optimized variants should share:

- sparse shifted-pattern workspaces;
- sparse direct and iterative solver policies;
- optional stored-factor memory accounting;
- symbolic reuse where available.

Nonlinear optimized variants should share:

- `AbstractFeastOperator` materialization/action interface;
- Beyn moment extraction;
- residual update hooks for action-only operators;
- sparse/dense shifted solve policies.

## Parallel Layer

Parallel FEAST should remain plan-based. The plan is the right abstraction in
Julia because workers need persistent ownership of:

- contour-node assignments;
- copied operator data or materializers;
- shifted factors;
- scratch buffers;
- BLAS thread settings.

Parallel code should not try to hide Julia's process model. The cost model and
failure modes are part of the research interface. The public API can stay
simple, but the implementation should keep these concepts explicit:

- master process owns reduced problems, residual checks, convergence, and stats;
- workers own contour solves;
- node assignments are fixed for the lifetime of the plan;
- worker BLAS threads are controlled to avoid oversubscription;
- setup time is measured separately from solve time.

The current `distributed_feast.jl` file is too large because it combines plan
types, plan constructors, loop orchestration, and variant-specific logic. Split
it into `parallel/plans.jl`, `parallel/linear_*.jl`, and
`parallel/workers.jl`.

## Experimental Code

The following should not be treated as finalized FEAST variants:

- `feast_experimental.jl`
- `nlfeast_experimental.jl`
- `moment_rii.jl`

They can remain in `src/experimental/` for now if tests or experiments depend on
package internals. Longer term, the moment-RII work should either become a
separate experimental module or live under `experiments/` until it is promoted.

The rule should be: finalized algorithms live in `reference/`, `optimized/`, or
`parallel/`; exploratory algorithms live in `experimental/` and are not used as
templates for source organization.

## Refactor Plan

1. **Add reference implementations.** Create small allocating reference versions
   for standard, generalized, dual generalized, and canonical nonlinear FEAST.
   Cross-check them against current optimized methods on small dense problems.
2. **Extract common optimized stage helpers.** Start with dense standard FEAST:
   validation, workspace allocation, reduced extraction, residual update, and
   RII filter. Keep behavior unchanged.
3. **Move LAPACK wrappers under `linalg/`.** Preserve the existing helper names
   initially; add clearer stage-level wrappers above them.
4. **Split sparse solve plumbing.** Separate sparse pattern/materialization and
   solver policies from sparse FEAST orchestration.
5. **Split distributed plans.** Move plan types and worker setup away from
   iteration loops. Keep existing APIs intact.
6. **Move experimental files.** Put unfinished IFEAST, old nonlinear moment
   variants, and moment-RII prototypes under `src/experimental/` or leave them
   only in `experiments/` if no public exports require them.
7. **Update tests incrementally.** After each move, run targeted tests for the
   affected variant. Do not combine source movement with algorithmic changes.

## Non-Goals

- Do not create a large public expert-workspace API before the internal
  workspace shape stabilizes.
- Do not generalize every variant into one mega-loop if that hides the algorithm.
- Do not add automatic parallelism behind serial APIs.
- Do not chase Fortran FEAST feature parity in this cleanup pass.
- Do not move the moment-RII research conclusions into the main solver API until
  the production algorithm boundary is clear.

## Immediate Recommendation

The first concrete cleanup should be adding `reference/linear.jl` and using it
in tests. That creates a clear correctness baseline. Then refactor
`feast.jl` one variant at a time, starting with dense standard FEAST, into a
workspace plus named stages. If that shape reads well, apply the same pattern to
generalized, dual generalized, sparse, and nonlinear variants.
