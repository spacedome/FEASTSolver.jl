# Moment-NLFEAST Implementation Rungs

This note records implementation boundaries for moving the experiment toward a
real solver. It is not a public API plan yet.

## Current Experiment Boundary

The candidate algorithm is expressed through these experiment objects:

- `ContourChart`: target chart and component scaling policy.
- `MomentBasisConfig`: initial contour-moment trial/test space construction.
- `TrialSpaces`: physical right/left spaces `X,Y`.
- `ReducedExtractorConfig`: determinant, counted SS/Hankel, or Loewner
  extraction of `Y' * T(lambda) * X`.
- `ResidualUpdateConfig`: residual Laurent update of `X,Y`.
- `CountDrivenPolicyConfig`: chart cover, support, count, and radius-ladder
  policy.
- `CountDrivenNumericsConfig`: numerical realization/extraction/update choices
  lowered into the basis, extractor, and residual-update configs.

`pipeline.jl` contains the chart/basis/extractor/update objects. `policy.jl`
contains the count-driven policy and numerics objects plus the named
count-stressed chart refinement rule. `experiment_matrix.jl` contains the
diagnostic runners that exercise those objects.

The sparse linear smoke confirms that generic sparse `Tmatrix` and sparse
`Tsolve` can pass through this pipeline. The sparse nonlinear gallery smoke
confirms that a FEAST-native sparse polynomial operator with an in-place
materializer also passes through the same moment pipeline. The stored-factor
sparse smoke adds the next rung: contour-node sparse factorizations can be
cached and reused across residual-Laurent updates while reproducing the generic
sparse update.
The sparse remote stored-factor smoke pins the same idea across persistent
Julia workers: each worker owns a fixed contour-node subset and reuses its
node-local sparse factors across repeated updates. These are still small
control problems, not sparse benchmarks.

## Sparse Rung

The next sparse step is not a new algorithm. It is an implementation of the same
operator boundary with reusable sparse storage:

1. Accept `T_update!(Tz, z)` and a sparse prototype, matching the existing
   nonlinear FEAST gallery path.
2. Reuse symbolic factorization when `store=false` and the sparsity pattern is
   fixed. The current stored-factor smoke only caches complete sparse
   factorizations at fixed contour nodes; symbolic-only reuse remains open.
3. Store node-local sparse factorizations when memory allows.
4. Keep reduced extraction dense; only contour solves and residual materializers
   should be sparse.
5. Add stats for materialization time, factorization/solve time, residual-rank
   RHS count, and retained algebraic/geometric counts.

The residual-Laurent update is favorable here because it solves against
low-rank residual bases `U_X,U_Y`, not every scalar Ritz vector when those
directions are redundant.

## Distributed Rung

The existing distributed canonical NLFEAST implementation already has the right
ownership model:

- workers own stable subsets of contour nodes;
- workers own node-local matrices/factors/work buffers;
- the master owns reduced compression, Ritz extraction, residual checks, and
  stopping policy;
- BLAS threading is set per worker to avoid oversubscription;
- sparse no-store mode can reuse symbolic factorization when a fixed sparse
  pattern is supplied.

Moment-NLFEAST should reuse that model, but the worker step must be generalized.
Canonical NLFEAST workers currently return `Q0/Q1` blocks for a scalar RII
state. Moment-NLFEAST workers need to accept low-rank residual bases and return
Laurent moment blocks:

```text
right worker input:  U_X, k = 1:K_update
right worker output: sum_nodes zeta^(-k) T(z)^(-1) U_X dz

left worker input:   U_Y, k = 1:K_update
left worker output:  sum_nodes zeta^( k) T(z)^(-H) U_Y dz
```

The master then concatenates those partial blocks with the existing `X,Y`,
compresses the physical spaces, and runs the reduced extractor again.

The partition diagnostic now verifies this algebra without process machinery:
four disjoint contour-node partitions return partial residual-Laurent blocks
whose sum reproduces the serial update to roundoff-level projection gaps and
the same recovered target roots. That makes the remaining distributed work an
ownership/workspace implementation problem, not a new numerical update.
The serial update and partition diagnostic now call the same
`residual_laurent_moment_blocks_generic` kernel and the same candidate-space
compression helper, so the experiment has one node-local accumulation boundary
to map onto workers.
`run_remote_residual_laurent_worker_diagnostic` is the first process-level
prototype: it stores the operator closures and contour-node subsets on actual
Julia worker processes, sends only the current low-rank residual bases for each
update, and the master reduces the returned Laurent blocks before running the
same compression step. The diagnostic now reports lightweight setup, serial
update, remote update, and worker-local elapsed times so we can see where the
prototype spends time without treating this small control as a benchmark. This
is still an experiment diagnostic, not a public or optimized sparse plan.

## What Not To Do

- Do not distribute reduced extraction first. The reduced NEP is small and is
  master-owned in the current model.
- Do not serialize `T(z)` every iteration. Either serialize `T` once to each
  worker, materialize `T(z)` worker-local, or use `T_update!` with a fixed
  prototype.
- Do not expose workspaces as a required public API while the experiment is
  still moving. Allocate them inside a plan object, mirroring the existing
  distributed FEAST plans.
- Do not conflate generic sparse compatibility with sparse performance. The
  sparse smoke test only proves the abstraction boundary.

## Immediate Prototype Target

The first distributed moment prototype should be a dense or sparse linear
operator with known eigenvalues, not a difficult NEP:

1. Build initial `X,Y` on the master using the existing basis config.
2. Run reduced extraction on the master.
3. Form low-rank residual bases.
4. Send those bases to persistent contour workers.
5. Reduce worker Laurent moments back to the master.
6. Verify the result matches the pinned partition diagnostic and the serial
   residual-Laurent update on the same problem.

The current remote diagnostic completes steps 1-6 for a dense analytic control
and keeps worker-local operator data and node assignments alive across two
residual-update calls. It also records per-stage timing metadata as a
profiling smoke check. The next implementation step is to add worker-local
buffers and sparse/factorization storage, then benchmark against the serial and
one-shot remote paths.
`run_sparse_stored_factor_residual_laurent_smoke` verifies cached contour-node
sparse factorizations reproduce the generic sparse residual-Laurent update and
that a repeated update reuses the same node factors.
`run_sparse_remote_residual_laurent_worker_smoke` also verifies that sparse
linear operator closures pass through the same persistent worker boundary and
records the same lightweight timing shape.
`run_sparse_nonlinear_gallery_moment_pipeline_smoke` verifies a sparse
quadratic polynomial gallery operator with known roots on the generic nonlinear
sparse pipeline. `run_sparse_symbolic_reuse_residual_laurent_smoke` verifies
the no-store fixed-pattern rung: one sparse factor per side is initialized
once, refreshed numerically with `reuse_symbolic=true` across contour nodes,
and still reproduces the generic sparse residual-Laurent update. The same
smoke now reuses one dense solve-result buffer per side across repeated
updates, which is the first explicit sparse workspace-reuse check in the moment
experiment.
`run_sparse_remote_stored_factor_worker_smoke` combines the linear sparse and
remote rungs: a
sparse linear control runs through persistent worker-owned contour partitions,
the first update creates the expected node-local sparse factors, and the second
update reuses those factors while matching the serial residual-Laurent update.
`run_sparse_nonlinear_remote_stored_factor_worker_smoke` repeats that
stored-factor worker path on the nonlinear sparse quadratic gallery control.
It does not yet provide full reusable sparse workspaces, broad realistic sparse
gallery coverage, or benchmark-level performance.

The first realistic sparse gallery smoke is now
`run_sparse_schrodinger_moment_gallery_smoke`: it uses the moving-boundary
Schrodinger sparse gallery operator, verifies the target count by the
full-operator argument-principle estimator, and checks residual-Laurent repair
on the small `n=128` instance. This is still a correctness/diagnostic rung, not
a sparse performance benchmark.
`run_sparse_schrodinger_remote_stored_factor_worker_smoke` runs the same
realistic sparse control through persistent worker-owned contour factors and
verifies a repeated update reuses the same worker-local factors and
node-local solve buffers. The next prototype should broaden this to larger and
less benign sparse gallery problems.

`just bench-moment` is the first BenchmarkTools-backed harness for this rung.
It times the small Schrodinger serial path and the persistent remote
stored-factor path after an untimed warmup and prints the correctness/reuse
counters beside timing and allocation summaries. This is a local regression and
profiling harness, not publication-level scaling evidence. On the current small
problem it deliberately exposes that remote/process overhead can dominate the
sparse solve work.

The argument-principle estimator now accepts sparse `T'(z)` outputs by
densifying only the derivative right-hand side before the trace solve. This
keeps the contour matrix `T(z)` sparse while avoiding SparseArrays' unsupported
`sparse_factor \ sparse_rhs` path in the diagnostic count.
