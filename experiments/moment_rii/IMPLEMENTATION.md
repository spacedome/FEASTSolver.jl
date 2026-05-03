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

The sparse linear smoke confirms that generic sparse `Tmatrix` and sparse
`Tsolve` can pass through this pipeline, but it is not sparse optimized.

## Sparse Rung

The next sparse step is not a new algorithm. It is an implementation of the same
operator boundary with reusable sparse storage:

1. Accept `T_update!(Tz, z)` and a sparse prototype, matching the existing
   nonlinear FEAST gallery path.
2. Reuse symbolic factorization when `store=false` and the sparsity pattern is
   fixed.
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
`run_sparse_remote_residual_laurent_worker_smoke` also verifies that sparse
linear operator closures pass through the same persistent worker boundary and
records the same lightweight timing shape. It does not yet reuse symbolic
sparse factorizations or keep sparse work buffers node-local.

Only after that should the prototype move to nonlinear sparse gallery problems.
