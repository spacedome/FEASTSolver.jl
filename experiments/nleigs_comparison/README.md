# NLEIGS Comparison Experiment

This experiment compares the nonlinear FEAST-Beyn implementation against
NEP-PACK's NLEIGS implementation on paper-style NEP problems. It is intentionally
not part of `benchmark/` or the automated tests: algorithm comparisons for NEPs
depend strongly on target region, stopping criteria, subspace size, interpolation
choices, and available parallelism.

Run the default small butterfly case:

```sh
just experiment-nleigs
```

Useful environment variables:

- `FEAST_EXPERIMENT_PROBLEMS=butterfly`, `gun`, `loaded_string`, `hadeler`, `pep0`, or `pep0_sym`
- `FEAST_EXPERIMENT_METHODS=feast,nleigs`
- `FEAST_EXPERIMENT_PROCS=0,2,4`
- `FEAST_EXPERIMENT_FORMAT=pretty` or `csv`
- `FEAST_EXPERIMENT_PROBLEM_N=500`
- `FEAST_EXPERIMENT_BLAS_THREADS=1`
- `FEAST_EXPERIMENT_WORKER_BLAS_THREADS=1`
- `FEAST_EXPERIMENT_HADELER_ALPHA=100`
- `FEAST_EXPERIMENT_CONTOUR_CENTER='10+0im'`
- `FEAST_EXPERIMENT_CONTOUR_RADIUS=2`
- `FEAST_EXPERIMENT_M=30`
- `FEAST_EXPERIMENT_FEAST_CONFIGS='label=m30-n16,m=30,nodes=16,iter=3,store=true;label=m40-n16,m=40,nodes=16,iter=3,store=true'`
- `FEAST_EXPERIMENT_WARMUP=true`
- `FEAST_EXPERIMENT_DISTRIBUTED_WARMUP=true`
- `FEAST_EXPERIMENT_FEAST_NODES=16`
- `FEAST_EXPERIMENT_FEAST_ITER=3`
- `FEAST_EXPERIMENT_FEAST_STORE=true`
- `FEAST_EXPERIMENT_FEAST_MATERIALIZE_NODES=false`
- `FEAST_EXPERIMENT_NLEIGS_MAXIT=100`
- `FEAST_EXPERIMENT_NLEIGS_MINIT=20`
- `FEAST_EXPERIMENT_NLEIGS_MAXDGR=100`
- `FEAST_EXPERIMENT_NLEIGS_BLKSIZE=20`
- `FEAST_EXPERIMENT_NLEIGS_STATIC=false`
- `FEAST_EXPERIMENT_NLEIGS_LEJA=1`
- `FEAST_EXPERIMENT_NLEIGS_REUSEFACT=0`

`FEAST_EXPERIMENT_PROCS=0` means serial `nlfeast!`; positive values use
`distributed_nlfeast!` with that many local Julia workers.

The default NLEIGS factorization reuse is disabled so no-store FEAST and NLEIGS
are compared under the same large-problem memory assumption. If FEAST is run
with `store=true`, set `FEAST_EXPERIMENT_NLEIGS_REUSEFACT=1` for a comparable
stored-factorization comparison.

The distributed FEAST experiment also defaults to not materializing all contour
matrices on the master. Workers build `T(z)` locally, which is the fairer mode
for large no-store experiments. Set `FEAST_EXPERIMENT_FEAST_MATERIALIZE_NODES=true`
if a local callable cannot be serialized to workers.

Distributed FEAST warmup is enabled by default when `FEAST_EXPERIMENT_WARMUP=true`.
It runs a small same-problem distributed solve before timed runs so worker-local
JIT compilation is not counted as algorithm setup. Disable it with
`FEAST_EXPERIMENT_DISTRIBUTED_WARMUP=false` when measuring cold Julia startup.

`FEAST_EXPERIMENT_FEAST_CONFIGS` overrides the single `FEAST_EXPERIMENT_M`,
`FEAST_EXPERIMENT_FEAST_NODES`, `FEAST_EXPERIMENT_FEAST_ITER`, and
`FEAST_EXPERIMENT_FEAST_STORE` settings with a semicolon-separated list of
named FEAST configurations. Use this for comparing a few subspace/node choices
against one tuned NLEIGS baseline.

`loaded_string` is a dense scalable native NLEVP problem. The default target is
a small circle around its clean eigenvalue near `4.48`; the pole at `1.0` is
passed to NLEIGS as a singularity.

`hadeler` is a dense scalable native NLEVP problem. The default target is a
circle centered at `-25` with radius `3`, which gives a clean multi-eigenvalue
FEAST region at `n=500` with the default `m=28`.

The FEAST target is a circular contour. NLEIGS uses a polygonal target set, so
the script approximates the FEAST circle by a polygon.

The gun problem uses an action-based residual hook instead of materializing the
full nonlinear operator during residual checks. That keeps the experiment closer
to how the large problem should be evaluated while leaving the solver's default
`T(λ)` interface unchanged.

The polynomial experiments (`butterfly`, `pep0`, `pep0_sym`) pass an in-place
matrix update hook to distributed FEAST. Workers reuse their `T(z)` buffers
instead of allocating a fresh matrix from NEP-PACK at every contour node.

FEAST result lines include:

- `iterations`: total Beyn/RII iterations recorded by the solver.
- `rii_steps`: `iterations - 1`; zero means the result is just the initial
  Beyn-style contour solve.
- `beyn_only`: true when no RII refinement was performed.
- `stop`: coarse termination classification. `spurious_or_partial_converged`
  means FEAST stopped because the non-spurious residuals converged while some
  inside Ritz values remained above the `spurious` threshold.
- `trace`: compact per-iteration internal residual diagnostics.
