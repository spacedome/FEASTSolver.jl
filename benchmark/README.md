# Benchmarks

Benchmarks use a separate Julia environment so measurement tooling does not
become a package dependency. Run through `nix develop` or the Justfile recipes.

## Dense Distributed Harness

```sh
just bench-dense
```

Useful environment variables:

- `FEAST_BENCH_VARIANTS=standard,generalized,nonlinear`
- `FEAST_BENCH_PROCS=1,2,4,8`
- `FEAST_BENCH_N=2048`
- `FEAST_BENCH_M=16`
- `FEAST_BENCH_NODES=16`
- `FEAST_BENCH_ITER=3`
- `FEAST_BENCH_STORE=true`
- `FEAST_BENCH_MATRIX=perturbed_hermitian`
- `FEAST_BENCH_SAMPLES=1`

Output is line-oriented and CSV-like. `trial` lines come from
BenchmarkTools. `stats` lines come from FEAST's internal distributed phase
timers and are usually more useful for understanding bottlenecks.

For large distributed runs, prefer `FEAST_BENCH_SAMPLES=1` or `2`. Repeating a
large solve many times is rarely worth the wall-clock cost until the benchmark
case is narrowed down.

The dense benchmark defaults to stored shifted factorizations because that is
the realistic direct FEAST regime when the contour is fixed across iterations.
Set `FEAST_BENCH_STORE=false` to measure the lower-memory path that refactors
each shift on every filter step.

BenchmarkTools reports master-process timing and allocation data. For
distributed solves, use FEAST's `stats` phase timings to separate master costs
from worker-step costs. Profiling inside worker kernels will need explicit
worker-side instrumentation once a specific bottleneck is isolated.

## Sparse FEAST / UMFPACK Harness

```sh
just bench-sparse
```

Useful environment variables:

- `FEAST_SPARSE_BENCH_GRID=45`
- `FEAST_SPARSE_BENCH_M=16`
- `FEAST_SPARSE_BENCH_NODES=16`
- `FEAST_SPARSE_BENCH_ITER=2`
- `FEAST_SPARSE_BENCH_STORE=false`
- `FEAST_SPARSE_BENCH_SAMPLES=3`

This harness separately times sparse shifted-matrix construction, fresh
`lu(A - zI)`, reusable-pattern materialization, `lu!` with
`reuse_symbolic=true`, factored solves, and the full sparse `feast!` path. It is
intended to catch accidental symbolic-analysis or sparse-structure allocation
inside the contour loop.
