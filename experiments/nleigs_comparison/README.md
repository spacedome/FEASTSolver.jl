# NLEIGS Comparison Experiment

This experiment compares the nonlinear FEAST-Beyn implementation against
NEP-PACK's NLEIGS implementation on paper-style NEP problems. It is intentionally
not part of `benchmark/` or the automated tests: algorithm comparisons for NEPs
depend strongly on target region, stopping criteria, subspace size, interpolation
choices, and available parallelism.

See [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) for the tuning history behind the
current defaults.

Run the full configured comparison set:

```sh
just experiment-nleigs
```

For a cheap script sanity check that does not run the large dense problem:

```sh
just experiment-nleigs-smoke
```

By default this runs the curated comparison set: `butterfly` as a small sanity
problem, `pep0` as the large dense polynomial problem, and
`schrodinger_movebc` as the current sparse large problem. Set
`FEAST_EXPERIMENT_PROBLEMS=butterfly` or another comma-separated list for a
targeted local run. `gun` remains available as a sparse exploratory comparison,
but is intentionally not part of the default set until its region and solver
parameters are tuned.

Useful environment variables:

- `FEAST_EXPERIMENT_PROBLEMS=butterfly`, `pep0`, `schrodinger_movebc`, or exploratory `gun`, `loaded_string`, `hadeler`, `pep0_sym`
- `FEAST_EXPERIMENT_METHODS=feast,nleigs`
- `FEAST_EXPERIMENT_PROCS=0,4,8`
- `FEAST_EXPERIMENT_FORMAT=pretty` or `csv`
- `FEAST_EXPERIMENT_COLOR=auto`, `always`, or `never`
- `FEAST_EXPERIMENT_REPEATS=1`
- `FEAST_EXPERIMENT_PROBLEM_N=50000` to override a problem's built-in size
- `FEAST_EXPERIMENT_BLAS_THREADS=16` by default on this machine, using `Sys.CPU_THREADS`
- `FEAST_EXPERIMENT_WORKER_BLAS_THREADS=1`
- `FEAST_EXPERIMENT_HADELER_ALPHA=100`
- `FEAST_EXPERIMENT_CONTOUR_CENTER='10+0im'`
- `FEAST_EXPERIMENT_CONTOUR_RADIUS=2`
- `FEAST_EXPERIMENT_NLEIGS_CONTOUR_CENTER='0+0im'` to tune only the NLEIGS target polygon
- `FEAST_EXPERIMENT_NLEIGS_CONTOUR_RADIUS=0.1` to tune only the NLEIGS target polygon
- `FEAST_EXPERIMENT_M=30`
- `FEAST_EXPERIMENT_FEAST_CONFIGS='label=m30-n16,m=30,nodes=16,iter=3,store=true;label=m40-n16,m=40,nodes=16,iter=3,store=true'`
- `FEAST_EXPERIMENT_WARMUP=true`
- `FEAST_EXPERIMENT_VALIDATE_T_UPDATE=true`
- `FEAST_EXPERIMENT_FEAST_NODES=16`
- `FEAST_EXPERIMENT_FEAST_ITER=3`
- `FEAST_EXPERIMENT_FEAST_STORE=true`
- `FEAST_EXPERIMENT_FEAST_MATERIALIZE_NODES=false`
- `FEAST_EXPERIMENT_NLEIGS_MAXIT=150` to override the problem-local default
- `FEAST_EXPERIMENT_NLEIGS_MINIT=20`
- `FEAST_EXPERIMENT_NLEIGS_MAXDGR=100`
- `FEAST_EXPERIMENT_NLEIGS_BLKSIZE=32` to override the problem-local default
- `FEAST_EXPERIMENT_NLEIGS_POLYGON_POINTS=32` to override the problem-local default
- `FEAST_EXPERIMENT_NLEIGS_POLYGON_PHASE=feast_nodes` or `vertices`
- `FEAST_EXPERIMENT_NLEIGS_STATIC=false`
- `FEAST_EXPERIMENT_NLEIGS_LEJA=1`
- `FEAST_EXPERIMENT_NLEIGS_REUSEFACT=auto`, `0`, `1`, or `2`

`FEAST_EXPERIMENT_PROCS=0` means serial `nlfeast!`; positive values use
`distributed_nlfeast!` with that many local Julia workers.

The default NLEIGS factorization reuse is `auto`. In auto mode, each NLEIGS run
is matched to the FEAST configuration it is printed with: `store=false` maps to
`reusefact=0`, and `store=true` maps to `reusefact=1`. Set
`FEAST_EXPERIMENT_NLEIGS_REUSEFACT=0`, `1`, or `2` only when intentionally
overriding that memory-policy match. For the large dense default, the intended
fair-memory comparison is `store=false` and `reusefact=0`; cached factorizations
are useful for diagnostics but are not the reported "too large for memory"
policy.

The distributed FEAST experiment also defaults to not materializing all contour
matrices on the master. Workers build `T(z)` locally, which is the fairer mode
for large no-store experiments. Set `FEAST_EXPERIMENT_FEAST_MATERIALIZE_NODES=true`
if a local callable cannot be serialized to workers.

Warmup is enabled by default. Each method/configuration runs the same measured
path once without recording its time, so Julia compilation and worker-local JIT
do not pollute the timed samples. This is necessary for useful steady-state
algorithm timings, but it is a full extra solve. Set
`FEAST_EXPERIMENT_WARMUP=false` only when intentionally measuring cold Julia
startup.

The run header records Julia version, CPU model/thread count, and total/free
memory at startup. CSV output includes memory in bytes for machine parsing.

`FEAST_EXPERIMENT_REPEATS` controls how many timed samples are collected after
warmup. Pretty output reports min, median, max, and the raw sample list. For
publication-style runs, prefer an odd value such as `5` or `7`.

Pretty output uses ANSI color when `FEAST_EXPERIMENT_COLOR=auto`, `TERM` is not
`dumb`, and `NO_COLOR` is unset. CSV output is never colored.

`FEAST_EXPERIMENT_FEAST_CONFIGS` overrides the single `FEAST_EXPERIMENT_M`,
`FEAST_EXPERIMENT_FEAST_NODES`, `FEAST_EXPERIMENT_FEAST_ITER`, and
`FEAST_EXPERIMENT_FEAST_STORE` settings with a semicolon-separated list of
named FEAST configurations. Use this for comparing a few subspace/node choices
against one tuned NLEIGS baseline.

The default large dense case is `pep0` with `n=3000`, contour center `0`, radius
`0.095`, `m=60`, `nodes=32`, `iter=4`, and `store=false`. On the local
16-thread machine this gives a clean FEAST region with 27 interior eigenpairs and
one RII refinement step. NLEIGS is intentionally run on the corresponding
polygonal target set with problem-local defaults `maxit=150`, `blksize=32`,
`polygon_points=96`, and the unaligned vertex phase. The unaligned 96-point
polygon is the robust fair-memory NLEIGS default found so far; an aligned
32-point polygon is geometrically tempting, but under `reusefact=0` it recovered
only 7 eigenpairs in the full benchmark.

The `pep0` region was chosen after trying larger radius `0.2` cases at `n=1200`,
which contained about 50 interior eigenvalues and required larger subspaces while
still leaving spurious/edge values for several `m`/node choices. Radius `0.1`
gave a cleaner FEAST region, but at `n=3000` NLEIGS recovered only a small subset
of that wider 32-eigenpair region. Radius `0.095` gives the cleaner
apples-to-apples comparison: under no-store/no-reuse, FEAST and NLEIGS both
recover 27 interior eigenpairs with the unaligned 96-point NLEIGS polygon.
NLEIGS checks included target radii `0.08`, `0.09`, `0.095`, `0.1`, and `0.12`;
32 and 96 polygon points; aligned and unaligned polygon phases; `blksize=32` and
`96`; `maxit=150`; `maxdgr=300`; `leja=1` and `2`; and `reusefact=0`, `1`, and
`2`. At radius `0.1`, aligning 32 NLEIGS polygon vertices with FEAST's
trapezoid-node angles improved NLEIGS to 18 eigenpairs, but it still did not
recover FEAST's wider 32-eigenpair region. At radius `0.095`, aligned 32-point
NLEIGS with `reusefact=0` recovered only 7 eigenpairs, so it is not the default.

`loaded_string`, `hadeler`, and `pep0_sym` remain available as exploratory
problem selectors, but they are intentionally not part of the default comparison
set. The current defaults did not make strong publication-style comparisons:
`loaded_string` is too small/clean, `hadeler` needs a better large-region study,
and `pep0_sym` duplicates `pep0` while triggering extra NEP-PACK representation
issues.

The FEAST target is a circular contour. NLEIGS uses a polygonal target set, so
the script approximates the FEAST circle by a polygon. The NLEIGS target can be
shifted independently with `FEAST_EXPERIMENT_NLEIGS_CONTOUR_CENTER` and
`FEAST_EXPERIMENT_NLEIGS_CONTOUR_RADIUS`; use this when FEAST has already
identified a clean cluster and the NLEIGS polygon needs a slightly different
region to recover it. `FEAST_EXPERIMENT_NLEIGS_POLYGON_PHASE=feast_nodes` rotates
the polygon by half a panel so its vertices have the same angular placement as
FEAST's midpoint trapezoid nodes.

The sparse `gun` problem is available as an explicit selector with
`FEAST_EXPERIMENT_PROBLEMS=gun`. It uses sparse NLFEAST through the in-place
FEAST gallery operator and sparse direct shifted solves. It is not yet part of
the default comparison set because the target region, node count, subspace size,
and NLEIGS settings still need the same level of tuning as `pep0`.

The default sparse comparison is `schrodinger_movebc` with `n=50000`, contour
center `-35`, radius `4.2`, `m=8`, `nodes=24`, `iter=6`, `store=false`, and an
absolute action-residual tolerance of `1e-5`. Unlike `gun`, the FEAST side is
implemented as a native gallery operator rather than a wrapper around NEP-PACK.
The region targets the three real eigenpairs near `-39.15`, `-34.94`, and
`-31.06`, while explicitly exposing nearby spurious Ritz values instead of
hiding them behind matrix-norm relative residuals. NLEIGS uses the same center
and radius with a 24-point polygon and no factorization reuse.

The polynomial experiments (`butterfly`, `pep0`, `pep0_sym`) pass an in-place
matrix update hook to distributed FEAST. Workers reuse their `T(z)` buffers
instead of allocating a fresh matrix from NEP-PACK at every contour node.
When `FEAST_EXPERIMENT_VALIDATE_T_UPDATE=true`, the experiment checks the
in-place hook against the materializing `T(z)` path before timing.

FEAST result lines include:

- `iterations`: total Beyn/RII iterations recorded by the solver.
- `rii_steps`: `iterations - 1`; zero means the result is just the initial
  Beyn-style contour solve.
- `beyn_only`: true when no RII refinement was performed.
- `stop`: coarse termination classification. `spurious_or_partial_converged`
  means FEAST stopped because the non-spurious residuals converged while some
  inside Ritz values remained above the `spurious` threshold.
- `trace`: compact per-iteration internal residual diagnostics.

Distributed FEAST timing fields ending in `_sum_s` are summed over workers.
They are useful for understanding where worker time is spent, but they are not
wall-clock timings and should not be added to master wall-clock phases.
