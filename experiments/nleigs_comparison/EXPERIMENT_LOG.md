# NLEIGS Comparison Experiment Log

This log records tuning observations that should not be rediscovered from
scratch. The goal is not to preserve every timing sample, but to document which
regions and knobs were tried and why the current default exists.

## 2026-04-25: `pep0` Large Dense Region

Problem:

- `pep0` from NEP-PACK/NLEVP, dense polynomial problem.
- Main large setting: `n=3000`.
- Fair-memory policy: FEAST `store=false`; NLEIGS `reusefact=0`.
- FEAST worker policy: contour workers use BLAS threads `1`; serial/NLEIGS use
  `Sys.CPU_THREADS`.

Current default:

- FEAST contour: center `0 + 0im`, radius `0.095`.
- FEAST subspace: `m=60`.
- FEAST nodes: `32`.
- FEAST iterations: `4`.
- FEAST processes: compare `0,4,8`.
- NLEIGS target: same center/radius by default.
- NLEIGS polygon points: `96`.
- NLEIGS polygon phase: unaligned vertices.
- NLEIGS defaults for this problem: `maxit=150`, `blksize=32`.

Representative local results:

- `n=3000`, radius `0.095`, FEAST 8 workers, no-store:
  27 interior eigenpairs, all converged, about `44s`.
- `n=3000`, radius `0.095`, NLEIGS no-reuse:
  27 interior eigenpairs, all converged, about `131s`.

These are single-sample local runs, not publication timing statistics.

## Region Search Notes

`radius=0.2`:

- At `n=1200`, the region contains about 50 interior eigenvalues.
- `m=44` and `m=60` were not robust enough with 16 nodes.
- 32 nodes improved FEAST, but the region still had enough boundary/spurious
  behavior that it was not a good comparison default.

`radius=0.1`:

- At `n=1200`, `m=24`, nodes `32` gave a clean 11-eigenpair FEAST region with
  two RII refinement steps.
- At `n=3000`, FEAST with `m=60`, nodes `32`, `store=false`, and 8 workers
  converged a wider 32-eigenpair region.
- NLEIGS was highly sensitive around this boundary. With the original 96-point
  unaligned polygon it recovered only a small subset. With 32 polygon points
  aligned to FEAST node angles it improved to 18 eigenpairs, but still missed
  many of FEAST's 32 eigenpairs.

`radius=0.095`:

- At `n=3000`, FEAST and NLEIGS both recover 27 interior eigenpairs.
- This is currently the best apples-to-apples dense comparison region found.
- The 27-eigenpair NLEIGS result uses the unaligned 96-point polygon. An aligned
  32-point polygon is geometrically closer to FEAST's trapezoid-node angles, but
  under `reusefact=0` it recovered only 7 eigenpairs in the full benchmark.

## FEAST Configurations Tried

- `n=1200`, radius `0.2`, `m=44`, nodes `16`, serial/2/4 workers.
- `n=1200`, radius `0.2`, `m=60`, nodes `16`, serial/2/4 workers.
- `n=1200`, radius `0.2`, `m=60`, nodes `32`, serial/2/4 workers.
- `n=1200`, radius `0.1`, `m=24`, nodes `16` and `32`, serial/4 workers.
- `n=3000`, radius `0.1`, `m=60`, nodes `32`, 8 workers.
- `n=3000`, radius `0.095`, `m=60`, nodes `32`, 8 workers.

Findings:

- 32 nodes are materially more reliable than 16 nodes near this cluster.
- `m` should stay near 2x the expected interior count; undersizing the subspace
  produces spurious/partial convergence.
- At large enough `n`, 8 contour workers are worthwhile under the no-store
  policy, despite worker setup cost.

## NLEIGS Configurations Tried

Target geometry:

- Radii: `0.08`, `0.09`, `0.095`, `0.1`, `0.12`.
- Polygon points: `32`, `96`.
- Polygon phase: unaligned vertices and `feast_nodes` aligned phase.
- Separate NLEIGS target overrides were added so FEAST and NLEIGS regions can be
  tuned independently when needed.

Algorithm knobs:

- `blksize=32`, `96`.
- `maxit=150`.
- `maxdgr=100`, `300`.
- `leja=1`, `2`.
- `reusefact=0`, `1`, `2`.

Findings:

- `reusefact=1` or `2` is useful for diagnostics, but the comparison result
  should use `reusefact=0` whenever FEAST uses `store=false`.
- Increasing block size and max degree did not fix the radius-`0.1` miss.
- Aligning the 32-point polygon with FEAST's contour-node angles improved
  NLEIGS at radius `0.1`, but did not make it recover the full FEAST region.
- At radius `0.095`, aligned 32-point NLEIGS with `reusefact=0` was worse than
  the unaligned 96-point polygon, recovering only 7 eigenpairs.
- Radius sensitivity itself is an important comparison result: the contour
  filter in FEAST is robust on a wider local region where NLEIGS is much more
  sensitive to target choice.

## Open Follow-Ups

- Implement explicit/custom contour support in NLFEAST so FEAST can be run on
  exactly the same polygonal target sets used by NLEIGS.
- For publication runs, repeat the default large comparison with multiple timed
  samples on a larger machine.
- Sparse NLFEAST now has a direct sparse shifted-solve path.
  `gun` is part of the default comparison set; `schrodinger_movebc` remains an
  explicit scalable sparse selector.

## 2026-04-26: `gun` Sparse NLEVP Region

Problem:

- Fixed-size NLEVP gun cavity problem, `n=9956`.
- FEAST side uses the native in-place `GunGalleryOperator`.
- NLEIGS side uses the low-rank-factorized representation from NEP-PACK's own
  gun tests. The raw `nep_gallery("nlevp_native_gun")` representation can fail
  in NLEIGS because its divided-difference setup evaluates matrix square roots
  through a fragile Schur path.
- Fair-memory policy: FEAST `store=false`; NLEIGS `reusefact=0`.

Current default:

- FEAST contour: center `140000 + 0im`, radius `30000`.
- FEAST subspace: `m=36`.
- FEAST nodes: `8`.
- FEAST iterations: `4`.
- NLEIGS target: same center/radius, 32 polygon points.
- NLEIGS pole candidates: `-10 .^ range(-8, 8, length=1000) .+ 108.8774^2`,
  matching the style used by NEP-PACK's gun tests.

Representative local results:

- Old FEAST default `m=32`, nodes `8`, iter `3`, serial:
  17 converged interior eigenpairs plus 1 spurious/edge interior Ritz value,
  three RII steps, about `23.1s`.
- Old FEAST default `m=32`, nodes `8`, iter `3`, 8 workers:
  same eigenpair classification, about `6.7s`.
- Tuned FEAST default `m=36`, nodes `8`, iter `4`, serial:
  same eigenpair classification, two RII steps, about `25.9s`.
- Tuned FEAST default `m=36`, nodes `8`, iter `4`, 4 workers:
  same eigenpair classification, about `7.0s`.
- Tuned FEAST default `m=36`, nodes `8`, iter `4`, 8 workers:
  same eigenpair classification, about `5.6s`.
- NLEIGS no-reuse, low-rank-factorized gun representation:
  17 converged interior eigenpairs, about `30.4s`, 54 factorizations.

These are single-sample local runs after warmup, not publication timing
statistics.

Tuning notes:

- The contour region was inherited from the legacy paper experiments and is a
  good sparse stress case.
- FEAST consistently finds 18 interior Ritz values; 17 satisfy the strict
  residual threshold and one remains around `6e-4` to `8e-4`, so we classify it
  as spurious/edge behavior rather than forcing the benchmark to chase it.
- Increasing to `m=36` or `m=40` reduces FEAST to two RII steps but does not
  remove the spurious/edge value. `m=36` is the best local distributed default.
- Increasing to 16 contour nodes gives very small residuals on the 17 good
  eigenpairs but still leaves the same spurious/edge value and doubles the
  number of sparse factorizations, so it is not the default.
- The native gun residual action avoids NEP-PACK residual allocation and roughly
  halves the residual phase, but the run remains correctly dominated by sparse
  LU factorization.

## 2026-04-25: `schrodinger_movebc` Sparse Region

Problem:

- Native FEAST gallery implementation of NEP-PACK's moving-boundary
  Schrodinger problem.
- Sparse problem with in-place `T!(M, z)` materialization and sparse direct
  shifted solves.
- Main local setting: `n=50000`.
- Fair-memory policy: FEAST `store=false`; NLEIGS `reusefact=0`.

Current default:

- FEAST contour: center `-35 + 0im`, radius `4.2`.
- FEAST subspace: `m=8`.
- FEAST nodes: `24`.
- FEAST iterations: `6`.
- FEAST tolerance: absolute action residual `1e-5`.
- NLEIGS target: same center/radius, 24 polygon points, singularity `-10`.
- NLEIGS tolerance: `1e-5`.

Representative local results:

- `n=50000`, FEAST serial, no-store:
  3 converged interior eigenpairs plus 4 spurious interior Ritz values, two RII
  steps, about `3.0s`.
- `n=50000`, FEAST 4 workers, no-store:
  same eigenpair classification, about `2.5s`.
- `n=50000`, FEAST 8 workers, no-store:
  same eigenpair classification, about `2.8s`; overhead beats the extra
  parallelism for this 24-node contour on the local 16-thread machine.
- `n=50000`, NLEIGS no-reuse:
  3 converged interior eigenpairs, about `11.9s`.

These are single-sample local runs after warmup, not publication timing
statistics.

Region and tolerance notes:

- The target cluster is the three real eigenpairs near `-39.15`, `-34.94`, and
  `-31.06`.
- Matrix-norm relative residuals are misleading for this problem because
  `norm(T(λ))` can be enormous near the target region. The benchmark therefore
  uses absolute action residuals on normalized vectors.
- At `n=10000`, radius `4.2` cleanly exposed the same three real eigenpairs and
  one nearby spurious Ritz value under absolute residual checks.
- At `n=50000`, `m=8`, nodes `24`, and `tol=1e-5` gives the current best local
  comparison: FEAST performs RII refinement and NLEIGS recovers the same three
  eigenpairs.
- At `n=100000`, radius `4.2` was not robust locally: `m=8,nodes=24,iter=6` and
  `m=12,nodes=32,iter=8` both hit the FEAST iteration limit with unstable
  spurious Ritz values. This should be revisited on a larger machine and with a
  broader contour/subspace search before using `n=100000` in reported results.

Implementation notes:

- Distributed sparse NLFEAST workers now use the same sparse symbolic-reuse path
  as serial NLFEAST when an in-place matrix materializer is provided.
- Distributed nonlinear accumulation now calls the shared fused moment
  accumulator used by serial NLFEAST. Before this change, worker accumulation
  dominated the sparse distributed run and hid the contour-level parallelism.
