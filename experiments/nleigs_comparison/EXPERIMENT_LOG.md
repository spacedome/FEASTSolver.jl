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
- When sparse NLFEAST is implemented, add one large sparse problem, likely
  derived from `gun`, as the third curated comparison case.
