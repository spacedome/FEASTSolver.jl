# Variant Ledger

The labels below identify algorithms, not configuration presets. A status of
`passed` means only that the listed controls passed.

| ID | Inner nonlinear step | FEAST action | Space memory | Status |
|---|---|---|---|---|
| `EV-DIRECT` | one closure update | occupied filter | `p` | passed with mixing and Anderson |
| `EV-RESPONSE` | closure Newton/Krylov | occupied filter and reused factors | `p` | passed on Hermitian local density |
| `EV-FULLQ` | fixed-tolerance reduced solve | filter `q` states | fixed `q` | weak control passed; strong coupling is fragile |
| `EV-TWO-FIXEDQ` | balance closure and leakage defects | occupied filter and complement enrichment | fixed `q` | passed for `g=1,q=5` and `g=5,q≥5` |
| `EV-TWO-ADAPTQ` | stop at leakage floor | occupied enrichment; grow on slow leakage contraction | thick-restarted adaptive `q` | passed from `q=p` on `g=5` |
| `EV-WINDOW-PRE` | fixed-count or leakage-forced reduced solve | append contour moment block before extraction | recent subspace window | passed for windows 2–4 on `g=5` |
| `EV-WINDOW-POST` | fixed-count or leakage-forced reduced solve | realize occupied space, then append it | recent subspace window | passed; agrees with pre-extraction controls |
| `EV-GROWING-PRE` | fixed-count or leakage-forced reduced solve | append contour moment block before extraction | monotonically growing span | passed on `g=5` |
| `EV-GROWING-POST` | fixed-count or leakage-forced reduced solve | realize occupied space, then append it | monotonically growing span | passed; agrees with pre-extraction controls |
| `EV-GROWING-RESP` | inexact reduced Newton–CG | append contour moments before extraction | monotonically growing span | passed; competitive when operator builds are expensive |
| `EV-CACHE-RESPONSE` | periodic or closure/leakage-forced full response, then reduced iteration | one frozen factor cache supplies moments and response | finite or growing pre-extraction window | passed; cost-dependent factor/RHS trade |
| `EV-LEVEL` | one closure update | filter `H(P)−σP` | `p` plus projector | passed, but slower on strong control |
| `EV-ORBITAL` | gauge-aligned orbital update | occupied filter | `p` orbitals | passed, but slower for density-only closure |

`EV-WINDOW-PRE` and `EV-GROWING-PRE` are the faithful descendants of the 2013
eigenvector-nonlinearity NLFEAST idea: append contour-generated subspaces and
solve the nonlinear problem in their span. They must not be represented by
`EV-FULLQ`, which filters a fixed `q`-state contour and discards its reduced
history.

The following are representation extensions rather than schedules of the same
Hermitian density problem:

| ID | Physical state | Status |
|---|---|---|
| `EV-DUAL` | oblique projector from right/left spaces | passed on similarity controls |
| `EV-DUAL-WINDOW` | oblique projector with coupled right/left memory | passed through similarity condition `1.7×10³` |
| `EV-DUAL-CACHE` | oblique projector with shared moment/dual-response factors | passed; useful under constrained memory/inner work |
| `EV-COMBINED` | density plus a `z`-nonlinear invariant pair | passed on quadratic control |
| `EV-COMBINED-WINDOW` | density plus accumulated corrected invariant-pair moments | passed with exact zero-quadratic reduction |
| `EV-COMBINED-CACHE` | density plus repeated residual-forced correction in one frozen cache | passed; accuracy safeguard, not a refresh-count improvement |
| `EV-FERMI` | finite-temperature density matrix and chemical potential | analysis only |
| `EV-PRODUCT` | tuple of orbital projectors | analysis only |
| `EV-NONLOCAL-ADAPTQ` | full orthogonal projector `P` | passed from `q=p` on nonlocal `H(P)` |
| `EV-NONLOCAL-WINDOW` | full orthogonal projector `P` | passed with finite and growing memory |

Rejected or unresolved observations are retained explicitly:

- `categorical_sweep.jl` gives the common `g=5,p=3,N=64` comparison. Factor
  sets/RHS are: response `8/5888`, growing pre-extraction `8/512`, fixed-`q`
  enrichment `12/768`, three-block windows `12/768`, direct Anderson `14/896`,
  adaptive `q=3→6` `15/960`, and full-`q` projection `30/3840`.

- `EV-TWO-FIXEDQ` with `q=p` is unaccelerated projector SCF and diverges on the
  `g=5` control. This is the correct direct-map reduction, not a solver claim.
- `EV-TWO-FIXEDQ` with `g=5,q=3` and `q=4` fails, while `q=5` converges in 18
  refreshes and `q=6,7,8` in 13, 12, and 12 when the forcing ratio is one.
  The earlier `q=5` failure at forcing ratio `0.25` was reduced-solve
  over-solving, not proof of an insufficient response dimension.
- `EV-TWO-ADAPTQ` starting from `q=3` detects slow contraction and converges in
  15 refreshes after one growth to `q=6`, versus 13 when `q=6` is known.
- On the `g=10,p=1` control, fixed `q=3` needs earlier refresh (`η=4`) while
  fixed `q=5` works at `η=1`. Adaptive growth from `q=1` reaches `q=5` and
  converges in 16 refreshes without a supplied response width.
- Carrying raw Anderson secants across changing reduced maps is rejected by
  `history_reuse_sweep.jl`. Physical-coordinate retention raises refresh counts
  from 7 to 8, 12 to 18, and 15 to 31 on three controls and greatly increases
  inner work. Physical coordinates remove gauge dependence, but do not make
  secants from different maps consistent.
- `EV-FULLQ` entangles the virtual-space dimension with the contour count and
  moment capacity. Occupied enrichment avoids that cost, but its fixed-`q`
  truncation may discard useful response directions.
- A small invariant residual alone is not convergence. A two-block rank-one
  run reaches machine-level invariant residual while its closure defect remains
  about 1.37; every solver result is gated on both defects.
- The nonlocal `H(P)=H₀+gKPK` control requires off-diagonal projector data.
  At `g=4.2`, four-block and growing pre-extraction spaces converge in 6 factor
  sets, fixed `q=10` in 8, and adaptive `q=3→6` without a supplied width. This
  reproduces the local-density ordering on a different physical state.
- Coupled dual moment windows preserve one LU per node and the common oblique
  projector gauge. Three-block `d=3,ℓ=1` needs 7 refreshes/1344 RHS versus
  direct mixed dual FEAST at 33/19008; growing memory needs 6/1152.
- For simultaneous `ρ,z` nonlinearity, four-block and growing corrected-moment
  spaces reduce the quadratic control from 11 factor sets/3968 RHS to 6/2048.
  At zero quadratic strength they agree with the linear-pencil window to
  roundoff and use the same refresh count.
- On `g=5`, `EV-WINDOW-POST` with two blocks needs 15 refreshes at two inner steps
  but 28 at three or more. Three and four blocks favor three inner steps and
  converge in 12 and 11 refreshes. Inner accuracy and subspace memory cannot be
  selected independently.
- `EV-WINDOW-PRE` appends only the first `d` moment blocks and avoids the `2d`
  Hankel realization before the reduced nonlinear solve. Depth-one wide agrees
  with `EV-WINDOW-POST` to roundoff and `d=3,ℓ=1` within about `10⁻¹⁰`; the rectangular
  `d=2,ℓ=2` form differs slightly but converges to the same solution.
- Finite-quadrature oversampling can reduce factor sets by retaining filtered
  virtual components. At `N=32,h=3`, depth-one `ℓ=5` takes 9 factors/1440 RHS
  versus higher-moment `d=3,ℓ=1` at 12/384. This is a backend cost tradeoff,
  not a different exact projector map.
- The growing forms reverse that tradeoff: fixed inner depths 2, 3, and 5 need 14,
  10, and 8 refreshes. Depths beyond 5 stay at 8 refreshes while only adding
  reduced work. The final 8-block space has dimension 24 for `p=3`.
- On sparse `n=256,p=4,g=5`, `EV-GROWING-PRE` takes 384 factors and 768 RHS
  versus full response Newton's 336 and 5088. The count crossover is 90 RHS
  columns per extra factorization.
- On that sparse control, five-step reduced Anderson and one-step inexact
  reduced Newton both need 8 refreshes. The former uses 37 inner operator builds;
  the latter uses 8 builds and 24 reduced derivative actions. Warmed times are
  2.660 s and 2.592 s, so neither dominates for the cheap contact Hamiltonian.
- The sibling `cache_native_corrected_moments` experiment makes factor-cache
  reuse explicit. On the dense `g=5` control, a four-block response every third
  refresh reduces 12 factor sets to 9 at a crossover of 8 RHS solves per saved
  node factorization. On sparse `n=256`, it improves an under-solved four-block
  schedule but does not beat growing five-step moments.
- In the dual constrained-memory control, two blocks and one reduced step fall
  from 16 refreshes to 7 with periodic cached response. With three blocks and
  two reduced steps, response adds RHS without reducing the 6 refreshes.
- Repeated corrected polynomial repair reduces frozen-state spectral residuals
  without reducing the six simultaneous `ρ,z` closure refreshes. Residual-forced
  repair is retained; unconditional repeated repair is rejected as oversolving.
