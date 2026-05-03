# Moment-RII / SS-FEAST Research Note

This experiment track is for the open question in `docs/article.tex`: how to
incorporate higher Hankel moments into the NLFEAST residual inverse iteration
without expanding the active subspace by a factor of `K` at every iteration.

## Current Situation

The canonical `nlfeast!` state is an eigenpair list `(X, Lambda)`, with
`Lambda` represented diagonally. The RII correction accumulated at contour node
`z` is

```text
(X - T(z) \ T(X, Lambda)) * inv(zI - Lambda)
```

The higher-moment prototypes in `src/nlfeast.jl` and
`src/nlfeast_experimental.jl` replace the two Beyn moments with block Hankel
matrices built from `Q_0, ..., Q_{2K-1}`. This works as a Beyn/SS extraction
step, but it creates a dimension-management problem: `m` active vectors produce
roughly `K*m` Ritz candidates. If all candidates are used in the next RII step,
the next Hankel extraction grows again; if we truncate back to `m`, we lose
exactly the information that made higher moments useful.

## Working Hypothesis

The right state for moment-RII is not a diagonal eigenpair list. It is an
invariant pair `(X, S)`, where `S` is a small dense matrix. For polynomial NEPs,
the residual is

```text
T(X, S) = sum_j A_j * X * S^j
```

For a general analytic NEP, the analogous object is the matrix-function action
`T(X, S)`, the same quantity NEP-PACK calls `compute_MM(nep, S, X)`.

With this state, the RII moment update becomes

```text
Q_k = contour_integral z^k * (X - T(z) \ T(X, S)) * inv(zI - S) dz
```

This closes the higher-moment iteration. If `(X, S)` is exact, the residual term
vanishes and Cauchy's formula gives

```text
Q_k = X * S^k
```

so the higher moments live on the finite-dimensional invariant-pair manifold
instead of creating independent new right-hand sides forever. The diagonal
current implementation is the special case `S = Diagonal(Lambda)`.

The current interpretation is sharper than the original name suggests: for
higher moments this is probably no longer RII in the strict scalar-eigenpair
sense. RII corrects a single root function/eigenvector with a scalar eigenvalue
in the resolvent. The higher-moment state is an invariant pair on a quotient
manifold: `T(X,S)=0` plus a gauge/minimality condition on the lifted block
`[X; X*S; ...]`. The natural local refinement is invariant-pair Newton/block
Newton. The contour step supplies a rationally filtered SS/Beyn-style
initializer and may still be useful as a globalization/filtering iteration, but
the local geometry is Newton on `(X,S)`, not scalar RII.

## Why This Matters

- Defective or tightly clustered nonlinear spectra are naturally represented by
  a small Schur/Jordan-like `S`, not by independent scalar eigenpairs.
- SS-Hankel and higher-moment Beyn already construct a projected pencil whose
  eigenspace approximates this invariant pair.
- NLFEAST's RII update can plausibly refine that invariant pair directly,
  rather than refining only a selected subset of scalar Ritz vectors.
- This gives a clean theoretical bridge: Beyn is the first extraction,
  SS-Hankel is the higher-moment extraction, and FEAST/NLFEAST is the RII
  iteration applied to the same invariant-pair object.

## What The Moment Blocks Represent

For simple eigenvalues, Keldysh's theorem gives the local resolvent expansion

```text
T(z)^-1 V ≈ X * inv(zI - S) * C
```

inside the contour, where `X` is the physical output map, `S` is multiplication
by the spectral parameter on the residue space, and `C` is how the right probe
excites that space. The contour moments are therefore Markov parameters:

```text
Q_k = contour_integral mu(z)^k * T(z)^-1 V dz = X * S_mu^k * C
```

up to the chosen contour coordinate `mu`. The block Hankel matrices are not
merely larger subspace projections; they factor as

```text
H_0 = observability(X, S) * controllability(S, C)
H_1 = observability(X, S) * S * controllability(S, C)
```

so the small pencil identifies the multiplication operator `S` up to
similarity. This is the same system-realization geometry used by SS methods.
The full one-sided Beyn form corresponds to using the identity as the left
probe; the SS form uses a separate left probe `W` and works with `W' * Q_k`,
giving much smaller Hankel matrices.

For the iterative correction, the goal is not to append more independent
vectors forever. The corrected contour blocks should again be interpreted as
new Markov parameters of the same finite-dimensional realization. RII is a
root-correction viewpoint for scalar/diagonal states; higher moments ask for a
correction of the realization `(X, S, C)` or its observable/controllable
quotient.

## Proposed First Prototype

Start with polynomial gallery operators only. They give an exact, cheap
implementation of `T(X, S)` and avoid making the first pass depend on generic
analytic matrix-function machinery.

1. Add an internal invariant-pair residual helper for `PolynomialGalleryOperator`.
   It should compute `sum_j A_j * X * S^j` into a caller-owned buffer.
2. Refactor the existing higher-moment accumulation into a workspace object that
   accepts `(X, S, R)` and accumulates `Q_0, ..., Q_{2K-1}` with right
   multiplication by `inv(zI - S)`.
3. Build the SS-Hankel extraction as a separate function returning a compressed
   invariant pair `(Xnew, Snew)` plus scalar eigenvalue/residual diagnostics.
4. Validate on the small deficient quadratic problem from the legacy
   experiments before trying butterfly or gun.
5. Only after the serial dense prototype is numerically convincing, decide how
   this maps onto sparse and distributed NLFEAST.

## Robust Path From Here

The current goal is not to claim a universal higher-moment NLFEAST formula.
The more defensible target is a small class of charted realization algorithms
with diagnostics good enough to choose the right chart in ordinary cases.

1. Lock down the linear layer: projected SS-FEAST must reduce to ordinary FEAST
   on `T(z)=zI-A`, while allowing fewer physical probe columns than wanted
   eigenvalues.
2. Lock down the polynomial layer: compare polynomial moment-RII against FEAST
   on the companion pencil, and treat infinite roots as projective chart
   changes through the reversed polynomial `nu^d P(1/nu)`.
3. Build a `Chart` abstraction for the nonlinear layer. A chart is a contour,
   spectral coordinate, moment basis or offset, extraction method, and
   realization gauge.
4. Attach diagnostics to every chart: Hankel rank gap, retained singular-value
   ratio, `cond(S)`, eigenvector conditioning of `S`, pair residual, scalar
   residuals, and whether pair/scalar residuals disagree.
5. Make adaptive selection explicit. First try gauge balancing in the current
   chart; then try moment offsets or rational coordinates; then split into
   nested or local contours and merge converged roots.
6. Replace the current supervised scalar experiments with real count/rank
   estimation. The scalar demos still use known roots only to size the local
   finite realization; production code needs a contour count estimator.

## Findings So Far

- Lifted normalization is essential. Normalizing only the physical block `X`
  collapses the deficient quadratic experiment from four interior eigenvalues
  to two. Normalizing the lifted block `[X; X*S; ...; X*S^(K-1)]` recovers all
  four.
- The moment coordinate should be scaled to the contour. Raw `z^k` moments are
  badly conditioned on large contours and bury interior eigenvalues near the
  center. On a circular contour, using `mu=(z-c)/r` for Hankel moments while
  mapping back with `S_lambda=cI+r*S_mu` is much more stable.
- A scalar `T(z)=sin(z)` problem with `n=1` and seven eigenvalues inside a
  radius-10 contour is a useful stress test. With scaled moments, `K=7` and 64
  contour nodes converged all seven eigenvalues after the contour/RII-style
  iteration, demonstrating more eigenvalues than physical dimension without
  expanding physical `X`.
- Pure block Newton, as implemented in NEP-PACK, can refine a good contour
  invariant-pair initializer to machine precision on the scalar sine problem.
  It can also fail badly from poor or ill-conditioned initial pairs, so the
  contour step remains important.
- Large scalar contours remain hard. At radius 20 for `sin(z)`, `K=13` and
  128--256 nodes produce small invariant-pair residuals but not uniformly small
  scalar eigenvalue residuals, indicating lifted minimality and conditioning are
  still the main unresolved issues.
- Re-extracting a full Hankel realization after every correction is not the
  only possible geometry. Once the state dimension has been discovered, the
  corrected moments should satisfy a shift relation
  `[Q_0; ...; Q_{K-1}] * S = [Q_1; ...; Q_K]`. A prototype shifted-realization
  update converges the scalar radius-10 sine problem with `K=7` and 32 nodes,
  where repeated Hankel extraction only converges part of the spectrum.
- The shifted realization is not yet robust. On the radius-20 scalar sine
  problem the lifted stack is severely ill-conditioned and the update can
  collapse the rank. This supports the interpretation that the missing piece is
  a well-chosen gauge/basis for the finite-dimensional multiplication operator,
  not merely "use Newton" or "use more moments".
- A Chebyshev-basis recurrence was tested as a first non-monomial coordinate
  system: if `Q_k = X*T_k(S_mu)`, then multiplication by `S_mu` is represented
  by `Q_0*S_mu = Q_1` and `Q_k*S_mu = (Q_{k-1}+Q_{k+1})/2`. This matches the
  shifted update on the radius-10 sine case, but does not cure the radius-20
  plateau by itself. The failure is not just the polynomial basis; it is also
  the conditioning of the minimal realization and physical observability through
  `X`.
- A two-sided projected Hankel extraction now mirrors the SS geometry. It forms
  small Hankel matrices from `W' * Q_k`, but reconstructs the physical output
  map with `[Q_0 ... Q_{K-1}] * V * Sigma^-1`. On the deficient quadratic case
  it reduces a `30 x 6` full Hankel to `6 x 6` and still recovers all four
  interior eigenvalues. On butterfly with `K=2`, it reduces `128 x 32` to
  `32 x 32` and recovers the thirteen wanted scalar eigenpairs.
- Spurious values are not merely output cruft, but pruning is subtle. On
  butterfly with `K=2`, retaining three extra states gives all thirteen wanted
  scalar eigenpairs with max residual about `2e-11` and wanted-pair residual
  around `1e-9`. An exact target-count projected run is also stable, but an
  exact target-count shifted-realization run can fail badly, matching only
  three wanted eigenvalues. A real policy needs rank/drop information, residual
  history, and possibly replacement vectors; contour membership alone is not
  enough.
- The linear SS-FEAST control problem behaves as expected. On a diagonal matrix
  with ten eigenvalues inside the contour and only four probe columns, standard
  FEAST with four columns cannot represent the target space, while ordinary
  FEAST with twelve columns converges to machine precision. Projected SS-FEAST
  with four right-hand sides and `K=3` also recovers all ten eigenvalues,
  showing that the moment realization gives the same effective subspace width
  without tripling the number of linear-solve right-hand sides. The slow
  regression `just test 'linear SS-FEAST'` now pins this reduction: same-probe
  FEAST returns four values, wide FEAST returns ten, and projected SS-FEAST
  converges all ten with final residual around `3e-13`.
- On a nonnormal Grcar control region with eight eigenvalues, projected
  SS-FEAST with `K=2` refines from `4e-9` initial scalar residuals to machine
  precision. This gives a clean linear baseline for the nonlinear moment
  update: if a proposed higher-moment NLFEAST step does not reduce to this
  behavior for `T(z)=zI-A`, it is probably the wrong geometry.
- The first nonlinear update comparisons support the invariant-pair view. On
  the deficient quadratic case, repeated projected extraction, shifted
  realization, and projected extraction plus one block-Newton refinement all
  recover the four interior eigenvalues to machine precision. On butterfly, the
  Newton-refined projected path reduces the full invariant-pair residual to
  about `1e-14`, while the pure contour updates give excellent scalar
  eigenpairs but larger pair residuals.
- The low-dimensional many-eigenvalue polynomial case is the strongest
  positive test so far. It has `n=4`, four probe vectors, degree eight, and
  twenty interior eigenvalues after a nonnormal similarity transform. With
  `K=5`, the projected Hankel has rank twenty and the nonlinear update recovers
  all twenty eigenvalues. With only eight contour nodes, repeated projected
  extraction fails, but the shifted-realization update converges all twenty to
  roundoff. With `K=4`, the rank capacity is only sixteen and both updates
  fail, as expected.
- A harness bug was found and fixed while running the low-dimensional case:
  moment-realization rank must not be capped by physical dimension `n`.
  For nonlinear problems the state dimension can exceed `n`; the correct
  experimental cap is the Hankel capacity, such as `K * probe_cols` and
  `K * left_probe_cols`.
- Ad-hoc residual/persistence selection was tested as a possible pruning aid,
  but it did not change the targeted failures. This supports the view that
  scalar-level deflation is the wrong primary lever for this method.
- Gauge control is now the main positive result. The scalar `sin(z)` radius-20
  case previously had a tiny invariant-pair residual but only one of thirteen
  scalar roots converged because the small multiplication matrix was violently
  nonnormal (`cond(eigenvectors(S))` around `1e12`). Applying diagonal
  similarity balancing to the pair during the shifted-realization iteration
  converges all thirteen roots with max scalar residual around `1e-10`. This is
  a natural realization-gauge fix, not a scalar Ritz deflation rule.
- The gauge experiment distinguishes useful and non-useful gauges. A Schur
  gauge is unitary and makes the small operator triangular, but it does not
  prevent rank collapse on the scalar radius-20 shifted update. Non-unitary
  diagonal balancing is the important operation. Chebyshev shifted moments plus
  diagonal balancing also converge all thirteen scalar roots, so the basis can
  help, but only after the realization gauge is controlled.
- Gauge balancing preserves the successful low-node `n=4` polynomial
  many-eigenvalue result and slightly improves its full pair residual. It does
  not fix the exact target-count shifted butterfly failure. In that case the
  small operator is already well conditioned, so the failure is not a gauge
  conditioning problem; it is likely a shifted-update/restriction geometry
  problem. Projected extraction and projected Newton remain the stable choices
  for that case.
- Scalar analytic functions with infinitely many roots expose a second
  limitation: a single monomial moment realization on a large contour can bury
  roots near the center while accurately recovering roots closer to the contour.
  `cos(z)` on radius 20 is the current counterexample. Diagonal gauge
  balancing is necessary but not sufficient; using offset moments such as
  `Q_2, Q_3, ...` recovers the outer ten roots, while smaller nested contours
  recover the central pair. A union of nested gauge-balanced contour solves
  recovers all twelve roots.
- This makes the companion-polynomial analogy useful again. A polynomial
  companion linearization gives a finite-dimensional coordinate system where
  all roots of the polynomial live at comparable footing in an enlarged state.
  For analytic functions with infinitely many roots, any finite moment
  realization is a local rational approximation. The right general method is
  therefore probably not "one huge contour, one monomial basis"; it is
  gauge-controlled local realizations, possibly with adaptive nested contours,
  moment offsets, or rational bases.
- Polynomial companion control now agrees with the moment formulation on the
  `n=4`, degree-eight many-eigenvalue polynomial. FEAST on the `32 x 32`
  companion pencil recovers all twenty finite target roots, and both projected
  Newton and gauge-balanced shifted moment updates recover the same twenty
  roots in the polynomial-native invariant-pair representation.
- A degree-deficient polynomial control confirms that infinite roots are a
  chart issue at the polynomial layer. The original companion pencil has one
  infinite/singular eigenvalue because the leading coefficient is singular.
  Reversing the polynomial and solving near `nu=0` recovers that root as a
  finite projective-chart eigenvalue.
- The first adaptive scalar chart prototype is deliberately supervised but
  encouraging. On `cos(z)` with outer radius 20, it grows nested contours,
  chooses between moment offsets by new converged roots, and recovers all
  twelve known roots. The remaining algorithmic gap is replacing exact local
  root counts with rank/count estimates and principled split rules.
- A first rank-adaptive scalar chart prototype now removes the exact root
  counts from chart sizing. It estimates local realization size from Hankel
  singular values, uses the outer-contour rank estimate as the target count,
  grows nested contours, and recovers all twelve `cos(z)` roots on radius 20.
  The rank threshold matters: `1e-5` undercounts this example as ten roots,
  while `1e-6` recovers the expected twelve. This makes rank-estimation
  diagnostics part of the algorithm, not an implementation detail.
- Scalar rank-estimation stress tests show both sides of the count problem.
  Near-contour exterior roots can inflate numerical Hankel rank, as in
  `sin(z)` on radius 6, while large contours can bury weak interior states and
  undercount, as in `sin(z)` and `sin(z)-0.3` on radius 20 or 30. The adaptive
  chart stress still recovers all tested scalar roots for radii 10 and 20, but
  `sin`-type radius-20 cases can recover all roots while the outer rank target
  remains too small. The policy therefore now treats the outer contour as a
  final consistency chart instead of stopping as soon as an estimated count is
  reached.
- The first non-scalar analytic toy problem is diagonal with `sin(z)` and
  `cos(z)` on separate physical components. A residual-normalization fix
  exposed that pair residual alone had been too optimistic: physical
  eigenvectors can be nearly annihilated by the first block `X`. With
  Chebyshev moments, observable-eigen gauge, and a best-history stopping
  diagnostic, a single rank-estimated radius-10 chart recovers all thirteen
  roots. Radius 20 remains unsolved: rank-adaptive nested charts recover the
  central thirteen roots, but the outer roots are not represented to tight
  scalar residual tolerance. This is a useful failure because it separates
  chart quality and physical observability from simple scalar deflation.
- A one-step generic lifted Newton refinement was tested on a poor radius-20
  diagonal analytic pair and did not improve it. That suggests local Newton is
  not a magic cleanup stage if the realization/chart has already mixed the
  residue directions badly. We need better chart quality, two-sided extraction,
  or a more structured analytic invariant-pair correction before local Newton.
- Dual FEAST is a distinct clue, not just a synonym for projected Hankel. The
  current projected Hankel path only observes right moments with a left probe.
  True dual FEAST filters left and right subspaces, biorthogonalizes them, and
  performs a Petrov-Galerkin extraction. For higher-moment NLFEAST this suggests
  that `S` may be the wrong final scalar extractor in hard nonlinear cases: the
  moment iteration should build left/right physical trial spaces, then solve or
  refine the small reduced NEP `Y' * T(lambda) * X`. A balanced Hankel
  realization was added as a dual-inspired gauge check; it does not fix the
  diagonal radius-20 failure by itself, so the missing piece is likely the real
  left/right residual update or reduced-NEP extraction, not just SVD scaling.
- The first true reduced-NEP extraction prototype is much stronger. Left and
  right moment-filtered bases reduce the diagonal `sin/cos` radius-20 problem to
  a `2 x 2` NEP. Applying the argument principle to
  `det(Y' * T(lambda) * X)` gives the exact root count 25, and a determinant
  Newton cleanup refines all 25 roots to about machine precision. This solves
  the failure that defeated direct `S` extraction, confirming that the reduced
  nonlinear problem can contain many roots even when the physical trial/test
  spaces are tiny.
- On the nonnormal degree-eight polynomial control with `n=4` and twenty target
  roots, reducing the polynomial coefficients with moment-filtered physical
  bases and solving the reduced polynomial recovers all twenty roots. In these
  full-rank controls Galerkin and dual Petrov-Galerkin extraction both work; the
  expected role of the left basis is conditioning and nonnormal robustness, as
  in linear dual FEAST.
- A more ill-conditioned polynomial control now shows the first concrete
  dual/Galerkin separation. The exact target roots are known by construction,
  but the Vandermonde similarity makes root locations sensitive enough that
  residuals are the primary validation metric. With matched left/right filtered
  spaces, dual extraction returns the twelve target roots and no residual-small
  extras. The same right space used as its own Galerkin test space admits an
  additional residual-small spurious root. This supports making left/right
  Petrov-Galerkin extraction and spurious diagnostics first-class rather than
  treating the left space as an optional conditioning tweak.
- The first iterative dual nonlinear FEAST prototype now works on a deliberately
  bad polynomial chart. A rank-truncated initial dual-sensitive solve finds
  residual-small but wrong Ritz values. One two-sided scalar RII step, using
  `T(z) \ R_right` and `T(z)' \ R_left` exactly as dual linear FEAST does,
  recovers all twelve target roots and removes the spurious values. The step
  then compresses the corrected columns back to physical left/right bases with
  an SVD.
- The expensive residual solves in that scalar RII step can be low-rank
  compressed. If the right residual block factors as `R = U*C`, each contour
  node only needs solves with `U`; the per-root columns are reconstructed
  cheaply before applying the scalar resolvent weights. On the bad polynomial
  chart this reduces the residual solve width from nine Ritz columns to three
  right and three left columns while preserving convergence. This is the first
  concrete bridge between scalar Ritz-vector RII and a compact realization
  update.
- The bridge can be made explicitly moment-based on circular contours. With
  `z=c+r*zeta` and `lambda=c+r*alpha`, the scalar FEAST weight satisfies
  `w/(z-lambda) = sum_k alpha^k zeta^(-k)/N`. Therefore the right correction
  lies in the span of the old right basis plus residual-solve Laurent moments
  `sum_z zeta^(-k) T(z)^-1 U`; the left correction uses the conjugate moments
  `sum_z zeta^k T(z)'^-1 U_left`. On the same bad polynomial chart, one
  residual moment already recovers all twelve target roots to machine
  precision after Petrov-Galerkin reduced extraction. This is the cleanest
  connection so far between FEAST RII and the Hankel/SS moment expansion.
- The same residual-moment update also resolves the low-dimensional
  many-eigenvalue polynomial control from a deliberately rank-deficient initial
  basis. The initial reduced solve sees fourteen inside values but no valid
  target matches; one residual Laurent moment adds the missing physical
  direction and recovers all twenty target roots. This is the first experiment
  that directly addresses the motivating case where `n` is small but the
  contour contains many nonlinear eigenvalues.
- The residual-moment update is not polynomial-specific. On a nonnormal
  similarity transform of a diagonal analytic NEP with scalar entries
  `sin(z)`, `cos(z)`, and `sin(z)-0.3`, a radius-10 chart starts from a
  deliberately rank-one left/right basis and sees only six determinant roots.
  One dual residual Laurent moment expands the basis to rank three, the reduced
  determinant extraction counts twenty roots, and all twenty match the true
  analytic roots to machine precision.
- The current reduced determinant extractor is now the weak link for larger
  analytic charts. It reconstructs roots from monomial power sums of
  `det(Y' T(lambda) X)`, which is ill-conditioned when the contour contains
  many roots. For the three-function analytic toy at radius 20 it gets the
  correct count of 38 with full-rank bases but does not refine all roots
  accurately. This should be treated as extraction instability, not evidence
  against the dual residual-moment update.
- A cleaner reduced extractor now uses the argument principle only for the
  count and SS/Hankel only for the roots/vectors of the reduced NEP. This
  counted-SS extractor avoids the unstable monomial power-sum root
  reconstruction and avoids a hidden Hankel rank threshold. On the
  three-function analytic toy at radius 20, a rank-one initial chart sees only
  twelve roots; one dual residual Laurent moment expands the basis to rank
  three, counted-SS reports 38 roots, and all 38 match the true roots to
  roundoff.
- The four-function analytic toy with `exp(z)-1` remains a deliberate chart
  stress case. At radius 10 the counted-SS extractor recovers the expected
  roots up to the duplicate zero eigenvalue/multiplicity issue. At radius 20,
  exponential growth around the circle and the extra shared root make the
  reduced projection much less reliable. This points to local contours or
  rational coordinates for stiff analytic functions, not a different RII
  formula.
- NEP-PACK's block SS implementation uses the generalized reduced Hankel pencil
  `U' H_1 V x = theta U' H_0 V x`, whereas the Beyn-style form uses the
  explicit SVD similarity `U' H_1 V Sigma^-1`. Both forms were tested in the
  reduced counted-SS extractor. They agree on the successful radius-20
  three-function control and fail similarly on the stiff `exp(z)-1` mixture, so
  the remaining issue is not this algebraic extraction choice.
- The stiff four-function toy exposed two chart-policy requirements. First,
  strict vector residuals are needed as a diagnostic because operator-relative
  residuals can hide displaced values when `exp(z)` dominates `norm(T(z))`.
  Second, local analytic scaling matters: multiplying each scalar component by
  a nonzero chart-local constant does not change the roots, but it removes the
  huge positive-real imbalance that made otherwise valid local charts appear to
  fail.
- With strict vector residuals and contour-max component scaling, supervised
  local charts recover all 44 unique roots inside the radius-20 four-function
  control. A residual-scored grid cover with spacing `2.4` and local radius
  `1.8` also recovers all 44 without using exact root centers. This is the
  current robust answer: one large monomial chart is a diagnostic/seed, while
  reliable generalized moment-NLFEAST uses scaled local charts, dual reduced
  extraction, residual Laurent updates, and a merge policy.
- A triangular analytic control now breaks the simultaneously diagonalizable
  assumption. The determinant roots are still known from the diagonal scalar
  factors, but the operator is upper triangular and can be strongly nonnormal.
  The first grid cover missed an entire real-axis chain because the Cartesian
  grid did not include the real axis; this was a chart-cover bug, not an
  update failure. A centered grid plus multiple local radii recovers all 44
  unique roots even with triangular coupling `10`.
- Multiple-root controls are diagnostic, not a near-term API target. The scalar
  `sin(z)^2` stress reports algebraic count 14 on radius 10 and recovers all
  14 multiplicity-counted roots. A shared-root triangular `sin/exp(z)-1`
  control at zero reports algebraic count 2 and recovers both copies despite
  geometric coalescence. This is useful evidence that the reduced counted-SS
  extraction sees generalized residue information, but the practical solver
  should not require explicit Jordan-chain handling unless it improves ordinary
  subspace quality or convergence diagnostics. This mirrors Beyn/SS/linear
  FEAST usage, where explicit Jordan calculations are usually avoidable.
- A zero-update comparison helps explain how Beyn/SS often avoid extra
  machinery. On the scaled local chart cover for the simultaneously similar
  four-function analytic control, counted-SS alone (`iterations=0`) already
  recovers all 44 unique roots. On the upper-triangular nonnormal control with
  coupling `10`, the same chart cover recovers only 34 roots with zero residual
  updates, 42 after one residual Laurent update, and all 44 after two. This
  suggests the lower rung is "good local moment realization plus reduced
  extraction"; the FEAST residual update is the next rung for repairing weak or
  nonnormal physical trial/test spaces, not a mandatory cost for every chart.
- A first Loewner reduced extractor is now available in the experiment. It
  reuses the same reduced contour solves as counted SS, samples the contour
  pole transfer function at interpolation points outside the chart, and forms
  Loewner/shifted-Loewner pencils. On the scalar `sin(z)` radius-20 validation,
  counted Loewner recovers the same thirteen roots as counted SS to roundoff.
  On the harder rank-deficient analytic reduced-basis cases it does not yet
  beat counted SS; interpolation-point placement and rank visibility are now
  explicit diagnostics rather than hidden moment-order choices.
- A Loewner interpolation-point sweep was added for the exponential many-root
  global chart. It varies the outside interpolation radius and phase while
  keeping the contour solve data, reduced basis, count, and residual policy
  fixed. On the current radius-10 exponential control all tested Loewner
  layouts recover the 18 roots; one updated layout (`rho=1.3`, zero phase)
  keeps 18 residual-small values but matches only 17 roots at the strict
  `1e-6` lattice-matching tolerance. The focused
  `run_global_loewner_interior_artifact_diagnostic` records this as an
  in-target artifact rather than an outside-domain local-chart extra: among six
  layouts, one single updated layout has one residual-small spurious value with
  nearest-root distance about `1.2e-6`, residual about `6.9e-11`, and Loewner
  singular ratio about `3e-8`. Cross-layout clustering with support threshold
  at least two removes the artifact and recovers all 18 roots after the residual
  update. The practical policy should therefore score several cheap Loewner
  layouts and prefer support across layouts/charts rather than treating a single
  Loewner pencil as canonical.
- The same Loewner-layout question was pushed into the local chart cover. On
  supervised radius-10 exponential charts with local radii `1.2` and `2.0`, all
  tested Loewner layouts recover the 18 roots after support merging. The raw
  union contains 20--22 residual-small candidates depending on interpolation
  radius/phase, but chart support `>=2` consistently returns exactly the 18
  expected roots. This separates two roles: Loewner layout selection is an
  important diagnostic on oversized global charts, while local chart geometry
  plus support merging is robust across the tested Loewner layouts.
- The first unsupervised Loewner local-chart cover for the exponential
  many-root radius-10 problem now removes exact-root-centered charts. A
  Cartesian grid with spacing `2.4` and local radii `1.2, 2.0` already has a raw
  union that matches all 18 roots, but only 12 roots have support `>=2`. A
  denser spacing `1.8` gives 97 centers and support `>=2` returns exactly the
  18 roots; spacing `1.2` gives 221 centers and support `>=3` still retains all
  18. This clarifies that support thresholds are cover-density diagnostics, not
  absolute pruning laws: coarse grids can recover roots with weak support, while
  denser grids buy stronger support at higher solve cost.
- A first adaptive refinement policy repairs the coarse-grid weak-support case
  without paying for the full denser grid. Starting from spacing `2.4`, it adds
  the residual-small candidate values whose chart support is below two as new
  chart centers. This adds five centers, increasing the cover from 57 to 62
  centers, and changes support `>=2` from 12/18 roots to exactly 18/18. The
  policy is still a supervised diagnostic only in its validation, not in the
  refinement rule: it uses computed residual-small weak-support candidates, not
  exact root locations. This is the first concrete adaptive-chart rung for the
  robust algorithm.
- The same weak-support refinement was then applied to a harder nonnormal
  triangular analytic control rather than the clean diagonal/similarity
  exponential control. On the radius-6, coupling-10 triangular case, a coarse
  spacing-`2.4` grid with 21 centers has a raw union containing all 11 roots,
  but support `>=2` retains only 5/11. Restricting adaptive refinement centers
  to weak-support candidates inside the final target contour avoids spending
  charts on outside-domain local roots: one refinement round adds five centers
  and raises global support `>=2` to 10/11; a second round adds two more centers
  and raises global support `>=2` to exactly 11/11. The unfiltered support-2 set
  still has 13 candidates because local charts can legitimately recover roots
  just outside the global contour. Applying the final global-contour membership
  test reduces that set to exactly 11/11 roots. This upgrades the policy from a
  pure support threshold to a chart-cover retention rule: local charts may
  extend past the target domain, so merged candidates must pass both repeated
  support and final target-contour membership, and refinement should prioritize
  weakly supported candidates in the requested target domain.
- The target-limited refinement rule also repairs a larger radius-20 analytic
  chart cover for the three-function `sin/cos/sin(z)-0.3` control, which has 38
  unique target roots. A coarse spacing-`3.0` grid with local radii `1.5, 2.4`
  initially recovers only 36/38 roots in the raw union and 16/38 with
  support `>=2`. The first weak-support refinement round finds all 38 in the
  raw union but still only 36/38 with final global support. A second round adds
  just two more target-domain weak centers and yields 38/38 final global
  support. Repeating this full adaptive solve for Loewner radii `1.15`, `1.3`,
  and `1.6` gives exact 38/38 final global support for each layout, and
  clustering final candidates across layouts with support `>=2` also returns
  exactly 38/38. This is stronger evidence that the adaptive rule is not just
  fixing the exponential lattice case or the small triangular radius-6 control;
  it can repair a larger many-root analytic chart where the initial cover misses
  roots outright, and the repaired retained set is stable across tested Loewner
  interpolation radii.
- The first automatic retention-policy wrapper now turns these diagnostics into
  an explicit decision ladder. On the radius-20 three-function adaptive cover,
  support `>=2` plus final target-domain membership already retains exactly
  38/38 with small residuals, but the local argument-principle count diagnostics
  still contain chart warnings. The policy therefore does not silently accept
  the single run: its first decision is `:escalate`, requesting Loewner-layout
  and reduced-extractor agreement. After those independent checks both certify
  the same 38/38 retained set, the policy accepts with chart warnings rather
  than deleting roots or hiding the count stress. The global exponential
  in-target artifact uses the complementary policy branch: because one single
  Loewner layout is unsafe, the policy refuses single-layout residual-small
  values and retains only cross-layout support-2 candidates, again recovering
  exactly 18/18.
- Count-stressed charts now have a concrete split/shrink refinement rung. The
  policy acts only on selected best-radius chart records, because diagnostic
  records from non-selected radii do not correspond to accepted chart entries.
  On the radius-20 three-function cover, blind half-radius child charts fail on
  both selected count-stressed parents. Reusing the parent chart's
  residual-small values as candidate centers recovers the local roots, but one
  parent still has weak support because root-centered charts do not overlap
  enough. Adding the parent center and four half-radius child cover anchors
  gives overlapping evidence and support-certifies the local roots on both
  count-stressed parents (`5/5` and `4/4`). This turns split/shrink from a
  loose suggestion into a reproducible adaptive-chart rung while preserving the
  geometric support policy.
- A nonnormal triangular count-error-only chart forced a sharper distinction.
  The selected chart at radius `1.8` has four residual-small values and a count
  estimate of four, but the argument-principle count error is still large. The
  count-deficit shrink radii `r/4, 3r/8, 5r/8` recover only one of the four
  local roots because the smaller nonnormal charts lose quality. For this
  stress type the policy now keeps the parent radius as a candidate alongside
  overlap anchors, recovering and support-certifying all four local roots. The
  refinement ladder is therefore not just "always shrink": count deficits and
  count errors are different chart diagnoses.
- A near-pole rational control adds a different analytic class. Five scalar
  rational components have simple roots inside the unit contour and poles only
  `0.005` outside it. The reduced contour solve recovers all five roots with no
  residual-small spurious values across Loewner vs counted-SS extraction,
  operator vs vector residual normalization, and with/without contour-max
  component scaling. This is useful negative evidence: nearby exterior
  rational singularities can stress quadrature and conditioning, but in this
  dense reduced setting they do not require a new moment update or chart-policy
  branch.
- A residual-Laurent compression control now pins the efficiency story. On a
  rank-deficient radius-10 three-function analytic chart, the initial reduced
  extraction has four inside values and no residual-small roots. The
  moment-realization update uses a rank-two residual basis, builds only three
  candidate columns per side, preserves a three-dimensional physical
  left/right realization, and recovers all twenty roots. The scalar expanded
  RII path sees the same residual rank but forms four per-Ritz candidate
  columns, compresses back to a two-dimensional physical basis, and recovers no
  roots. Thus residual Laurent moments are not just a memory optimization over
  scalar RII; they are the compact realization geometry needed to keep the
  many-root nonlinear state observable.
- A focused literature pass on the missing iteration step did not turn up an
  existing method that is simply "Loewner FEAST" or "SS RII". The nearest
  established machinery is invariant-pair/block Newton refinement, polynomial
  invariant-pair extraction/refinement, and SS-RR reduced extraction. This
  supports the current boundary: use Loewner/Hankel/companion methods as local
  reduced extractors, then use the dual residual Laurent-moment update to
  repair left/right physical spaces when extraction alone is not enough.
- The reduced polynomial extractor now has an optional invariant-pair
  block-Newton refinement rung. On the `n=4`, degree-eight, twenty-root
  nonnormal polynomial control, reduced companion extraction already recovers
  all target roots, while one block-Newton step improves the reduced residuals
  from about `1e-13` to `1e-15` without changing the physical trial/test spaces.
  This is exactly the intended role: a local refinement/check after reduced
  extraction, not a replacement for the residual Laurent update.
- The generic analytic reduced extractor now has an optional two-sided scalar
  Newton cleanup on the small reduced NEP. On the radius-20 three-function
  analytic control, this is not useful before the residual Laurent update, but
  after one update it improves the max residual from about `8e-6` to `8e-10`.
  It also increases residual-small spurious values, so it is a cleanup/refinement
  rung only; it must be paired with count, contour, matching, and spurious
  diagnostics rather than treated as convergence by itself.
- The generic analytic reduced extractor now also has a true invariant-pair
  block-Newton prototype. Its residual is the contour integral
  `∮ Tred(z) X (zI-S)^-1 dz`, and its Newton matrix uses the corresponding
  Fréchet derivative in `(X,S)` plus the lifted-pair gauge constraint. This is
  the geometrically correct analytic analogue of the polynomial invariant-pair
  Newton step, but the first boundary diagnostic is deliberately conservative:
  on a small radius-4 `sin/cos` chart it accepts a residual-reducing step and
  preserves the exact post-Laurent root set; on the large radius-20
  three-function chart it rejects an improving step after the Laurent update and
  badly degrades the residual-small retained set. The conclusion is that
  analytic block Newton is a local reduced-pair refinement/check, not a
  replacement for chart splitting, support evidence, or residual Laurent
  trial/test-space repair.
- The local-chart triangular analytic comparison reinforces that distinction.
  On the radius-6 triangular coupling-10 control, supervised local charts with
  no Laurent updates recover 10 of 11 roots whether or not scalar Newton cleanup
  is enabled. With two residual Laurent updates, the same chart setup recovers
  all 11 roots, again with or without scalar cleanup. The missing capability in
  weak/non-normal charts is therefore trial/test-space repair, not scalar root
  polishing.
- Chart-merge support is now tracked as a spurious diagnostic. On the same
  radius-6 triangular coupling-10 control, unioning all residual-small local
  chart values gives 15 candidates containing all 11 true roots. Requiring
  support from at least two chart centers reduces this to 12 candidates while
  retaining all 11 true roots. Requiring support from three centers gives 10
  candidates and drops one true root. Support is therefore useful evidence, but
  not a standalone pruning law; it should be combined with local counts,
  residuals, and chart geometry.

## Linear SS-RII Control Result

The linear control is more than a smoke test. For `T(z)=zI-A`, the
invariant-pair residual is

```text
R = X*S - A*X
```

and the SS-RII contour correction used in this experiment is

```text
Q_k = contour_integral mu(z)^k *
      (X - (zI-A)^-1 R) * inv(zI-S) dz .
```

If `(X,S)` is exact, `R=0` and the moments are exactly `Q_k = X*S_mu^k` in the
scaled coordinate. If `(X,S)` is approximate, `(zI-A)^-1 R` is the same
residual-inverse/Newton correction used by linear FEAST, now applied to the
finite realization rather than to a diagonal list of Ritz values.

The important consequence is that higher moments provide effective subspace
width without increasing the number of right-hand sides solved at each contour
node. In the diagonal control, four solved right-hand sides and `K=3` moments
recover the same ten-dimensional invariant subspace that ordinary FEAST needs
twelve solved right-hand sides to represent. The two-sided projected Hankel is
therefore the square reduced problem for this moment-expanded realization, in
the same sense that Rayleigh-Ritz is the square reduced problem for ordinary
FEAST.

This gives us a concrete reduction requirement for nonlinear work: any
candidate higher-moment NLFEAST update must become the formula above when
`T(z)=zI-A`.

## Angles Of Attack

- Treat the contour moments as a realization problem for the multiplication
  operator on the residue quotient space. The Hankel/SS step is then a
  Ho-Kalman-style realization; the iteration should update a balanced or
  otherwise well-gauged realization of this operator.
- Avoid committing to monomial coordinates. Even after scaling to
  `mu=(z-c)/r`, high powers give a Vandermonde-like lifted stack that becomes
  badly conditioned for many eigenvalues in a small physical dimension. The
  likely replacement is an orthogonal rational/polynomial basis on the contour,
  or an Arnoldi-like basis built from the moment sequence itself.
- Preserve physical observability. A small pair residual `T(X,S)` is not
  enough if eigenvectors of `S` are nearly annihilated by the first block `X`;
  scalar residuals then look bad because `X*y` is tiny. Any production method
  needs a gauge/minimality condition that keeps scalar eigenvectors visible in
  the physical space.
- Add a true dual/Petrov-Galerkin nonlinear extraction. The linear
  `dual_gen_feast!` model says the robust nonnormal path is left/right filtering
  plus biorthogonalized reduced equations. The moment analogue should carry a
  left realization or left physical test space and solve
  `Y' * T(lambda) * X`, rather than relying solely on eigenvalues of the small
  multiplication matrix `S`. The current determinant prototype supports this
  direction but still needs a production-grade reduced NEP solver interface.
- Distinguish "mathematically minimal" from "numerically observable". The
  radius-20 scalar sine case can have small first-block and lifted pair
  residuals while scalar residuals plateau around `1e-6`--`1e-5`, which is a
  direct sign that the realization is too ill-conditioned for scalar
  extraction at stricter tolerances.
- Add an explicit deflation/spurious policy. Current NLFEAST-style code mostly
  carries all inside Ritz values and filters only in diagnostics. For moment
  methods, spurious states are part of the next realization and can harm the
  update; a production method needs rank, contour, residual, and possibly trace
  history criteria for pruning or replacing them.
- Treat residual normalization as part of the chart. The stiff analytic
  `exp(z)-1` mixture demonstrates that a small `norm(T(lambda)x)/norm(T(lambda))`
  can be a pseudospectral statement rather than evidence of a true root when
  unrelated components dominate `norm(T(lambda))`. Reduced determinant count,
  local chart consistency, and problem-aware backward error must be visible
  diagnostics.
- Use block Newton as the local model, but not necessarily as the final
  algorithmic form. Its value is that it tells us the correct tangent space and
  gauge constraints for `(X,S)`. A cleaner solver may look like a
  contour-filtered, balanced realization update rather than an explicit dense
  Newton solve.
- If the nonlinear geometry stalls, backtrack to the linear control problem:
  linear FEAST with SS moments. There the contour projector and exact reduction
  are understood, so any proposed higher-moment RII update should first be
  explainable as a stable SS-moment version of linear FEAST before being ported
  back to nonlinear invariant pairs.

## Current Algorithm Boundary

The next serious prototype should be factored into explicit stages:

1. Accumulate right moments `Q_k` with contour-scaled coordinates.
2. Observe them with a left probe `W` to form projected Hankel matrices.
3. Extract a minimal realization `(X, S)` and diagnostics from the small pencil.
4. Apply an update that produces corrected Markov parameters, not just corrected
   scalar Ritz vectors.
5. Control the realization gauge so the small multiplication operator is a
   numerically meaningful representative of the same invariant pair.
6. Only after the gauge is stable, deflate, retain, or replace states using
   rank/residual/history evidence.

The linear control says stages 1--3 are sound. The first nonlinear polynomial
tests say stage 4 is viable when it is treated as a realization update rather
than scalar RII. The scalar stress tests say stage 5 is not optional: without a
stable gauge, small pair residuals can hide unusable scalar Ritz values. Stage 6
should not become ad-hoc scalar pruning unless the natural realization tools
stall.

The first implementation pass of this boundary is now factored into
`pipeline.jl` for the analytic experiments. The code has explicit
experiment-layer objects for:

- `ContourChart`: contour center/radius plus chart-local scaling policy.
- `TrialSpaces`: left/right physical spaces and their singular diagnostics.
- `ReducedExtractorConfig`: counted SS, Loewner, determinant, and reduced
  solver knobs.
- `ResidualUpdateConfig`: residual Laurent moment update knobs.

This is deliberately still an experiment API. `run.jl` remains the demo and
stress-test driver, while `pipeline.jl` owns the stable chart/trial-space/update
wrappers and chart-merge diagnostics. The purpose is to make the algorithm
stages visible enough to compare extractors, chart policies, and update rules
without prematurely committing to a public solver interface.

`experiment_matrix.jl` is the next layer of separation. It keeps the
cross-problem comparison harness out of the historical prototype file and
returns structured rows for:

- Regular polynomial controls.
- Deficient polynomial controls.
- Many-root low-dimensional polynomial controls.
- Clustered near-multiple polynomial controls.
- Rational NEPs with nearby poles.
- Exponential many-root NEPs.
- Many-root scalar analytic controls.
- Multiple-root analytic diagnostics.
- Nonnormal triangular local-chart support diagnostics.

`ALGORITHM.md` records the current candidate generalized moment-NLFEAST
boundary: local dual contour realization, reduced Petrov-Galerkin extraction,
residual Laurent repair, and explicit chart policy. It also lists the known
reductions to FEAST, SS-FEAST, Beyn/SS, and canonical NLFEAST, plus the negative
boundaries that should not be rediscovered as candidate defaults.

`DERIVATION.md` records the current derivation sketch: Keldysh local resolvent
form, scalar RII as a circular-chart Laurent correction, the finite
left/right realization state, and the reductions to linear FEAST, canonical
NLFEAST, Beyn/SS, and higher-moment NLFEAST.

`AUDIT.md` maps the research objective to concrete artifacts, evidence, and
remaining gaps. It is the current guardrail against over-claiming completion.

Current first-pass matrix result:

- The matrix now spans physical dimensions `n=1,2,3,4,5,6,8,15`. This is
  intentional: small low-dimensional many-root cases test the moment geometry,
  while the larger polynomial controls catch ordinary matrix-size issues.
- The regular `n=8` quadratic, deficient `n=15` quadratic, many-root `n=4`
  nonnormal polynomial, and clustered `n=6` near-multiple polynomial controls
  are solved by two-sided reduced extraction already. The residual Laurent
  update improves the deficient quadratic residual from about `6e-13` to
  `8e-16`; block Newton improves some reduced cleanup but is not essential for
  root recovery on these cases.
- The `n=5` rational nearby-pole control is also solved cleanly. This is an
  important sanity check because nearby non-eigenvalue singular structure is a
  natural failure mode for contour extraction.
- The global many-root analytic control remains a deliberate failure case:
  one global low-dimensional chart recovers only 26 of 38 roots after one
  Laurent update and leaves residual-small spurious values. This reinforces
  that analytic many-root problems need local charts or a better realization,
  not just scalar cleanup.
- The `n=6` exponential many-root control exposed two separate sensitivities.
  With the Loewner reduced extractor and no component scaling, it recovers all
  18 known roots with nearest-root error below `1e-6`. With counted SS on the
  same unscaled problem, it misses several lattice roots. With the Loewner
  extractor but `component_scaling=:contour_max`, it returns 18 residual-small
  values displaced by about `2e-2` from the known root lattice. This is a useful
  warning: residual-small reduced roots are not enough in high dynamic-range
  analytic NEPs, and naive contour-max component scaling can change the
  numerical extraction geometry even though it leaves the exact roots invariant.
- The focused exponential Hankel chart/gauge sweep clarifies the failure.
  Plain Hankel, balanced Hankel, and multi-offset Hankel can recover all 18
  true exponential roots on the global chart, but they also produce
  residual-small spurious candidates. Scalar Newton cleanup polishes those
  spurious candidates rather than removing them, and count-capping the Hankel
  rank drops true roots before it cleanly removes extras. Shifted, balanced
  shifted, and Chebyshev shifted realizations recover only a small subset of the
  roots on this global chart. Loewner is the only clean global realization in
  the current sweep.
- Rational coordinate variants do not repair that global chart. The inverse
  coordinate loses all 18 roots, Möbius Hankel coordinates recover 17/18 or
  18/18 depending on shift but keep residual-small spurious candidates, and
  shifted/Chebyshev Möbius variants recover only a few roots. On the same
  physical trial/test spaces, `loewner_counted` returns exactly 18/18 with no
  spurious values. The practical conclusion is that rational coordinates remain
  a chart diagnostic/escalation rung, while Loewner realization or local chart
  splitting is the cleaner response for oversized analytic charts.
- Local charts change the picture. Supervised root-centered local charts with
  radii `1.2` and `2.0` recover all 18 exponential roots with both counted SS
  and Loewner; requiring support from at least two chart centers prunes the
  merged candidates down to exactly the 18 true roots. The same local SS run
  succeeds with and without `component_scaling=:contour_max`. This means the
  scaling/extractor failure is primarily an oversized-chart realization problem,
  not evidence that Hankel moments are unusable for exponential NEPs.
- The multiple-root `sin^2` diagnostic matches all unique roots but returns
  algebraic duplicates as residual-small values. This is useful evidence for
  the escalation ladder, but not something to overfit into the lower-rung
  default algorithm yet.
- The triangular local-chart support run recovers all true roots with
  support>=2 while retaining a small number of extra candidates. Support is
  therefore a retention diagnostic before it is a pruning rule.
- The radius-20 three-function adaptive chart solve is also stable across
  reduced-extractor families. Running the same target-domain weak-support
  refinement with Loewner counted extraction and counted SS/Hankel extraction
  gives exact final target-domain support in both cases: Loewner reaches 38/38
  with 158 centers and counted SS reaches 38/38 with 159 centers. Clustering
  the final retained values across the two extractors with cross-extractor
  support>=2 again returns exactly 38/38. This is stronger than Loewner-layout
  stability alone: the retained set is not an artifact of either one Loewner
  interpolation circle or one reduced realization algebra.
- A first retention-score diagnostic now separates three different notions of
  evidence on the same radius-20 case. Final target-domain support>=1 and
  support>=2 both retain exactly 38/38 roots, while support>=3 retains only
  30/38 because the chart cover is not uniformly triple-overlapping. At the
  chart level, three local records have argument-principle count deficits and
  count errors above `1e-2` even though the merged support>=2 set is exact and
  the maximum local residual among good records is about `1.4e-13`. Optional
  scorecard checks also attach the already-established Loewner-layout and
  reduced-extractor agreement. This confirms that support threshold, local count
  consistency, residual quality, chart geometry, and extractor/layout agreement
  are distinct diagnostics; none should be promoted to a standalone pruning law.
- The three-function retention policy now has an oracle-free completion check.
  The target count is computed from the full analytic operator by the argument
  principle, and support-2 retention is complete only when the retained
  target-domain count matches that reliable contour count. Exact roots are
  still used in the experiment after the decision, as validation that the
  retained values are the intended ones.
- The same idea now drives an adaptive stopping diagnostic. Starting from the
  coarse radius-20 three-function grid, the count-driven loop adds weak
  target-domain candidate centers until support-2 retained values match the
  full-operator contour count. It stops after the same two refinements as the
  fixed-round prototype and validates against 38/38 known roots, but the stop
  condition itself uses only computed count/support evidence.
- A scalar delay control removes the exact-root oracle entirely. The harness
  deliberately returns an empty known-root list for
  `f(z)=z + 0.4 - 2exp(-z)`, while the full-operator argument-principle count
  on the radius-6 contour is three. The count-driven loop retains three
  support-certified values and stops by count completion, so this path now has
  a real oracle-free numerical control rather than only "exact roots used after
  the fact" validation.
- A three-component delay control pushes the same no-oracle path into a
  nonnormal left/right setting. The components use different delay parameters
  and are coupled by an upper-triangular operator. On the radius-6 contour the
  full-operator argument-principle count is nine, support retention returns
  nine values, and the loop stops without exact-root validation or local
  multiplicity probes.
- A near-pole rational control extends oracle-free count completion to a
  different analytic class. Five rational components have poles just outside
  the unit contour and are coupled through the same triangular nonnormal model.
  At pole gap `0.01`, the full-operator argument-principle count is reliable,
  support retention returns five values, and the loop stops without exact-root
  validation or local multiplicity probes. The stricter gap `0.005` remains a
  useful boundary stress for quadrature reliability rather than a clean
  oracle-free regression: with the stricter count tolerance used by this
  diagnostic, the policy reports `:target_count_unreliable` instead of
  accepting a contour placed too close to an exterior pole.
- A duplicate-delay triangular control removes the exact-root oracle from the
  multiplicity branch as well. Two identical copies of
  `f(z)=z + 0.4 - 2exp(-z)` with nonnormal upper-triangular coupling have full
  contour count six on the radius-6 contour, but only three geometric retained
  values. Local cluster counts assign multiplicity two to each retained value,
  so the algebraic count is satisfied without exact roots or explicit
  Jordan-chain data. This is the first oracle-free evidence that the
  algebraic/geometric retained-count distinction is not just an artifact of
  analytic validation lists.
- Applying the count-driven loop to the nonnormal triangular control exposes
  and resolves an important algebraic/geometric distinction. The full
  determinant count is algebraic and returns 12, while the support-2
  target-domain clusters contain 11 unique values because two scalar components
  share a root. Small local contour counts around each retained cluster assign
  multiplicity two to the coincident root, so the multiplicity-weighted retained
  count satisfies the algebraic count without duplicating scalar values.
  These local multiplicity probes are only used on the algebraic/unique
  mismatch branch; simple-root count completion skips them.
- The same multiplicity rule also works on a true repeated-root analytic
  control. For `sin(z)^2` on the radius-10 contour, the full determinant count
  is 14 while the retained set contains seven unique roots. Local cluster counts
  assign multiplicity two to every retained root, giving algebraic completion
  without requiring explicit Jordan-chain extraction in the lower rung.
- The no-oracle count-driven branch now includes non-triangular dense controls:
  a 2x2 coupled two-delay NEP and a 3x3 dense multi-delay NEP. These are useful
  because the determinant is no longer a simple product of scalar components,
  yet the policy still completes from the full-operator argument-principle count
  and chart support rather than exact roots.

## Nonlinear Experiment Plan

1. Done: use the projected two-sided Hankel extraction as the default nonlinear
   experiment path. The one-sided full Hankel should remain only as a reference
   implementation and diagnostic.
2. Done: re-run the existing polynomial nonlinear problems with identical
   configurations under one-sided and two-sided extraction: deficient
   quadratic, butterfly, and a small polynomial problem with more eigenvalues
   than probe columns.
3. Done for polynomial tests: compare three update modes using the same extracted
   realization: repeated projected Hankel extraction, shifted realization
   update, and a small invariant-pair Newton refinement after the contour
   correction.
4. In progress: add diagnostics that separate wanted states from retained transient states:
   scalar residuals, pair residual restricted to wanted states, pair residual on
   all retained states, and retained-rank singular-value ratios. Ritz-value
   persistence across iterations is now printed for nonlinear comparison runs.
5. In progress: make gauge control part of the algorithmic state. The scalar
   experiments show diagonal balancing and observable-eigen scaling are useful
   in different failure modes; Chebyshev moments plus observable-eigen scaling
   are currently the cleanest non-scalar radius-10 chart.
6. In progress: compare stronger realization-theoretic gauges. Schur gauge
   alone is not enough, balanced Ho-Kalman/Hankel extraction alone is not
   enough, and orthogonal moment bases still need observability control.
7. In progress: implement the true dual-FEAST analogue for nonlinear moments:
   left/right contour filtering, biorthogonal physical bases, and a
   Petrov-Galerkin reduced NEP extraction. The first determinant/argument
   principle prototype solves the diagonal `sin/cos` radius-20 failure and the
   reduced polynomial controls. A first scalar Ritz-vector RII loop confirms
   the two-sided nonlinear correction formula, and the new residual Laurent
   moment version expresses the same correction as a compact left/right basis
   update. A non-polynomial diagonal-similarity control now confirms the same
   update outside polynomial companion structure. The remaining task is to
   replace the fragile determinant power-sum extractor with a robust reduced
   NEP solver, then combine rank/count diagnostics, gauge control, and repeated
   reduced-NEP extraction into a coherent algorithm rather than an
   experiment-specific update.
8. In progress: turn reduced-NEP extraction into an algorithmic component. Polynomial
   problems can use reduced companion/QZ solves; generic analytic problems need
   counted reduced SS/Hankel extraction, reduced NLFEAST/SS, or NLEIGS-style
   reduced solvers. The current best analytic prototypes use argument-principle
   counting with counted SS/Hankel or Loewner root extraction on the reduced
   NEP. Loewner now has global interpolation-layout, supervised local-chart
   layout, unsupervised grid-chart spacing, and adaptive weak-support refinement
   sweeps on a clean exponential control, a nonnormal triangular control, and a
   larger radius-20 three-function analytic control. The current boundary is
   sharper: target-domain weak-support candidate centering repairs missing roots
   without refining outside-domain local roots, while final target-contour
   membership removes supported local-chart candidates that are valid nearby
   roots but outside the requested domain. A three-layout Loewner-radius sweep
   on the repaired radius-20 three-function cover gives exact cross-layout
  support, and a Loewner-vs-counted-SS rerun gives exact cross-extractor
  support, so the retained set is not an artifact of one interpolation circle
  or one reduced-extractor algebra. A first automatic retention wrapper now
  escalates exact-but-count-stressed support-2 retained sets to those agreement
  checks and then returns `:accept_with_chart_warnings` when both pass; the
  global in-target Loewner artifact follows the complementary branch and
  requires cross-layout support before accepting residual-small values. The
  policy now also records chart-specific actions: do not raise a support
  threshold unless the cover is dense enough, add weak target candidate centers
  when target-domain support is missing, and split or shrink count-stressed
  charts before strict acceptance. The radius-20 three-function path now uses
  the full-operator argument-principle count for target completion and for a
  first count-driven adaptive stopping diagnostic rather than the known-root
  list. This is still an experiment policy, but it
  turns the escalation ladder into explicit chart actions rather than hidden
  post-processing. The score diagnostic now returns a concrete `plan` object:
  retained support-2 target roots, weak target centers to add, and count-stressed
  chart records with four overlapping child-chart suggestions at half radius.
  This is deliberately conservative; it makes chart refinement reproducible. A
  blind half-radius child split fails on the first count-stressed radius-20
  chart, but a candidate-centered split that reuses the parent chart's
  residual-small values as centers recovers the parent's five locally counted
  roots, with four of them receiving support from at least two child centers.
  A nonnormal triangular count-driven run now pins the complementary algebraic
  multiplicity rule: count completion is not the same as unique-root support
  when roots coincide, but small local contour counts can attach multiplicity
  to retained clusters. The same rule handles `sin(z)^2`, so it is not merely a
  coincident-component workaround. The scalar delay control complements this by
  showing simple-root count completion without exact roots, and the dense 2x2
  and 3x3 delay controls now show weak-support refinement and residual-small
  outside-domain diagnostics without relying on triangular determinant
  structure. The multi-delay triangular control exercises the same
  count-completion path with nonnormal left/right spaces, the near-pole
  rational control exercises it with exterior rational singularities, and the
  duplicate-delay triangular control does the same for the multiplicity branch.
9. In progress: add the invariant-pair/block-Newton refinement rung explicitly for
   reduced polynomial and small dense analytic problems. This should be treated
   as a local refinement/check on a reduced NEP, not as a replacement for the
   FEAST-style residual Laurent update. The polynomial reduced-companion path
   now has an invariant-pair Newton rung. The generic analytic path has both a
   two-sided scalar Newton cleanup and a true contour-residual invariant-pair
   Newton prototype. The analytic block step is useful as a local check, but the
   current boundary diagnostic rules it out as a large-chart retention/update
   mechanism.
10. Done for the first exponential control: formalize the role of local rational
   coordinates. The companion problem suggests why finite polynomial problems
   behave better: the enlarged linear state gives a global finite coordinate
   system. Analytic NEPs with infinitely many roots need local finite
   realizations, so nested contours or rational bases may be the natural
   replacement for one global companion. The first rational-coordinate boundary
   diagnostic says inverse/Möbius moment coordinates are useful probes but not a
   clean replacement for Loewner/local-chart realization on an oversized global
   analytic chart.
11. Later: add invariant-pair deflation after roots or clusters are reliable
   enough that reconvergence is the actual problem. Treat scalar residual-based
   deflation as a fallback, not the main method.
12. Later: broaden the literature review beyond NEP methods into adjacent
   realization/system-identification and rational Krylov filtering work.

## Design Constraints

- Do not expose the expanded Hankel columns as the public iterative state.
- Keep `K=1` and diagonal `S` behavior aligned with canonical `nlfeast!`.
- Prefer a Schur form for `S` during iteration if it improves conditioning or
  makes `inv(zI - S)` solves cheaper and more stable.
- Keep residual diagnostics scalar-eigenpair friendly, but add invariant-pair
  residual norms because they are the actual convergence measure for this path.
- Treat generic analytic `T(X, S)` as a second phase. It likely needs an
  explicit operator interface, not accidental NEP-PACK compatibility.
- Distinguish the moment coordinate matrix from the physical lambda matrix.
  Hankel extraction should use a well-scaled coordinate; residuals and shifted
  solves must use the true lambda matrix.

## Local References

- `docs/article.tex`, section "Higher Moments", states the current limitation.
- `src/nlfeast.jl` has the canonical diagonal-state NLFEAST-Beyn implementation
  and older `nlfeast_moments!` prototype.
- `src/nlfeast_experimental.jl` has additional moment/SS variants.
- `experiments/moment_rii/LITERATURE.md` records the local NEP-PACK and
  external literature review that should guide API decisions.
- NEP-PACK's local `method_block_SS.jl` shows the standard SS-Hankel extraction.
- NEP-PACK's `compute_MM(nep, S, V)` interface confirms that invariant-pair
  residuals are the right abstraction for nonlinear block states.
