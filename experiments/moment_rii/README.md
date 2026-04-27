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
  without tripling the number of linear-solve right-hand sides.
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
   reduced polynomial controls. The current missing piece is a compact
   iteration/update step for the left and right spaces. A scalar Ritz-vector
   residual update is available conceptually, but it expands the number of
   right-hand sides to the number of reduced roots and therefore does not solve
   the low-dimensional many-root case. The desired update must correct the
   reduced realization or reduced NEP as a whole.
8. Next: turn reduced-NEP extraction into an algorithmic component. Polynomial
   problems can use reduced companion/QZ solves; generic analytic problems need
   either derivative-based argument-principle extraction, reduced NLFEAST/SS, or
   NLEIGS-style reduced solvers.
9. Next: formalize the role of local rational coordinates. The companion
   problem suggests why finite polynomial problems behave better: the enlarged
   linear state gives a global finite coordinate system. Analytic NEPs with
   infinitely many roots need local finite realizations, so nested contours or
   rational bases may be the natural replacement for one global companion.
10. Later: if natural gauge and realization methods stall, inspect NEP-PACK and
   adjacent NEP/SS/Beyn literature for deflation strategies. Treat scalar
   residual-based deflation as a fallback, not the main method.
11. Later: broaden the literature review beyond NEP methods into adjacent
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
- NEP-PACK's local `method_block_SS.jl` shows the standard SS-Hankel extraction.
- NEP-PACK's `compute_MM(nep, S, V)` interface confirms that invariant-pair
  residuals are the right abstraction for nonlinear block states.
