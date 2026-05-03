# Generalized Moment-NLFEAST Candidate

This is the current experiment-level algorithm boundary. It is not a public
`FEASTSolver` API yet.

The derivation sketch is in `DERIVATION.md`; this file summarizes the candidate
algorithm, evidence, and known boundaries.

## Candidate Answer

The most defensible solution so far is:

```text
generalized moment-NLFEAST =
    local dual contour realization
  + reduced Petrov-Galerkin NEP extraction
  + residual Laurent correction of left/right physical trial spaces
  + explicit chart policy driven by support, counts, residuals, and agreement
```

The important design choice is to stop treating expanded Hankel columns as the
iterative state. The iterative state is instead:

- a target contour and local spectral coordinate;
- right and left physical spaces `X` and `Y`;
- a reduced extractor for `Y' * T(lambda) * X`;
- a residual Laurent update that repairs `X` and `Y`;
- chart diagnostics and a reproducible policy for retention/refinement.

## Reference Algorithm

For one local chart, the experiment-level solver loop is:

```text
given T, T', a circular chart Gamma(c,r), and right/left probes V,W:

1. Build right and left contour-moment trial spaces
       X = orth([int zeta^k T(z)^(-1) V dz]_{k=0}^{K_basis-1})
       Y = orth([int zeta^k T(z)^(-H) W dz]_{k=0}^{K_basis-1})

2. Solve/extract the reduced Petrov-Galerkin NEP
       Y^H T(lambda) X u = 0
   using a realization coordinate such as counted SS/Hankel or Loewner.

3. Score physical Ritz data
       x = X u,  y = Y v
   by right/left residuals, chart membership, local counts, support across
   overlapping charts, and agreement across layouts/extractors when requested.

4. If the retained set satisfies the reliable full-operator contour count,
   accept the geometric roots. If the algebraic count is larger than the
   retained geometric count, assign local cluster multiplicities by small
   contour counts.

5. If roots are missing but weak residual-small candidates exist, add local
   chart centers or escalate chart radii/overlap according to policy.

6. If extraction alone is insufficient, repair the physical spaces with the
   residual Laurent update and repeat from step 2:
       X <- orth([X, int zeta^(-k) T(z)^(-1) U_X dz])
       Y <- orth([Y, int zeta^( k) T(z)^(-H) U_Y dz])
   where U_X and U_Y are low-rank bases for the physical right/left residual
   blocks from the current reduced Ritz data.
```

The central implementation path is currently `ContourChart`, `TrialSpaces`,
`MomentBasisConfig`, `ReducedExtractorConfig`, `ResidualUpdateConfig`,
`run_dual_moment_compressed_rii_analytic_iteration`, and
`run_count_driven_policy_diagnostic`. These are experiment objects, not public
`FEASTSolver` API.

## Core Update Formula

Work in a local circular chart

```text
z = c + r*zeta,      lambda = c + r*alpha.
```

After reduced Petrov-Galerkin extraction, we have right and left physical
Ritz vectors `x_j = X*u_j`, `y_j = Y*v_j`, values `lambda_j`, and residual
blocks

```text
R_X = [T(lambda_j) x_j],
R_Y = [T(lambda_j)' y_j].
```

Compress these blocks,

```text
R_X ~= U_X C_X,      R_Y ~= U_Y C_Y,
```

then repair the physical trial/test spaces by adding residual inverse Laurent
moments

```text
Q_X,k = contour_integral zeta^(-k) T(z)^(-1) U_X dz,
Q_Y,k = contour_integral zeta^( k) T(z)^(-*) U_Y dz,
```

for `k = 1, ..., K_update`, with the same quadrature contour used by FEAST.
The next physical spaces are compressed bases for

```text
span([X, Q_X,1, ..., Q_X,K_update]),
span([Y, Q_Y,1, ..., Q_Y,K_update]).
```

The scalar FEAST/RII factor appears because on a circular chart

```text
1 / (z - lambda_j) = (1 / r) * sum_k alpha_j^k zeta^(-k-1)
```

for target values inside the contour. The update therefore keeps the FEAST
residual-inverse geometry, but applies it to the low-rank residual directions
and the local realization instead of carrying one persistent correction column
per scalar Ritz value.

## Reductions

- Linear FEAST: with `T(z)=zI-A`, one moment, and diagonal scalar states, the
  residual Laurent update reduces to the usual FEAST/RII residual-inverse
  correction.
- Dual FEAST: nonnormal reduced extraction uses independent left and right
  contour-filtered physical spaces. On the dual-sensitive polynomial control,
  true dual and biorthogonal dual extraction recover all 12 target roots, while
  one-sided Galerkin extraction has tiny reduced residuals but zero acceptable
  original NEP residuals. This pins the Petrov-Galerkin condition as structural
  evidence, not presentation.
- SS-FEAST: with `T(z)=zI-A` and higher moments, the same correction acts on a
  finite SS/Hankel realization. This gives effective subspace width larger than
  the number of physical right-hand sides.
- Beyn/SS: if the residual Laurent update is disabled, the method is a contour
  realization extractor. Counted SS/Hankel, companion/QZ, and Loewner are
  interchangeable reduced extractors for this stage.
- Canonical NLFEAST: with one local chart, one moment, and diagonal scalar
  extraction, this matches the existing NLFEAST/Beyn-RII path on the
  one-root-per-component limit. The compressed residual-Laurent update gives
  the same roots and residual scale without carrying scalar-expanded update
  columns as persistent state.
- Moment-NLFEAST: with higher moments, the reduced realization is kept finite
  and local. The method avoids forcing a many-root Hankel realization back into
  a single global diagonal scalar-RII state.

## Evidence

- Linear SS-RII control: documented in `README.md`, showing the update becomes
  the linear FEAST residual correction and recovers a larger invariant subspace
  than the physical probe count.
- Linear dual RII reduction: for `T(z)=zI-A`, the scalar residual-inverse
  contour step is algebraically identical to applying the FEAST contour filter
  to the extracted right and left Ritz vectors. The diagnostic pins this on a
  diagonal many-root control and a nonnormal Grcar control with projection gaps
  near roundoff.
- Polynomial controls: regular, deficient, many-root low-dimensional, and
  near-multiple polynomial cases are recovered by two-sided reduced extraction;
  polynomial invariant-pair Newton is useful as a reduced cleanup. The
  polynomial bridge diagnostic now compares the degree-eight nonnormal
  polynomial against FEAST on the `32 x 32` companion pencil: companion FEAST,
  polynomial-native initial extraction, reduced block-Newton cleanup, and the
  residual-Laurent update all recover the same 20 target roots. This pins the
  polynomial rung between linearized FEAST and the generic analytic charted
  method.
- Dual extraction control: a dual-sensitive polynomial now records that
  Galerkin one-sided extraction can produce false reduced Ritz data. The
  one-sided run returns 13 inside values with reduced residual near
  `1e-15`, but zero values pass the original residual tolerance, while both
  true dual variants recover the 12 expected target roots.
- Analytic controls: global many-root charts fail in predictable ways, while
  local chart covers plus residual Laurent updates recover the target roots.
- Residual-Laurent compression control: on a rank-deficient analytic chart, the
  moment update recovers all roots with fewer candidate columns and a larger
  observable physical realization than scalar expanded RII, which fails. The
  diagnostic now returns an explicit efficiency scorecard: candidate columns
  saved, candidate-column ratios, basis-size gains, low residual-rank
  completeness, and a flag showing scalar-expanded RII is worse despite using
  more candidate columns.
- Low-rank residual-basis equivalence: on the same analytic chart, compressing
  residual directions reduces the residual rank and candidate columns while
  preserving the updated right/left physical spaces to roundoff projection
  gaps. This pins residual compression as an efficiency transformation, not a
  numerical branch of the algorithm.
- Residual-Laurent update ladder: on the radius-20 upper-triangular nonnormal
  analytic chart cover, reduced extraction alone (`iterations=0`) recovers
  `34/44` target roots, one residual-Laurent update recovers `42/44`, and two
  updates recover `44/44`. This pins the algorithm-family interpretation:
  Beyn/SS-style extraction is the lower rung, while the FEAST residual update
  repairs weak/nonnormal physical trial-test spaces when extraction alone is
  insufficient.
- Sparse pipeline smoke: a sparse diagonal linear operator `T(z)=zI-A` flows
  through the same experiment pipeline using sparse `Tmatrix` and sparse
  backslash solves. Initial reduced extraction sees the eight target values but
  misses the strict residual tolerance; one residual-Laurent update recovers
  all eight with residuals near machine precision. This is implementation
  evidence for the generic operator boundary, not a sparse-optimized moment
  solver.
- Partitioned residual-Laurent update: splitting the residual-update contour
  nodes into four independent partitions, summing the partial Laurent blocks,
  and recompressing produces the same physical right/left trial spaces as the
  serial update to projection gaps near roundoff. This pins the algebra needed
  for persistent worker-owned contour partitions.
- Remote residual-Laurent worker diagnostic: the same update runs on actual
  Julia worker processes with each process retaining the operator closures and
  a stable contour-node subset across two residual-Laurent updates. The reduced
  Laurent blocks match the serial update to roundoff projection gaps. This is
  process-level evidence for the worker model, not yet an optimized sparse or
  benchmarked implementation.
- Sparse remote worker smoke: the persistent worker path also accepts a sparse
  diagonal linear operator through `Tmatrix/Tsolve` closures and reproduces the
  serial residual-Laurent update on the sparse linear control. This is generic
  sparse compatibility evidence, not sparse factorization reuse.
- Canonical NLFEAST limit control: a three-component one-root-per-component
  rational NEP compares the existing `nlfeast!`, scalar-expanded residual RII,
  and compressed residual-Laurent update. All three recover the same three
  target values to residuals near machine precision, pinning the `K=1` bridge.
- Rational controls: poles placed just outside the contour are stable across
  Loewner vs counted-SS extraction and residual normalization in the current
  dense reduced setting.
- Oracle-free near-pole rational control: rational components with exterior
  poles close to the target contour satisfy the count-driven path without exact
  roots when the pole gap is large enough for reliable argument-principle
  quadrature.
- Loewner/extractor agreement: the radius-20 three-function analytic solve is
  stable across Loewner layouts and across Loewner vs counted SS/Hankel
  extraction after adaptive chart refinement.
- No-oracle extractor agreement: the scalar delay control now compares
  Loewner-counted and counted SS/Hankel extraction under the same count-driven
  policy without supplying exact roots. Both extractors retain three values,
  the cross-extractor supported set has size three, and the target count is the
  full-operator argument-principle count. This pins Beyn/SS/Loewner as
  interchangeable reduced-extraction coordinates on a small solver-like rung.
- Retention policy: the current scorecard separates support thresholds, target
  membership, residual size, local count stress, layout agreement, and extractor
  agreement; the policy returns explicit retained roots and chart-refinement
  suggestions. The count-driven stress harness now carries the reusable
  `CountDrivenPolicyConfig` object so grid spacing, support threshold,
  refinement depth, local chart radii, optional radius-ladder stages, residual
  tolerance, and count tolerance are treated as one experiment policy rather
  than incidental keyword clutter. Its companion `CountDrivenNumericsConfig`
  now lowers into the shared `ReducedExtractorConfig` and `ResidualUpdateConfig`
  objects used by the local chart sweep and central analytic iteration, so
  extraction/update choices are explicit experiment objects rather than loose
  keyword bundles.
- Oracle-free control: a scalar delay NEP with no exact-root list stops from
  the full-operator argument-principle count alone, validating that the
  count-driven loop is not secretly supervised by analytic roots.
- Oracle-free extractor swap: the same scalar delay control also completes with
  `CountDrivenNumericsConfig(extractor=:ss_counted)`, pinning that the
  count-driven numerics object can switch from Loewner-counted to counted
  SS/Hankel reduced extraction without changing the stopping policy or using a
  root oracle.
- Oracle-free nonnormal control: three distinct delay components with
  triangular nonnormal coupling retain all nine algebraic/geometric values from
  support and contour count alone. The pinned stress version starts from a
  coarse cover with all nine residual-small candidates visible but only three
  support-2 retained roots; adding weak target candidates completes the
  count without exact roots.
- Oracle-free two-delay control: a single quasipolynomial component
  `z+a-b exp(-tau z)-c exp(-sigma z)` has no closed-form root oracle in the
  harness. The count-driven loop retains three roots from the full-operator
  argument-principle count alone, exercising multiple delay scales inside one
  scalar component.
- Oracle-free coupled two-delay control: a dense 2x2 NEP couples two
  two-delay components through analytic off-diagonal terms. Its determinant is
  not the product of scalar component functions, and the count-driven loop
  still retains six roots from the full-operator count alone.
- Oracle-free fully coupled mixed diagnostic: a larger dense 2x2 coupled
  two-delay contour has 16 counted target roots and no exact-root oracle. The
  base cover sees 20 residual-small values, retains only 15 target roots,
  contains a weak target cluster, and records local count warnings. One
  weak-center refinement retains all 16 target roots while keeping the
  outside-domain residual-small values out of the accepted set.
- Oracle-free dense multi-delay stress: a fully dense 3x3 NEP couples two
  scalar delay components and one two-delay component through analytic
  off-diagonal terms. The base cover sees 11 residual-small values for a
  nine-root target count, retains only five target roots, and has four weak
  target clusters. One weak-center refinement retains all nine target roots
  while leaving the two outside-domain residual-small values as diagnostics.
- Oracle-free multiplicity control: a duplicate delay NEP with triangular
  nonnormal coupling has algebraic contour count six, three retained geometric
  values, and local count multiplicity two on each value without any exact-root
  list.
- Matrix-valued mixed-diagnostic control: the triangular analytic multiplicity
  case now records the coarse-cover diagnostic split directly. The base cover
  has more residual-small values than retained support-2 roots, weak target
  clusters, and a selected local count warning; the final accepted state is
  algebraic-count complete through local multiplicity rather than by accepting
  every residual-small candidate.

Representative tests:

- `just test --preset moment-core`
- `just test --slow 'dual linear RII'`
- `just test --slow 'low-rank compression preserves update'`
- `just test --tags distributed 'remote contour workers'`
- `just test --tags distributed 'sparse residual Laurent update runs on remote contour workers'`
- `just test --preset moment-heavy 'residual Laurent update'`
- `just test --preset moment-count`
- `just test --slow 'analytic block Newton'`
- `just test --slow 'rational coordinates'`
- `just test --slow 'adaptive retention score'`
- `just test --slow 'candidate-centered split'`
- `just test --slow 'near-pole rational'`
- `just test --slow 'without root oracle'`
- `just test --slow 'oracle-free nonnormal delay'`
- `just test --slow 'fully coupled mixed diagnostics'`
- `just test --slow 'dense multi-delay weak support'`
- `just test --slow 'oracle-free multiplicity'`

## Negative Boundaries

- A true analytic invariant-pair block Newton step is not the missing global
  update. It is geometrically correct and useful as a local reduced-pair check,
  but on the radius-20 many-root analytic chart it worsens retention.
- Rational moment coordinates do not replace Loewner/local charts on the
  exponential many-root global chart. Inverse coordinates lose the roots;
  Mobius coordinates recover many roots but keep spurious candidates.
- Support threshold alone is not a pruning law. Support `>=3` can drop true
  roots when the chart cover is not dense enough.
- A count-driven policy must not over-accept a sparse chart cover just because
  some residual-small values exist. The coupled two-delay radius-12 control
  with spacing `4.0` retains only 10 of 12 counted roots after refinement and
  correctly stops as an unresolved defect.
- Blind geometric fill-in is not a principled fix for that boundary. Adding
  supplemental half-grid centers to the sparse coupled cover increases the
  number of chart solves but still plateaus at 10 of 12 retained roots. Missing
  count with no weak residual candidate needs a better chart/probe diagnostic,
  not arbitrary cover densification.
- Enlarging the local chart radius is a principled repair for that specific
  boundary: using radii `(1.2, 3.0)` on the same sparse coupled cover exposes
  an additional weak candidate, adds one center, and then retains all 12 counted
  roots. The likely policy rung is adaptive overlap/radius selection when
  count deficits occur near the outer contour, not unconditional grid fill-in.
- The experiment now has a radius-ladder policy prototype: run the usual
  count-driven refinement with small local charts first; if it stops with an
  unresolved count deficit, rerun with a larger chart radius/overlap schedule.
  On the sparse coupled two-delay control this turns a diagnostic `10/12`
  failure into a certified `12/12` solve without exact roots. The ladder is now
  attached to `CountDrivenPolicyConfig.chart_radii_stages` and runs through the
  same `run_count_driven_policy_diagnostic` path as the other no-oracle stress
  cases, so radius escalation is a policy rung rather than a one-off runner.
- Scalar residuals alone are not acceptance evidence in high dynamic-range
  analytic NEPs. Local count, support, target membership, and extractor/layout
  agreement are needed.
- Repeated-root/Jordan machinery should remain an escalation rung unless it
  becomes necessary for ordinary retained-set quality.
- Contours placed too close to poles or singularities are not a case to "solve"
  by force. The correct lower-rung behavior is to expose unreliable
  argument-principle counts and request a safer contour/chart.

## Current Algorithmic Ladder

1. Build local right/left spaces from contour moments in a scaled chart.
2. Solve the reduced Petrov-Galerkin NEP using the chart's extractor.
3. Retain target-domain candidates with residual, count, and support evidence.
4. If target support is weak, add weak target candidate centers and rerun local
   chart extraction.
5. If local counts are stressed but target support is exact, request
   Loewner-layout and reduced-extractor agreement; accept with chart warnings
   only when both agree.
6. If charts remain count-stressed, split or shrink those charts before strict
   acceptance. Count-deficit charts use smaller candidate radii; count-error
   charts with no deficit keep the parent radius as an option, which is
   important for nonnormal local charts. In both cases use the selected parent
   chart's residual-small values as candidate centers, but keep overlapping
   parent/child cover anchors so local roots receive support from more than one
   chart.
7. Use full-operator contour counts as the target-completion criterion, with
   exact roots reserved for experiment validation. Support-2 retention is
   complete only when the retained target-domain count matches a reliable
   argument-principle count; local reduced counts remain chart-quality
   diagnostics.
8. Run adaptive chart refinement until the count criterion is satisfied, the
   target count is unreliable, no new weak target centers are available, or a
   conservative round limit is reached. This makes the loop solver-like: known
   roots are not used to decide when to stop.
9. If the algebraic target count is larger than the number of supported unique
   retained values, assign multiplicities to retained clusters with small local
   contour counts around each cluster. The algebraic retained count is the sum
   of those local counts. Escalate only if this multiplicity-weighted count
   still fails, or if any local count is unreliable. Do not force unique-root
   support clustering to satisfy an algebraic count by inventing duplicate
   scalar values. Skip these local multiplicity probes entirely when unique
   retained support already equals the algebraic target count.
10. Use block Newton, rational coordinates, or deflation only as local
   refinement/escalation rungs, not as the central update.

## Remaining Gaps

- The split/shrink chart policy is now a reproducible local rung, but not yet a
  globally optimized chart-cover algorithm. A blind half-radius child cover
  fails on the selected count-stressed radius-20 charts. Residual-root-centered
  charts recover the local roots but can leave weak support; adding overlapping
  parent/child cover anchors certifies all local roots on the current
  count-stressed charts. A nonnormal triangular count-error-only chart also
  shows that aggressive shrinking is not always correct: preserving the parent
  radius certifies the local roots, while the same shrink radii fail.
- The three-function retention policy now has an oracle-free completion path:
  the target count comes from the full analytic operator by the argument
  principle, while exact roots are used only after the decision to validate the
  experiment.
- A count-driven adaptive refinement diagnostic now stops the radius-20
  three-function solve from that contour count. It reaches the same retained
  set as the fixed two-round experiment, but the stop condition is computed
  evidence rather than an experiment-script round count or known-root list.
- A scalar delay control deliberately returns no exact roots to the harness.
  On the radius-6 contour, the full-operator count is three and the
  count-driven loop retains three values with no validation oracle and no local
  multiplicity probes.
- A multi-delay triangular control lifts this no-oracle simple-root path from
  scalar to nonnormal dimension three. The full-operator contour count is nine,
  support retention returns nine values, and no multiplicity probes are needed.
- A two-delay scalar control tests a quasipolynomial component with two
  exponential delay scales. The full-operator contour count is three on the
  radius-6 contour, support retention returns three values, and no exact-root
  list or multiplicity probes are used.
- A coupled two-delay 2x2 control removes diagonal/triangular determinant
  factorization from the no-oracle path. The analytic off-diagonal terms move
  the determinant roots away from the scalar component roots. The full-operator
  contour count is six on the radius-6 contour, and support retention returns
  six values with no validation oracle.
- Pushing that coupled control to radius 12 gives a useful chart-cover
  boundary. With a moderately dense cover, refinement recovers all 12 counted
  roots. With spacing `4.0` and small local radii, the policy sees only 10
  supported roots after adding weak centers and stops with
  `:count_multiplicity_or_unresolved_defect` instead of silently accepting an
  incomplete solve. The missing roots lie near the outer target contour and are
  repaired by allowing a larger local radius `3.0`, which gives enough overlap
  to generate the needed weak center. The radius-ladder policy captures this as
  an explicit two-stage `CountDrivenPolicyConfig`: diagnose with small charts,
  then retry with larger overlap only when the count deficit remains unresolved.
  Each stage uses the same extractor/update configs, isolating the chart-policy
  change from numerical extraction settings.
- A near-pole rational triangular control exercises the same no-oracle path
  with meromorphic components whose poles lie just outside the target contour.
  With pole gap `0.01`, the full-operator count is reliable, support retention
  returns five values, and no special rational chart branch is needed. With
  pole gap `0.005`, the count diagnostic is deliberately unreliable and the
  policy stops before accepting the retained set. Arbitrarily close poles are a
  contour-placement failure to diagnose, not a limitation this algorithm should
  try to hide.
- A duplicate-delay triangular control removes the root oracle from the
  algebraic/geometric mismatch too. Two identical delay components give
  algebraic count six on the same radius-6 contour. The retained set has three
  geometric values, and local cluster counts assign multiplicity two to each
  retained value, so algebraic completion is certified without exact roots or
  explicit Jordan-chain construction.
- The same count-driven loop on the nonnormal triangular case resolves the
  algebraic/unique mismatch by local multiplicity counts: the full determinant
  count is `12`, the supported unique retained set has `11` values, and one
  retained cluster has local multiplicity two. This keeps the retained set
  geometric while satisfying the algebraic contour count.
- A true repeated-root analytic control, `sin(z)^2`, exercises the same rule
  without relying on coincident components. The full contour count is `14`,
  the retained set has seven unique roots, and every retained cluster receives
  local multiplicity two.
- The experiment has strong dense reduced-problem evidence, but no sparse or
  distributed moment-NLFEAST implementation.
- The reduced-extractor interface is still experimental and should not be
  promoted before more problem classes are covered.
- The literature pass supports this boundary, but adjacent rational Krylov,
  realization, and system-identification literature should be reviewed before
  making publication-level novelty claims.
