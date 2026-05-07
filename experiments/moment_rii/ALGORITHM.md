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

Equivalently, the current proof-facing description is:

```text
a FEAST-filtered block correction equation for a finite two-sided contour
realization.
```

This phrasing connects the pieces without collapsing them: FEAST supplies the
contour-filtered residual correction, SS/Beyn/Loewner supply the finite
realization, and the Petrov-Galerkin left/right spaces supply the nonnormal
geometry.

The important design choice is to stop treating expanded Hankel columns as the
iterative state. The iterative state is instead:

- a target contour and local spectral coordinate;
- right and left physical spaces `X` and `Y`;
- a reduced extractor for `Y' * T(lambda) * X`;
- a residual Laurent update that repairs `X` and `Y`;
- chart diagnostics and a reproducible policy for retention/refinement.

## Claim Status

This experiment has a coherent candidate answer to the higher-moment NLFEAST
update problem: FEAST-style iteration survives as a two-sided residual-Laurent
repair of physical trial/test spaces, not as scalar RII on every expanded
moment column. This is the algorithmic boundary supported by the current tests.

The claim is deliberately narrower than "a black-box NEP solver." General
analytic NEPs still need chart policy, count diagnostics, extractor agreement,
and occasional escalation to refinement or multiplicity handling. Those are
not hacks around the update; they are the acceptance and realization layers
needed once the problem is no longer a single global linear invariant subspace.

The remaining bar for a publication-level claim is not another ad-hoc solver
trick. It is a tighter proof/novelty pass showing that the residual-Laurent
physical-space repair is genuinely distinct from known contour projection,
Loewner/realization, invariant-pair Newton, and refined Rayleigh--Ritz
formulations, plus broader evidence that the chart policy selects reliable
local realizations across representative NEP classes.

## Reference Algorithm

For one local chart, the experiment-level solver loop is:

```text
given T, T', a circular chart Gamma(c,r), and right/left probes V,W:

1. Build right and left contour-moment trial spaces
       X = orth([int zeta^k T(z)^(-1) V dz]_{k=0}^{K_basis-1})
       Y = orth([int zeta^k T(z)^(-H) W dz]_{k=0}^{K_basis-1})

2. Extract a finite reduced realization of the Petrov-Galerkin transfer data
       Y^H T(lambda) X u = 0
   preferably by a linear small pencil/realization such as counted
   SS/Hankel, Beyn, Loewner, or a companion/QZ construction.

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
`MomentPipelineConfig` over `MomentBasisConfig`, `ReducedExtractorConfig`, and
`ResidualUpdateConfig`, `run_dual_moment_compressed_rii_analytic_iteration`, and
`run_count_driven_policy_diagnostic`. These are experiment objects, not public
`FEASTSolver` API.

## Reduced Realization Boundary

Stages 2--5 deliberately keep reduced realization/extraction and acceptance
policy separate from the FEAST-style update. This mirrors linear FEAST more
than it may first appear: FEAST theory controls the contour projection, the
filtered subspace iteration, and the Rayleigh--Ritz reduction, while the
details of the small reduced solve are mostly an implementation choice.

For moment-NLFEAST the projected analytic object is:

```text
Tred(lambda) = Y^H T(lambda) X.
```

But the practical route should usually not be "globally solve this reduced
analytic NEP." The point of higher moments is precisely to expose a finite
linear realization of the local pole data, so that extraction reduces to a
small linear generalized eigenvalue problem or equivalent small realization
problem:

```text
positive contour moments -> finite realization -> small pencil -> roots
```

This is the practical bridge to the original NLFEAST/PEP idea: polynomial
problems can be linearized explicitly by a companion expansion, while general
analytic problems can be linearized locally by SS/Hankel, Beyn, or Loewner
realizations of contour data. Direct nonlinear solution of `Tred` should be
treated as a fallback, validation tool, or local cleanup step, not as the main
algorithmic promise.

This distinction matters most for problems like sine or delay equations. A
low-dimensional physical operator can contain many eigenvalues inside one
chart. Moment expansion is meaningful only if those moments produce a finite
local realization whose roots can be extracted by a small pencil. If the
extractor cannot produce such a realization, then the problem has fallen back
to hard analytic rootfinding and the FEAST-style update has not bought us a
practical solver.

The extractor layer should therefore be understood as a coordinate choice for
the finite local realization, not as part of the residual-Laurent update
itself:

```text
SS/Hankel, Beyn, Loewner, companion/QZ, or local Newton cleanup after a
linearized realization
```

are interchangeable reduced-realization coordinates when their assumptions
hold. The policy layer then decides whether the extracted roots are credible
using physical residuals, contour counts, support across overlapping charts,
left/right consistency, extractor agreement, and near-pole diagnostics.

This boundary is important for the research claim. The algorithmic finding is
not "we have made every reduced NEP easy." It is:

```text
given a trustworthy reduced local realization, the FEAST-style higher-moment
iteration should repair physical left/right spaces by compressed
residual-Laurent enrichment, then re-extract in whatever reduced coordinate is
appropriate for the local problem.
```

The hard cases are then assigned to the right layer:

- bad or oversized chart: refine, split, or increase overlap;
- unreliable reduced coordinate: switch extractor or compare extractor
  agreement;
- algebraic/geometric mismatch: use local contour counts for multiplicity;
- contour near a pole or singularity: diagnose as unsafe rather than hiding
  the placement error;
- failure to linearize the local pole data into a finite realization: treat as
  an extractor/chart limitation, not as a successful moment-NLFEAST solve.

## Collapsing Outer And Inner Moments

The current prototype still contains an avoidable redundancy:

```text
outer FEAST layer:
    compute contour samples T(z_j)^(-1) B and T(z_j)^(-H) C
    build physical trial/test spaces X,Y

inner extractor layer:
    form Tred(z) = Y^H T(z) X
    compute new contour samples Tred(z_j)^(-1)
    build a reduced SS/Beyn/Loewner pencil
```

This is the same redundancy that motivated the original NLFEAST/Beyn hybrid:
if FEAST has already paid for contour solves at the target quadrature nodes,
then asking Beyn/SS to run a second contour solve on the projected problem is
conceptually ugly, even if the second solve is small.

The practical higher-moment algorithm should collapse these layers. The
primary contour samples should be treated as the transfer data:

```text
R_j = T(z_j)^(-1) B,
L_j = T(z_j)^(-H) C,
G_j = C^H R_j = C^H T(z_j)^(-1) B.
```

From those same samples we can form:

```text
right physical moments:      M_k = sum_j w_j zeta_j^k R_j
left physical moments:       N_k = sum_j conj(w_j) zeta_j^k L_j
small two-sided moments:     H_k = sum_j w_j zeta_j^k G_j
```

The small moment sequence `H_k` is the reduced finite realization. SS/Hankel,
Beyn, or Loewner extraction should build a small linear pencil from `H_k`
directly. Physical right/left Ritz vectors are reconstructed from the physical
moment data `M_k,N_k` and the small realization coordinates. The projected
analytic object `Y^H T(lambda) X` then becomes a validation/refinement object,
not the source of a second contour moment computation.

This is the important practical claim:

```text
moment-NLFEAST should not be
    FEAST projection -> reduced NEP -> Beyn/SS on the reduced NEP.

It should be
    one set of contour solves -> physical spaces + small finite realization
    -> residual-Laurent repair -> updated contour-sample/realization data.
```

After a residual-Laurent update, the same principle should apply. The update
already computes contour responses to compressed residual bases:

```text
T(z_j)^(-1) U_X,      T(z_j)^(-H) U_Y.
```

Those node-local responses should be retained as realization data for the next
extraction whenever possible, rather than discarded and recomputed through
`Tred(z)^(-1)`. The long-term implementation shape is therefore a contour
sample cache owned by the chart/worker:

```text
node z_j owns factorizations/solvers;
node applies them to active right/left probe blocks;
global reduction forms physical moments and small two-sided moments;
extractor builds a small pencil from the reduced moment sequence.
```

The current `ReducedExtractorConfig` path is still useful as a validation and
fallback path, but it should not be the final efficient moment-NLFEAST design.
The final design should make this collapsed one-contour-sample realization the
default.

## Unified Candidate Algorithm

The current final candidate is a **chart-owned contour sample realization
iteration**:

```text
state per chart:
    contour nodes and node-local solvers/factorizations
    active right probe blocks P = [initial probes, residual probes, ...]
    active left probe blocks Q = [initial probes, residual probes, ...]
    node responses T(z_j)^(-1) P and T(z_j)^(-H) Q

extract:
    reduce initial/realization probe responses with positive chart moments
    form a small two-sided transfer realization H_k = Q^H M_k(P)
    solve a small linear pencil / Loewner realization / SS-Hankel problem
    reconstruct physical Ritz data from the same physical moment blocks

certify:
    check physical right/left residuals, contour count, support, and
    extractor/layout agreement

repair:
    compress physical residual blocks to U_X,U_Y
    append U_X,U_Y as new probe blocks in the same chart cache
    reduce those residual probe responses with inverse-Laurent weights
    repair X,Y and re-extract from the updated finite realization
```

In this formulation there is no conceptual inner nonlinear solve. The reduced
analytic object `Y^H T(lambda) X` is still useful, but only as a validation,
cleanup, or fallback coordinate. The main path is:

```text
one contour sample cache -> finite linear realization -> residual-Laurent
cache augmentation -> finite linear realization -> ...
```

This is the cleanest version of the original NLFEAST/Beyn observation: once
the contour solves have been paid for, running Beyn/SS as an inner method on a
projected nonlinear problem wastes the transfer data that the outer contour
already generated.

## RII As A Coordinate Chart

The unifying theoretical interpretation is that RII is not the general state;
it is a special coordinate chart on the contour realization.

```text
linear FEAST:
    T(z)=zI-A gives a global scalar Ritz chart.
    RII is exactly the FEAST filter in that chart.

canonical NLFEAST:
    Keldysh gives a local scalar pole chart near simple eigenvalues.
    Scalar nonlinear RII is a local coordinate update in that chart.

SS/Beyn/Loewner:
    positive contour moments expose a finite realization with gauge freedom.
    Extraction is a choice of realization coordinates.

higher-moment NLFEAST:
    scalar RII on expanded moment columns is generally the wrong chart.
    The coordinate-free update is physical residual-Laurent cache
    augmentation followed by re-extraction of the finite realization.
```

This explains both the success and the failure modes. RII works miraculously
when the local realization admits a scalar Ritz coordinate chart. In
higher-moment charts, the realization has gauge freedom and possibly more
roots than physical dimensions; the update must act on physical residual
subspaces, not on an arbitrary expanded scalar Ritz list.

## Next Research Steps: Fused Realization And RII Charts

The fused contour-sample view changes the next research target. The important
question is no longer "how do we solve the projected nonlinear problem
`Y^H T(lambda) X` well?" The projected nonlinear problem should mostly be a
validation/refinement object. The main question is:

```text
how much of Beyn/SS/Loewner extraction can be built directly from the same
contour samples that FEAST already computed?
```

The next concrete steps are:

1. **Define the sample cache algebra.** For one chart, make the node-local data
   explicit:

   ```text
   R_j(P) = T(z_j)^(-1) P,
   L_j(Q) = T(z_j)^(-H) Q,
   G_j(Q,P) = Q^H T(z_j)^(-1) P.
   ```

   From this, define all physical moments, small two-sided moments, and
   residual-update moments by weighted reductions over the same node cache.

2. **Reproduce Beyn/SS without inner solves.** Add a diagnostic that compares:

   ```text
   old path: build X,Y -> form Tred(z) -> run Beyn/SS on Tred
   fused path: use G_j and physical moments from the original contour samples
   ```

   on the same chart and contour nodes. The target is not just matching roots;
   it should also match reconstructed physical Ritz vectors/residuals up to
   the expected basis/gauge transformations. This directly tests the original
   motivation: the inner reduced contour solve is wasted information.

   Current status: `run_fused_contour_sample_realization_diagnostic` pins the
   first version on a diagonal sine/cosine/shifted-sine chart with eight roots
   in a three-dimensional physical problem. The fused projected Hankel path
   recovers all eight roots from the original contour samples, and the
   redundant path that forms `Tred` and performs an inner reduced SS contour
   extraction recovers the same set. The focused test is:

   ```text
   just test --slow 'fused contour samples'
   ```

   `run_fused_polynomial_contour_sample_realization_diagnostic` adds the
   polynomial lower rung on the deficient quadratic MatrixMarket control. The
   fused projected Hankel path recovers the same four roots as the redundant
   inner reduced SS path and the companion reference. This is important because
   polynomial problems have an exact companion linearization, so the fused
   realization can be checked against the linear FEAST/RII ladder rather than
   only against scalar analytic rootfinding.

3. **Make extraction linear by construction.** The extractor should consume a
   small moment/Loewner sequence and return a small linear pencil or equivalent
   finite realization. Direct solution of `Y^H T(lambda) X` should be a
   cleanup or validation option, not the default route.

4. **Fuse residual-Laurent repair into the same cache.** After computing
   compressed residual bases `U_X,U_Y`, the node responses

   ```text
   T(z_j)^(-1) U_X,
   T(z_j)^(-H) U_Y
   ```

   should extend the same chart cache. The next realization should be formed
   by augmenting the active probe blocks and reducing the retained node data,
   not by building a new projected NEP and sampling it from scratch.

5. **Study RII as a realization-coordinate chart.** In the linear case, scalar
   RII is a coordinate chart where the residual correction is exactly the
   FEAST rational filter. In canonical NLFEAST, Keldysh makes that scalar chart
   locally valid near a simple pole. In higher-moment NLFEAST, the scalar Ritz
   chart is generally not valid because the local object is a finite
   realization with gauge freedom. The research task is to show that
   residual-Laurent repair is the coordinate-free version of the same FEAST
   correction:

   ```text
   scalar RII chart            -> one pole / one Ritz coordinate
   SS/Beyn/Loewner chart       -> finite realization coordinates
   residual-Laurent repair     -> physical-space correction invariant to
                                   realization-coordinate changes
   ```

6. **Use hard analytic cases as extractor tests, not update tests.** Problems
   like sine or delay equations should stress whether the fused moment data
   admits a usable finite local realization. If it does, the algorithm should
   reduce to a small linear pencil. If it does not, the failure belongs to the
   chart/extractor layer rather than the FEAST residual update.

This is the likely final form of the algorithmic story:

```text
one contour sample cache
  -> physical left/right spaces
  -> small two-sided linear realization
  -> physical residual certification
  -> compressed residual-Laurent cache augmentation
  -> repeat
```

Current implementation status: `ContourSampleCache` in `pipeline.jl` is the
first explicit experiment object for this boundary. The fused analytic and
polynomial diagnostics now build their physical moments and projected transfer
moments from that cache, rather than assembling one-off local moment arrays.
`run_fused_cache_residual_augmentation_diagnostic` also pins the update side:
augmenting the same cache with compressed residual probe responses and reducing
those responses with inverse-Laurent weights produces the same repaired
physical spaces as the explicit residual-Laurent update.

One important detail fell out of this diagnostic. The same node samples can
serve both extraction and residual repair, but the reductions are different:

```text
positive powers of zeta       -> finite realization / extractor moments
inverse-Laurent powers        -> residual repair moments
```

Simply appending residual probes to the cache and reducing them as positive
realization moments gives the wrong space. This supports the two-moment-role
story in a concrete implementation way: the cache is unified, while the
reduction applied to each probe block depends on whether that block is an
initial realization probe or a residual correction probe.

This is still an experiment layer object, but it pins the direction for the
final implementation: chart-owned contour samples first, extractors and update
reductions second.

## Moment Roles

The algorithm uses two moment families with different jobs.

- Basis/extractor moments are positive contour moments of `T(z)^(-1)`. They
  build a finite transfer realization in the local chart. Hankel/SS, Loewner,
  companion/QZ, and related reduced extractors live here.
- Residual inverse-Laurent moments are FEAST-style enrichment directions for
  physical residual subspaces. They repair `X,Y`; they are not themselves the
  persistent Hankel state and are not justified by a naive scalar denominator
  truncation.

The loop is therefore:

```text
extract a local realization in X,Y
  -> compress physical residuals
  -> enrich X,Y by inverse-Laurent residual images
  -> extract a new local realization
```

This is the cleanest current answer to the "moment update" question. It
preserves the FEAST iteration idea while avoiding scalar RII on expanded
moment columns.

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
- Linear SS-FEAST: with `T(z)=zI-A` and higher SS/Hankel moments, the moment
  realization gives a wider effective invariant subspace than the number of
  physical right-hand sides. The residual inverse step still reduces to the
  FEAST contour filter on extracted Ritz vectors, so SS-style moments and FEAST
  iteration are compatible in the linear case.
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

The resulting story is a compatibility ladder. Linear FEAST and linear
SS-FEAST admit a true RII/filter identity. Canonical NLFEAST is the `K=1`
nonlinear residual-inverse rung explained locally by Keldysh. Polynomial moment
problems can be checked against companion FEAST or invariant-pair refinement.
Fully analytic moment-NLFEAST does not appear to admit a literal scalar RII on
expanded moment columns without artificial deflation; the residual-Laurent
two-sided physical-space repair is the FEAST-style iteration that survives in
that setting.

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
  residual-Laurent update all recover the same 20 target roots, and the
  polynomial-native good values stay within `1e-6` of the companion-FEAST good
  values. This pins the polynomial rung between linearized FEAST and the
  generic analytic charted method.
- Dual extraction control: a dual-sensitive polynomial now records that
  Galerkin one-sided extraction can produce false reduced Ritz data. The
  one-sided run returns 13 inside values with reduced residual near
  `1e-15`, but zero values pass the original residual tolerance, while both
  true dual variants recover the 12 expected target roots.
- Dual residual-Laurent update control: on the same dual-sensitive polynomial
  with a deliberately weak initial basis, the two-sided residual-Laurent repair
  expands both physical spaces from dimension three to six and recovers all 12
  target roots. Updating only the right or left space is not a square
  Petrov-Galerkin reduced NEP without truncating back to the old dimension; that
  truncation recovers zero target roots. This pins "repair both sides" as part
  of the update geometry, not only the extraction geometry.
- Analytic controls: global many-root charts fail in predictable ways, while
  local chart covers plus residual Laurent updates recover the target roots.
- Residual-Laurent compression control: on a rank-deficient analytic chart, the
  moment update recovers all roots with fewer candidate columns and a larger
  observable physical realization than scalar expanded RII, which fails. The
  diagnostic now returns an explicit efficiency scorecard: candidate columns
  saved, candidate-column ratios, basis-size gains, low residual-rank
  completeness, residual improvement above `1e8`, and a flag showing
  scalar-expanded RII is worse despite using more candidate columns. This is
  evidence about the update geometry, not only allocation count:
  scalar-expanded RII fixes a bad scalar gauge for the finite realization,
  while residual-Laurent repair acts on the low-rank physical residual
  subspace.
- Low-rank residual-basis equivalence: on the same analytic chart, compressing
  residual directions reduces the residual rank and candidate columns while
  preserving the updated right/left physical spaces to roundoff projection
  gaps. This pins residual compression as an efficiency transformation, not a
  numerical branch of the algorithm.
- Residual-coordinate invariance: the reduced Ritz residual columns can be
  mixed by independent nonsingular right/left coordinate changes before
  compression, and the residual-Laurent update still recovers the same roots
  with roundoff-level projection gaps between the updated physical spaces. This
  pins the first proof obligation in `DERIVATION.md`: the update depends on
  the physical residual subspace, not on the scalar Ritz coordinate list used
  to present it.
- Scalar Laurent truncation boundary: a small linear diagnostic verifies that
  finite denominator expansion alone is not the proof of the method. Even with
  small chart coordinates, truncating only `1/(z-lambda)` can fail when
  `T(z)^(-1)` has interior poles. This supports the current proof direction:
  finite update order must be argued through the captured contour realization,
  recurrence/rank, and the exact lower-rung FEAST identities.
- Positive realization recurrence: a diagonal linear transfer diagnostic
  verifies that positive contour moments satisfy `M_k = X*S^(k-1)*C`, expose
  the expected realization rank when enough probe directions are used, and
  filter outside-contour modes to roundoff. This pins the extraction-moment
  half of the proof story separately from residual inverse-Laurent enrichment.
- Exact realization closure: on an exact linear left/right eigenspace, the
  physical residual blocks have zero numerical rank and the residual-Laurent
  update adds no physical directions. The updated `X,Y` spaces have
  roundoff-level projection gaps from the original spaces. This pins the
  realization-closure proof obligation in the simplest setting.
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
- Sparse linear subspace/residual boundary: the same sparse diagonal control
  now records that the extracted right/left subspaces can already have
  projection gaps near roundoff before the residual-Laurent update, while the
  physical residuals are still too large. One update improves the physical
  residual scale by more than `1e4` and recovers all retained eigenpairs. The
  update repairs residual quality, not merely subspace angle. This rejects a
  pure angle-improvement theorem as insufficient.
- Sparse nonlinear gallery smoke: a FEAST-native sparse quadratic polynomial
  gallery operator `T(z)=z^2I-D^2` flows through the same moment pipeline using
  a sparse prototype and in-place gallery materializer. The target positive
  roots are known exactly, and one residual-Laurent update recovers them with
  strict residuals. This pins the nonlinear sparse gallery boundary without
  claiming sparse nonlinear worker storage or benchmark performance.
- Sparse symbolic-reuse smoke: on the same sparse quadratic gallery control,
  one reusable UMFPACK factor per side is initialized once and then refreshed
  numerically across contour nodes with symbolic reuse. Dense solve result
  buffers are allocated once per side and reused across repeated updates. The
  resulting residual-Laurent update matches the generic sparse update and
  remains stable across a repeated update. This pins the no-store fixed-pattern
  sparse rung without claiming complete sparse workspace reuse or benchmark
  performance.
- Sparse stored-factor smoke: the same sparse linear control caches
  contour-node sparse factorizations for the residual-Laurent update. The
  cached update reproduces the generic sparse update to roundoff projection
  gaps, and a repeated update reuses the same node factors. This pins the
  stored-factor implementation rung without claiming benchmark-level sparse
  performance.
- Partitioned residual-Laurent update: splitting the residual-update contour
  nodes into four independent partitions, summing the partial Laurent blocks,
  and recompressing produces the same physical right/left trial spaces as the
  serial update to projection gaps near roundoff. This pins the algebra needed
  for persistent worker-owned contour partitions.
- Remote residual-Laurent worker diagnostic: the same update runs on actual
  Julia worker processes with each process retaining the operator closures and
  a stable contour-node subset across two residual-Laurent updates. The reduced
  Laurent blocks match the serial update to roundoff projection gaps. The
  diagnostic also reports setup, update, and worker-local timing metadata as a
  profiling smoke check. This is process-level evidence for the worker model,
  not yet an optimized sparse or benchmarked implementation.
- Sparse remote worker smoke: the persistent worker path also accepts a sparse
  diagonal linear operator through `Tmatrix/Tsolve` closures and reproduces the
  serial residual-Laurent update on the sparse linear control while preserving
  the same lightweight timing shape. This is generic sparse compatibility
  evidence, not sparse factorization reuse.
- Sparse remote stored-factor worker smoke: persistent Julia workers own
  node-local sparse factors for fixed contour subsets. The first update creates
  the expected right/left node factors and node-local solve buffers, the
  repeated update reuses those same factors and buffers, and both remote
  updates match the serial residual-Laurent physical spaces to roundoff
  projection gaps. This pins fixed-node sparse factor and solve-buffer
  ownership across the process boundary without claiming full sparse workspace
  reuse or benchmark-level performance.
- Sparse nonlinear remote stored-factor worker smoke: the same worker-owned
  factor cache also runs on the sparse quadratic polynomial gallery control.
  This verifies that nonlinear sparse gallery materialization, residual
  Laurent repair, and persistent worker-owned contour factors compose correctly
  on a known-root NEP.
- Sparse Schrodinger gallery smoke: the FEAST-native moving-boundary
  Schrodinger sparse gallery operator is exercised on the established
  `center=-35`, `radius=4.2` region at small size. A full-operator contour
  count estimates three target eigenvalues, and the residual-Laurent update
  improves all three action residuals to the strict threshold. This is
  realistic sparse NEP evidence, not a scaling benchmark.
- Fused Schrodinger domain-decomposition diagnostic: a one-dimensional
  finite-difference Schrodinger operator is split into subdomains and local
  interiors are eliminated, producing a rational Schur-complement NEP on the
  interface variables. The default hard chart keeps the contour below the first
  eliminated-interior pole, so the sampled interface NEP remains analytic. On
  the current `full_n=207`, `interface_n=15` control, the fused contour
  realization identifies the 13 physical roots in the chart, but raw extracted
  values have large full residuals; reduced `Tred` Newton cleanup recovers all
  13 with residuals below `1e-8`. This is the better Schrodinger DD stress
  than the small moving-boundary smoke because it exercises a genuinely
  nonlinear Schur complement while preserving a full linear reference spectrum.
  The companion refinement sweep records the practical boundary: at 64 and 96
  contour nodes the fused realization has the right rank but the chart is still
  underresolved after cleanup (`12/13` and `11/13` matched respectively in the
  current run), while 128 nodes plus reduced `Tred` cleanup recovers all 13.
  Reduced cleanup is therefore necessary but not a replacement for sufficient
  contour sampling on this harder rational interface problem.
- Fused Schrodinger domain-decomposition scale smoke: the DD interface operator
  is now assembled from independent local interior blocks rather than a single
  dense eliminated-interior solve. On the current scale smoke,
  `full_n=2111`, `interface_n=63`, and 64 local blocks of size 32 give a
  compression ratio above 30. The local-block assembly matches the dense Schur
  complement and derivative to roundoff on a small reference check, and the
  fused algorithm again recovers all 13 target roots after reduced cleanup
  (`cleanup_required`). This is still not an HPC implementation, but it
  verifies the algorithm is compatible with the subdomain-local operator shape.
- Fused Schrodinger/DD baseline comparison: the experiment now reports direct
  full `SymTridiagonal` eigvals and sparse full linear FEAST beside the fused
  compressed interface solve. On the small default DD case all three recover
  the same 13 roots; full sparse FEAST is faster at this size, which is the
  expected baseline warning rather than a failure. On the larger local-block
  smoke, full sparse FEAST is also still faster locally. The capability claim is
  therefore not a small-machine speed claim: the fused DD path is evidence that
  the nonlinear interface formulation can be solved automatically, with the
  scaling proposition depending on larger subdomain-local/HPC workflows where
  full global solves or linearizations become the bottleneck.
- Fused Schrodinger/DD packet-defect diagnostic: the Lean-facing checkpoint is
  now implemented numerically. A high-resolution fused/refined run defines a
  reference packet projector; lower-node raw and `Tred`-refined packet
  projectors are compared by splitting the defect into reference-packet-visible
  and invisible parts. On the current DD case, the successful 128-node cleanup
  reduces packet-visible defect to roundoff, while the underresolved 64-node
  case retains large packet-visible defect. This distinguishes "cleanup fixed
  the chart coordinate" from "the chart is still underresolved" more sharply
  than residual norms alone.
- Fused Schrodinger/DD packet policy diagnostic: the packet-visible status now
  drives a first action policy. `packet_visible_defect` means increase contour
  nodes or refine the chart, `packet_invisible_acceptance_gap` means the packet
  is essentially right but extraction/acceptance needs work, and
  `accepted_visible_removed` accepts the solve. On the current DD sweep this
  policy rejects 64 nodes, flags 96 nodes as an acceptance/refinement gap, and
  selects 128 nodes.
- Sparse Schrodinger remote stored-factor smoke: the same realistic sparse
  Schrodinger control runs through persistent worker-owned contour partitions.
  The worker factors and node-local solve buffers are created once on the first
  update, reused on the repeated update, and the remote physical spaces match
  the serial update. This extends the distributed stored-factor evidence beyond
  diagonal controls.
- Moment benchmark harness: `just bench-moment` runs a BenchmarkTools-backed
  warmup-plus-timed comparison for the small sparse Schrodinger serial path and
  remote stored-factor worker path. It prints timing, allocation, correctness,
  and factor/solve-buffer reuse counters. This is a regression/profiling
  harness for the current implementation rung, not publication-level scaling
  evidence.
- Adjacent implementation recheck: RSRR, SLEPc CISS, and Riesz-projection
  methods all support the current separation between contour-node solves,
  reduced extraction, and selection/observability policy. They do not appear to
  replace the residual-Laurent physical-space repair step, so they are evidence
  for keeping extractor and policy as explicit layers rather than evidence for
  dropping FEAST-style iteration.
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
  now lowers into the shared `MomentPipelineConfig` used by the local chart
  sweep and central analytic iteration, so basis/extraction/update choices are
  explicit experiment objects rather than loose keyword bundles.
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
- `just test --slow 'residual-coordinate invariant'`
- `just test --slow 'scalar Laurent truncation alone'`
- `just test --slow 'positive moments expose'`
- `just test --slow 'closes on exact realization'`
- `just test --slow 'sparse symbolic reuse'`
- `just test --slow 'sparse nonlinear gallery operator'`
- `just test --slow 'sparse Schrodinger gallery'`
- `just test --tags distributed 'remote contour workers'`
- `just test --tags distributed 'sparse residual Laurent update runs on remote contour workers'`
- `just test --tags distributed 'sparse remote workers reuse'`
- `just test --tags distributed 'sparse nonlinear remote workers reuse'`
- `just test --tags distributed 'sparse Schrodinger remote workers reuse'`
- `just bench-moment`
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
  `count_stressed_chart_refinement` now records this as an explicit policy
  split: count deficits use smaller overlapping radii around residual
  candidates, while count-error-only charts preserve parent-radius candidates.
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
- The experiment has strong dense reduced-problem evidence and first sparse
  remote implementation rungs, but no publication-level sparse/distributed
  moment-NLFEAST scaling evidence.
- The reduced-extractor interface is still experimental and should not be
  promoted before more problem classes are covered.
- The literature pass supports this boundary, but adjacent rational Krylov,
  realization, and system-identification literature should be reviewed before
  making publication-level novelty claims.
