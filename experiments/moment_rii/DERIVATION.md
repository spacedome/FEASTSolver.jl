# Residual-Laurent Moment Update Derivation

This note records the current experiment-level derivation. It is not a proof of
global convergence. Its purpose is to pin the geometry behind the candidate
algorithm so future experiments do not drift into unrelated heuristics.

## Setup

Let `T(z)` be a regular analytic matrix-valued function and let `Gamma` be a
closed contour that does not intersect the spectrum or other singularities of
`T`. For simple eigenvalues inside `Gamma`, Keldysh theory gives the local
resolvent form

```text
T(z)^(-1) = sum_j (x_j y_j^H) / ((z - lambda_j) gamma_j) + H(z),
```

where `T(lambda_j) x_j = 0`, `y_j^H T(lambda_j) = 0`, `gamma_j =
y_j^H T'(lambda_j) x_j`, and `H(z)` is analytic in the chart. Moment methods
integrate `T(z)^(-1)` against simple basis functions, so the analytic remainder
does not contribute to the ideal contour moments.

Work in a local circular chart

```text
z = c + r*zeta,        lambda = c + r*alpha,        |alpha| < 1.
```

The scalar pole factor has the convergent Laurent expansion

```text
1 / (z - lambda)
  = (1 / r) * 1 / (zeta - alpha)
  = (1 / r) * sum_{k >= 0} alpha^k zeta^(-k-1),
```

valid on the contour `|zeta| = 1`.

## From Scalar RII To A Contour Update

Residual inverse iteration for one approximate nonlinear eigenpair
`(lambda, x)` applies an inverse solve to the residual `T(lambda) x`.
NLFEAST replaces one fixed shift by contour shifts. In local chart coordinates,
the FEAST/RII update is driven by contour integrals of

```text
T(z)^(-1) T(lambda) x * 1 / (z - lambda).
```

Using the Laurent expansion above, the scalar correction can be represented by
moments of `T(z)^(-1)` applied to the residual direction:

```text
integral_Gamma zeta^(-k) T(z)^(-1) T(lambda) x dz.
```

The usual scalar RII form is therefore the `K = 1`, one-column version of a
more general residual-moment correction.

## Finite Realization State

Higher-moment methods such as SS and Beyn do not merely produce unrelated Ritz
vectors. They produce a finite realization of the local transfer behavior of
`T(z)^(-1)` in a chart. The moment-NLFEAST state should therefore not be the
expanded Hankel columns themselves. The experiment uses:

```text
right physical space X,
left physical space Y,
reduced NEP Y^H T(lambda) X,
reduced extractor for that chart,
policy diagnostics for retained values.
```

Given reduced Ritz data, form physical Ritz vectors

```text
x_j = X u_j,        y_j = Y v_j,
```

and residual blocks

```text
R_X = [T(lambda_j) x_j],
R_Y = [T(lambda_j)^H y_j].
```

Compress these residual directions,

```text
R_X ~= U_X C_X,        R_Y ~= U_Y C_Y,
```

because the correction should depend on the residual subspace, not on an
arbitrary scalar enumeration of possibly redundant Ritz values.

## Why The Test Space Is Structural

The reduced equation

```text
Y^H T(lambda) X u = 0
```

is a Petrov-Galerkin stationarity condition: the physical residual
`T(lambda) X u` is orthogonal to the chosen left test space `Y`. If `Y` is a
good approximation to the left spectral subspace, a small reduced residual is
evidence that the physical residual is small in the directions that matter. If
`Y` is replaced by an unrelated one-sided choice, the reduced problem can
annihilate the wrong components and return Ritz data that is internally
consistent but physically false.

This is the nonlinear/moment analogue of why dual FEAST matters for nonnormal
linear problems. The current dual-sensitive polynomial control makes the point
numerically: one-sided Galerkin extraction returns more inside reduced Ritz
values than the target count and has reduced residuals near machine precision,
yet zero values satisfy the original NEP residual tolerance. True dual and
biorthogonal dual extraction recover the full target set. The left contour
moments are therefore part of the realization geometry, not a cosmetic
stabilization option.

The same point appears in the update, not only extraction. On a deliberately
weak dual-sensitive basis, the residual-Laurent correction succeeds when both
`X` and `Y` are repaired. A right-only or left-only correction changes only one
dimension, so a square Petrov-Galerkin reduced NEP can only be formed by
truncating away the newly added side; that truncated one-sided update returns
the same false residual-small data as the initial weak basis. This is why the
candidate update is explicitly two-sided.

## Residual-Laurent Update

The right update adds the chart-local residual moments

```text
Q_X,k = integral_Gamma zeta^(-k) T(z)^(-1) U_X dz,
        k = 1, ..., K_update.
```

The dual left update adds

```text
Q_Y,k = integral_Gamma zeta^(k) T(z)^(-H) U_Y dz,
        k = 1, ..., K_update.
```

The next trial spaces are compressed bases for

```text
span([X, Q_X,1, ..., Q_X,K_update]),
span([Y, Q_Y,1, ..., Q_Y,K_update]).
```

The opposite powers on the left/right updates are the local-chart analogue of a
two-sided contour realization: right samples see poles in `T(z)^(-1)`, while
left samples see the adjoint chart with conjugate orientation. Numerically this
matches the same role as dual FEAST: stabilize nonnormal extraction by giving
the reduced NEP a Petrov-Galerkin pair rather than a one-sided projection.

## What This Update Is Claiming

The update should be read as a finite-dimensional residual-inverse correction
on the physical realization, not as a new contour extractor. The precise
claim, at the current experiment level, is:

```text
Given a local two-sided realization X,Y and reduced Ritz data, replacing the
expanded scalar residual columns by a compressed residual basis U_X,U_Y and
adding inverse Laurent moments of those bases gives the smallest FEAST-style
physical-space repair that is invariant under Ritz-coordinate changes and
compatible with the lower rungs of the FEAST family.
```

There are four pieces to that statement.

1. **Coordinate invariance.** If the same residual subspace is represented by
   `R_X` or by `R_X*C` for any nonsingular coefficient matrix `C`, the update
   space is unchanged after residual compression. This is the core reason to
   update with `U_X` and `U_Y` rather than one column per scalar Ritz value:
   the finite realization has a gauge, while the residual subspace is physical.

2. **Scalar RII compatibility.** If the chart has a diagonal scalar state and
   the residual columns are independent, the correction for a Ritz residual
   `r_j` is formally represented by the same Laurent basis because

   ```text
   1/(z - lambda_j) = (1/r) * sum_{ell >= 0} alpha_j^ell zeta^(-ell-1).
   ```

   This does not mean a denominator-only tail bound proves the finite update:
   `T(z)^(-1)` itself has poles inside the contour, so truncating only the
   scalar denominator can fail as an approximation to the full RII integrand.
   In the linear case the FEAST/RII algebraic identity below collapses the
   correction back to the usual rational filter exactly. In higher-moment
   cases, finite order must instead be justified through the captured contour
   realization and its recurrence/rank, not through `|alpha|^K` alone.

3. **Higher-moment realization compatibility.** For a finite pole realization,
   the moment blocks satisfy a recurrence determined by the small
   multiplication operator. Once the realized rank is captured, adding more
   raw Hankel columns does not create a new physical state; it changes the
   realization coordinates. The residual-Laurent update respects that by
   repairing `X,Y` and then resolving the reduced NEP, instead of persisting a
   particular expanded Hankel basis as the nonlinear iteration state.

4. **Two-sided Petrov-Galerkin compatibility.** The reduced equation is
   `Y^H T(lambda) X u = 0`. A right-space correction without a matching
   left-space correction changes the trial manifold but not the test
   stationarity condition, and conversely for a left-only correction. The
   nonlinear nonnormal controls show this is not just conditioning: one-sided
   reduced residuals can be tiny while the original residuals are false.

This is weaker than a global convergence theorem. It is stronger than a
heuristic: it identifies the invariant object being updated, the exact lower
reductions it must satisfy, and the approximation introduced by finite
Laurent order.

## Two Different Moment Roles

The experiment now needs to keep two moment roles separate.

1. **Realization/extraction moments** are the usual positive contour moments

   ```text
   M_k = integral_Gamma mu(z)^k T(z)^(-1) V dz.
   ```

   Under Keldysh, these are Markov parameters of the pole transfer function.
   In a semisimple chart they factor as

   ```text
   M_k = X * S_mu^k * C,
   ```

   so block Hankel/Loewner extraction is a finite-realization problem. This is
   where rank, recurrence, gauge, observability, and Loewner interpolation
   live. Once the realized rank is captured, additional positive moments change
   the realization coordinates or improve conditioning; they are not the
   nonlinear iterative state.

2. **Residual inverse-Laurent moments** are enrichment directions

   ```text
   Q_X,k = integral_Gamma zeta^(-k) T(z)^(-1) U_X dz,
   Q_Y,k = integral_Gamma zeta^( k) T(z)^(-H) U_Y dz.
   ```

   These are not a replacement Hankel extractor and should not be justified as
   a naive finite expansion of the full RII denominator. They are FEAST-style
   inverse-resolvent images of the physical residual subspaces. After these
   directions repair `X,Y`, the algorithm returns to the first layer and
   re-extracts a reduced Petrov-Galerkin realization.

This separation explains the negative scalar truncation diagnostic. The
denominator expansion

```text
1/(z - lambda) = (1/r) * sum_k alpha^k zeta^(-k-1)
```

is the local language connecting the enrichment to scalar RII, but finite
`K_update` is not a quadrature proof for the whole product
`T(z)^(-1) T(lambda)x /(z-lambda)`. The poles of `T(z)^(-1)` also lie inside
the contour. The reliable finite-dimensional object is the realized pole
transfer function exposed after the enriched physical spaces are re-extracted.

Thus the current algorithm is best viewed as an alternating loop:

```text
realize/extract local transfer data in X,Y
  -> form physical residual subspaces
  -> enrich X,Y by inverse-Laurent residual images
  -> realize/extract again
```

This is closer to FEAST subspace iteration than to one-shot Beyn/SS. It also
explains why `K_update` is an enrichment parameter, while basis/extractor
moment order is a realization parameter.

## Candidate Proof Obligations

A publication-quality version should prove or explicitly assume the following
local statements.

- **Residual-subspace invariance:** the compressed update is unchanged, up to
  basis equivalence and truncation tolerance, under nonsingular changes of
  reduced Ritz coordinates and under duplicate scalar parameterizations of the
  same residual directions.
- **Local realization truncation:** finite Laurent order must be justified by
  the rank and recurrence of the captured contour realization, not only by the
  scalar denominator expansion. A negative diagnostic now verifies why: even
  with small `|alpha_j|`, denominator-only truncation can fail when
  `T(z)^(-1)` has interior poles. The proof should treat `K_update` as a
  residual-enrichment parameter and treat basis/extractor moments as the layer
  where realization rank and recurrence are measured. It needs constants
  involving the realized pole geometry, rank decisions, quadrature error, and
  the angle between the enriched physical spaces and the target residue spaces.
- **Realization closure:** if the current `X,Y` exactly contain the right and
  left spectral residue spaces for all roots inside the chart, then the
  residual blocks vanish and the update adds no physical directions. If the
  residuals are small, the added directions are first-order correction
  directions for the physical realization.
- **Petrov-Galerkin necessity:** for nonnormal or defective-looking reduced
  data, physical residual acceptance must be measured against both trial and
  test spaces. One-sided extraction can be a useful diagnostic, but it is not a
  valid replacement for the two-sided update unless additional symmetry or
  normality assumptions are present.

The current experiment has numerical evidence for all four statements, but not
formal constants or convergence rates. This is the main remaining proof gap.

## Reductions

Linear FEAST:

For `T(z) = zI - A`, the residual is `(lambda I - A) x`. The contour inverse
`(zI - A)^(-1)` and the scalar pole expansion recover the standard FEAST
residual-inverse correction. Higher moments correspond to an SS/Hankel
realization of the same invariant subspace.

The scalar dual RII identity is exact. For a Ritz pair `(lambda, x)`,

```text
x - (zI - A)^(-1) (lambda I - A) x = (z - lambda) (zI - A)^(-1) x.
```

Therefore the RII integrand
`(x - (zI - A)^(-1) r)/(z - lambda)` is exactly the FEAST filter
`(zI - A)^(-1) x`; the adjoint equation gives the same identity for the left
trial space. The diagnostic `run_linear_dual_rii_reduction_diagnostic` pins
this on both diagonal and nonnormal Grcar controls.

Canonical NLFEAST:

For one local chart, one moment, and one root per physical component, the
residual block has the same rank as the scalar Ritz list. The compressed update
and scalar-expanded RII update span the same correction space. The experiment
pins this with `run_canonical_nlfeast_limit_diagnostic`.

Beyn/SS:

If the residual update is disabled, the method is only a contour realization
extractor. Hankel/SS, companion/QZ, and Loewner extractors all fit this stage.
They differ in realization coordinates, not in the outer FEAST-style update.

Moment-NLFEAST:

When a local chart needs more moments than physical probe columns, scalar RII
on expanded columns is the wrong state model. The reduced realization should be
kept finite, and only the physical left/right spaces should be repaired by
low-rank residual Laurent moments.

## RII Compatibility Ladder

The central research question is not just "can we solve the roots?" It is
whether the higher-moment algorithms preserve the special FEAST/NLFEAST feature:
residual inverse iteration gives a genuine iterative correction instead of only
a one-shot contour extractor.

The current answer is a ladder rather than one universal scalar formula.

Linear FEAST:

For `T(z)=zI-A`, scalar RII is exactly the contour filter identity recorded
above. This is the cleanest case: FEAST is subspace iteration with a rational
filter, and residual inverse iteration is another algebraic form of the same
filter.

Linear SS-FEAST:

Higher SS moments can be used without abandoning the FEAST correction. The
moments form a finite realization of the same linear spectral projector, and
the RII residual identity still applies to extracted Ritz vectors. In this
setting an "SS-FEAST" is possible: SS/Hankel supplies a wider effective
realization than the number of physical right-hand sides, while the residual
correction reduces to ordinary FEAST on the represented invariant subspace. The
experiment pins this with the linear SS-FEAST control and the dual linear RII
reduction.

Canonical NLFEAST:

For `K=1`, NLFEAST uses the same residual inverse idea locally. Keldysh theory
explains why this can work surprisingly well: near a simple eigenvalue the
analytic NEP has a local pole structure that behaves like a linearized
eigenproblem. This is the miracle that makes scalar nonlinear RII useful.

Polynomial moment problems:

Polynomial NEPs sit between linear and fully analytic problems. A companion
linearization always exists, so a literal linear FEAST/RII story exists in a
larger space. Polynomial-native moment extraction and invariant-pair refinement
can be viewed as compressed ways to avoid exposing that companion space. This
is why polynomial controls are the right lower rung for testing any proposed
moment update: the method should agree with companion FEAST while keeping the
physical-space realization compact.

Fully analytic moment problems:

For a general analytic NEP with many roots in a low-dimensional physical space,
there is no fixed finite companion linearization that makes scalar RII on a
`K*m` expanded Ritz list canonical. Literal scalar RII on all moment-expanded
candidates forces an artificial deflation/truncation step, which destroys the
very information higher moments were added to expose. This is where the
candidate algorithm becomes genuinely different: it keeps the finite contour
realization local to a chart, extracts a reduced Petrov-Galerkin NEP, and
repairs the physical left/right spaces by residual Laurent moments.

The issue is not only memory. Expanded scalar RII chooses one scalar Ritz list
as coordinates for a finite realization. In a low-dimensional many-root chart,
that list is a poor gauge: multiple roots may share physical directions,
residual blocks may be rank-deficient, and the expanded scalar columns can be
more numerous while observing less of the useful physical realization. The
rank-deficient analytic compression diagnostic pins this numerically:
scalar-expanded RII uses more candidate columns and still misses target roots,
while compressed residual-Laurent repair uses the residual subspace itself,
adds fewer columns, and recovers the full target set.

Thus the residual-Laurent update is best understood as the FEAST/RII geometry
that survives the higher-moment analytic setting. It is not classical scalar
RII on expanded moment columns. It is a two-sided residual-inverse repair of the
physical realization, designed to reduce to scalar RII when the chart has only
one moment/root direction and to reduce to FEAST when the operator is linear.

## Diagnostics And Boundaries

The update is not a complete black-box solver by itself. The contour realization
can still be poor if:

- the contour is too close to an eigenvalue or pole;
- one global chart contains too many roots for a stable monomial realization;
- local charts are too sparse to give support for every target root;
- scalar residuals are small for values outside the requested target contour;
- algebraic multiplicity differs from the number of retained geometric values.

The current algorithmic answer is therefore explicitly charted:

```text
local dual contour realization
  -> reduced Petrov-Galerkin NEP extraction
  -> residual Laurent repair of physical trial/test spaces
  -> count/support/residual/agreement policy
```

This is the clean lower-rung story. Deflation, multiplicity/Jordan data,
invariant-pair Newton, and sparse/distributed implementations remain escalation
or implementation rungs, not part of the minimal update geometry.
