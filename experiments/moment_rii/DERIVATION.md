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
