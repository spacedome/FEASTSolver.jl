# Count And Window Policy

The contour has two distinct meanings.

For a user-fixed target region, the region is part of the mathematical
problem.  Every filter now computes the sampled determinant winding of
`det(zI−H)` from the same node factorizations and rejects a full-operator count
different from the requested realization rank.  This prevents a truncated
Hankel pencil from silently reporting a complete result when additional states
are present.

The sampled winding is a diagnostic, not a proof of resolution.  Integrality
and agreement with the reduced pencil do not exclude phase aliasing.  A
certified fixed-contour result additionally needs a boundary-invertibility and
phase-variation bound, as developed in `fused_nlfeast`, or a caller-supplied
count certificate.

For “the lowest `p` states,” the contour is only a numerical chart.  The exact
projector is unchanged by moving its boundary anywhere inside the exterior
gap.  `OccupiedChartPolicy` therefore uses Hermitian inertia to bracket
`λ₁`, `λₚ`, and `λₚ₊₁`, resolves those brackets only relative to the observed
gap, and places a circular chart between them.  No converged eigenvalues or
root oracle enter the solver path.

The oracle-free policy repairs the previous `g=10`, four-state failure:

```text
outer response steps          9
contour factorizations      864
real inertia factorizations 175
final invariant residual   4.7×10⁻¹²
relative density error     8.3×10⁻¹⁴
```

The inertia cost is the robust baseline.  It is about one real shifted
factorization for every five complex contour factorizations in this control,
before accounting for the lower constant of a Hermitian factorization.  A
production sparse solver can reduce it further by retaining the previous
chart, verifying its count with the already-computed node factors, and only
rebuilding through inertia after a count or boundary-margin warning.

Other regimes need different policies:

- A fixed interior interval keeps the contour fixed and rejects count changes.
- A large occupied space can track a chemical-potential interval instead of
  individual eigenvalue brackets.
- Metallic or gap-closing problems require finite-temperature occupations;
  no sharp moving contour can make the zero-temperature projector continuous.
- Non-Hermitian problems replace inertia by argument-principle or determinant
  counts and need an independent phase-resolution certificate.

