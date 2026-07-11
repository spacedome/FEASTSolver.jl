# Direct Projector Form

For a unitarily invariant Hermitian NEPv, write

```text
H(P)X = XΛ,   P=XXᴴ.
```

If `Γ` encloses the occupied spectral block and excludes the rest, the problem
is equivalent to the projector fixed point

```text
P = ΠΓ(H(P)),
ΠΓ(H) = ∮Γ (zI−H)⁻¹ dz/(2πi).
```

For a narrow probe `V`, form moments

```text
Mₖ(P,V) = ∮Γ μ(z)ᵏ(zI−H(P))⁻¹V dz/(2πi).
```

Block Hankel realization recovers the occupied state when moment depth times
probe width reaches the occupied count and the corresponding realization is
minimal. The output range gives the next projector. At depth one with a full
width probe this is ordinary FEAST-SCF; higher moments trade probe width for
realization depth.

The invariant-pair RII form is exact. With

```text
Tₚ(z)=zI−H(P),   E=XS−H(P)X,
```

the corrected response satisfies

```text
[X−Tₚ(z)⁻¹E](zI−S)⁻¹ = Tₚ(z)⁻¹X.
```

Thus the direct nonlinear projector iteration is the eigenvector-nonlinear
linear-limit identity of moment NLFEAST, applied to the Hamiltonian generated
by the current projector.

For a density parameterization `H(ρ)`, the local derivative is

```text
DΦ(ρ)[δρ] = diag(∮Γ R(z) DH(ρ)[δρ] R(z) dz/(2πi)) / h.
```

For the contact model, `DH[δρ]=g diag(δρ)`. The experiment evaluates this
Jacobian both from the eigendecomposition perturbation formula and directly
from the same contour resolvents. This separates nonlinear SCF contraction
from contour quadrature and moment-realization error.

The full diagonal response does not require a dense inverse. If `X` spans the
occupied projector, `S=XᴴH(ρ)X`, and

```text
C = (I−XXᴴ) DH(ρ)[δρ] X,
```

then the horizontal orbital response is

```text
Y = ∮Γ (zI−H(ρ))⁻¹ C (zI−S)⁻¹ dz/(2πi),
DΦ(ρ)[δρ] = diag(YXᴴ+XYᴴ)/h.
```

This costs occupied-width solves at each contour node. The response-Newton
step solves

```text
(I−DΦ(ρ))δρ = Φ(ρ)−ρ
```

by a matrix-free Krylov method, reusing the factorizations and occupied
invariant pair from the moment projector step.  The repulsive contact model
uses CG; nonsymmetric response operators use GMRES.

For the contact model this derivative is symmetric negative semidefinite.
`I−DΦ` is therefore positive definite even when plain SCF is not contractive.
The exact finite-dimensional identity, gap bounds, contour bounds, and local
Newton consequences are collected in `LOCAL_DERIVATIVE.md`.

## Claim Boundary

The current construction applies directly to Hermitian, unitarily invariant
NEPv problems where the Hamiltonian depends on an occupied projector or a
density derived from it. It includes local-density Kohn–Sham/Hartree models and
the rank-one Gross–Pitaevskii equation.

Companion probes now cover an oblique-projector non-Hermitian form, moving
occupied charts, and simultaneous quadratic spectral/state nonlinearity.  The
derivative result in this file remains specific to the Hermitian linear pencil.
Truly orbital-specific operators, non-Hermitian response Newton, and general
simultaneous-nonlinearity response require their own tangent equations.  The
current response step uses occupied-width right-hand sides; compressing that
action is a separate implementation question.
