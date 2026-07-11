# Simultaneous Spectral And State Nonlinearity

The master system is exercised by

```text
T(ρ,z) = −H(ρ) + zI + z²C,
ρ = diag(XXᴴ)/h,
Tρ(X,S)=0.
```

Here `X` is an orthonormal representative of the realized right invariant
subspace and `S` is its small matrix state.  One outer step forms the corrected
response

```text
[X−T(ρ,z)⁻¹Tρ(X,S)](zI−S)⁻¹,
```

realizes its moments, changes to an orthonormal gauge, and closes the density.
This is a direct block iteration on the coupled system.  It does not solve a
projected nonlinear eigenproblem inside every density step.

The reductions are numerical identities in the experiment:

```text
C=0                 linear-in-z projector moment FEAST
H independent of ρ  fused moment NLFEAST for a quadratic NEP
moment depth 1      NLFEAST-Beyn realization
moment depth >1     narrow SS/Hankel realization
```

On the `n=48`, `p=3`, quadratic-strength `0.02` control:

```text
mixed closure, α=0.4       38 outer updates
mixed closure, α=0.6       22 outer updates
Anderson closure, α=0.4    17 outer updates
Anderson closure, α=0.6    13 outer updates
final invariant residual   1.9×10⁻¹¹
```

At `C=0`, the quadratic implementation and the dedicated linear projector step
agree to a subspace gap of `9.6×10⁻¹⁶`.  This is stronger evidence for the
master form than merely placing the two algorithms in the same taxonomy.

The response-Newton extension is less automatic than in the linear-in-`z`
case.  Individual moment derivatives still have the double-resolvent form in
`LOCAL_DERIVATIVE.md`, but the selected nonlinear invariant subspace also
depends on left residues, state separation, and realization conditioning.
There are two defensible routes:

1. Differentiate the coupled invariant-pair and closure equations, eliminate
   the pair response, and solve the closure Schur complement.
2. Differentiate the corrected moment realization and include its Hankel
   singular-gap constants explicitly.

The first route is the cleaner analytic definition.  The second is needed only
for a finite-quadrature implementation bound.  Unlike the repulsive Hermitian
linear pencil, the resulting closure Jacobian need not be negative
semidefinite, so GMRES and globalization replace the unconditional CG result.

The current experiment establishes the direct and Anderson forms.  It does not
claim a general contraction theorem for simultaneous nonlinearities.

