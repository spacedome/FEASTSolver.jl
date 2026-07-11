# 011: Convergence Reframing

The 2026 NLFEAST proof establishes a rate of the form `c γₙ` for the corrected
contour filter, with exponentially decreasing quadrature error `γₙ`. Its
Assumption 3 requires simple eigenvalues with linearly independent eigenvectors,
so `m≤n`. Assumption 4 then assumes extraction returns eigenpairs whose errors
are proportional to the angle of the trial subspace. The paper explicitly
points to invariant pairs as the way to remove Assumption 3.

The reusable part is the filter estimate, not the nonlinear Rayleigh-Ritz
extraction. Higher moments must be formed after the corrected state filter;
applying residual inverse iteration independently to moment blocks has the
wrong fixed-point structure.

The proof target should be the quotient of minimal invariant pairs `(X,S)` by
similarity. A concrete local metric is obtained by orthonormalizing the lifted
map `[X; XS; …; XSᵈ⁻¹]`. The numerical gauge-invariance regression now checks
the corresponding backward error directly.

The proof can be split into independent library boundaries:

1. Keldysh/Cauchy representation and the corrected state-filter identity.
2. Quadrature error and inexact-solve perturbation bounds.
3. Local Ho-Kalman stability under a nonzero Hankel singular gap.
4. Right-leading and left-trailing Schur restriction under spectral separation.
5. Common-gauge stability under a nonzero divided-overlap singular gap.
6. Composition of these maps into a contraction modulo similarity.

The resulting rate will have the preprint's filter factor multiplied by
Ho-Kalman, Schur-separation, minimality, and overlap condition numbers. These
are also exactly the quantities exposed by the numerical acceptance tests.

LEAN_FEAST should initially take the analytic Fredholm/Keldysh and matrix
functional-calculus statements as library theorems with explicit hypotheses.
The project-specific finite-dimensional algebra can then be proved without
reconstructing all of complex analysis inside one chain. Deeper mathlib work
can proceed behind those interfaces independently.

Primary references:

- Kressner, Liu, Roman, Shao, and Shao, 2026: <https://arxiv.org/abs/2606.13357>
- Kressner, invariant-pair block Newton method: <https://doi.org/10.1007/s00211-009-0259-x>
- Brennan, Embree, and Gugercin, systems realization view: <https://doi.org/10.1137/20M1389303>
