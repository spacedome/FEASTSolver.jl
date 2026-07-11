# Local Derivative Bounds

Let `ρ⋆` be a fixed point, let `X` contain the `p` occupied eigenvectors of
`H(ρ⋆)`, and let `Z` contain the virtual eigenvectors.  Write their eigenvalues
as `λᵢ` and `λₐ`, with occupied–virtual gap

```text
δ = min₍ᵢ,ₐ₎ (λₐ−λᵢ) = λₚ₊₁−λₚ > 0.
```

For the real contact model `H(ρ)=H₀+g diag(ρ)`, define
`uᵢₐ = xᵢ ⊙ zₐ`.  The exact density-map derivative is

```text
DΦ(ρ⋆) = −(2g/h) ∑ᵢ₌₁ᵖ ∑ₐ₌ₚ₊₁ⁿ uᵢₐuᵢₐᵀ/(λₐ−λᵢ).
```

Consequently `DΦ` is symmetric negative semidefinite.  This gives three
immediate local results.

1. Plain SCF is locally contractive exactly when `β = ‖DΦ‖₂ < 1`.
2. The response equation `I−DΦ` is positive definite for every `g ≥ 0`, with
   eigenvalues in `[1,1+β]`.  Loss of plain-SCF contraction therefore does not
   imply loss of the Newton step.
3. For mixed SCF, the local factor is bounded by
   `max(|1−α|, |1−α(1+β)|)`.  The best scalar mixing predicted by the local
   model is `α = 2/(β+2)`, with factor `β/(β+2)`.

The simplest analytic upper bound retains the occupied geometry.  With
`P=XXᵀ` and

```text
C = ∑ᵢ,ₐ uᵢₐuᵢₐᵀ = P ⊙ (I−P),
```

we have

```text
β ≤ (2g/(hδ)) ‖C‖₂.
```

This is already substantially better structured than a bound using only
`‖DH‖/δ`, but it can still be pessimistic.  A higher-gap hierarchy follows by
retaining selected low-gap transitions exactly and bounding the remaining
terms with the smallest omitted gap.  This is the natural finite-dimensional
counterpart of the higher-gap SCF bounds in the literature.

There is also a basis-free contour bound.  If a contour of length `LΓ` remains
at distance at least `η` from the spectrum, then

```text
‖DΠΓ(H)[E]‖ ≤ LΓ ‖E‖/(2πη²).
```

For the contact density map, using Euclidean density norm and Frobenius matrix
norm gives the coarse bound

```text
‖DΦ‖ ≤ LΓ g/(2πhη²).
```

Because the contact Hamiltonian is affine in `ρ`, its second derivative
vanishes and differentiating the resolvent once more gives

```text
‖D²Φ(ρ)[v,w]‖ ≤ LΓ g² ‖v‖ ‖w‖/(πhη³).
```

This supplies a local Lipschitz constant for `DΦ` and hence a standard
quadratic Newton estimate.  Moreover, Weyl's inequality shows that a solution
gap `δ⋆` persists whenever

```text
g ‖ρ−ρ⋆‖∞ < δ⋆/2.
```

These estimates concern the exact Riesz projector.  Contour quadrature and
moment-realization errors should be added as separate perturbations; folding
them into the derivative obscures the algorithmic statement.

For a general parameter-dependent nonlinear operator, the individual moment
derivative is also direct.  If

```text
Aₖ(θ) = ∮Γ μ(z)ᵏ T(θ,z)⁻¹V dz/(2πi),
```

then

```text
DAₖ(θ)[δθ]
  = −∮Γ μ(z)ᵏ T(θ,z)⁻¹ DθT(θ,z)[δθ] T(θ,z)⁻¹V dz/(2πi).
```

Thus a first bound uses the contour length, `supΓ |μ|ᵏ`, two resolvent
factors, and a bound on `DθT`.  In exact arithmetic a minimal moment
realization produces the same invariant projector, so its local convergence
should be analyzed through the coordinate-free projector derivative above,
not by differentiating a particular Hankel SVD.  A separate numerical
stability result must then propagate moment errors through the realization;
that result necessarily depends on the smallest retained Hankel singular
value and the separation at the truncation boundary.

The exact Jacobian and its contour action are already implemented in
`response.jl`.  Small experiments can form it explicitly.  Large experiments
can estimate its extreme eigenvalue matrix-free because the contact response
is self-adjoint; rigorous computer-assisted upper bounds would additionally
require interval residual bounds, which this experiment does not yet provide.
