# 005: Realization Fixed Point

The June 2026 NLFEAST convergence preprint supplies a useful necessary
condition: a finite-quadrature filter must retain the exact target state as a
fixed point. Its theorem is not directly applicable here because it assumes
simple, linearly independent eigenvectors and therefore m≤n. Higher moments are
needed precisely when this physical-subspace model is too small.

The corrected modal column already has the right integrand:

```text
q(z;β,x) = [I − T(z)⁻¹T(β)]x / (z−β).
```

The coordinate-free version for an approximate invariant pair `(X,S)` is

```text
Q(z;X,S) = [X − T(z)⁻¹T(X,S)](zI−S)⁻¹.
```

For `T(z)=zI−A`, `T(X,S)=XS−AX`, and the expression reduces identically to
`(zI−A)⁻¹X`. This is the full-state FEAST reduction, not only a diagonal modal
limit.

There is also an exact discrete fixed point special to the circular midpoint
trapezoid rule. Let `A=(S−cI)/r`, use N nodes, and take 0≤k<N. At an exact
invariant pair,

```text
Σq wq μqᵏ X(zqI−S)⁻¹ = X Aᵏ(I+Aᴺ)⁻¹.
```

Thus every corrected moment has one common state `A`; quadrature changes only
the input map. A Hankel realization recovers the spectrum of `S` exactly at
the fixed point even for finite N, provided `I+Aᴺ` is nonsingular and the
realization is minimal. This is stronger than a generic quadrature-error
argument and explains why naive iteration of uncorrected Beyn moments and the
corrected moment iteration behave differently.

The two-sided divided overlap also exists before modal diagonalization. For a
polynomial `T(z)=Σₖ Aₖzᵏ`, define

```text
G = Σₖ≥1 Σⱼ₌₀ᵏ⁻¹ Rʲ(YᴴAₖX)Sᵏ⁻¹⁻ʲ.
```

With right and left invariant-pair residuals `Eᴿ=T(X,S)` and `Eᴸ`, it obeys

```text
RG−GS = EᴸᴴX−YᴴEᴿ,
C = GS−YᴴEᴿ = RG−EᴸᴴX.
```

Balancing `(C,G)` therefore produces one common state without matching or
diagonalizing first. The current modal Loewner construction is exactly its
diagonal specialization. For a structured analytic operator, each scalar term
uses the bivariate matrix divided difference, equivalently the upper-right
block of the function evaluated on `[R M; 0 S]`.

This exposes a real boundary in the present solver. For `T(z)=z²`, the minimal
pair `X=[1 0]`, `S=[0 1; 0 0]` retains algebraic multiplicity two without asking
for a Jordan decomposition. Replacing it by two scalar modes at zero makes the
corrected columns identical and drops the Hankel rank to one. The robust outer
state should therefore remain a small Schur/invariant-pair realization; modal
eigenpairs should be output data, not the state used by the next filter.

The state-level formulas now drive `structured_fused_state_nlfeast`. Generic
holomorphic problems use Cauchy integrals for `T(X,S)` and the divided overlap
from right/left operator actions alone. Common state is the accepted candidate
when its lift-normalized backward error is competitive; independent states are
the fallback.

The corresponding convergence object is the lifted observability space

```text
O_d(X,S) = [X; XS; …; XSᵈ⁻¹].
```

Minimality says this stack has rank m even when `rank(X)<m`. At the exact
finite-quadrature fixed point the corrected moments replace `O_d(X,S)` only by
a nonsingular right factor, so its range is unchanged. This suggests a direct
reinterpretation of the 2026 proof on a Grassmannian in dimension `n·d`: replace
the physical target eigenspace by `range(O_d)`, and replace its Rayleigh–Ritz
assumption by a conditioned Ho–Kalman extraction assumption modulo similarity.
The shared-left and shared-right quadratic controls now measure contraction of
these lifted spaces while the corresponding physical eigenspace has deficient
dimension.

This does not yet prove contraction. The missing estimate must show that an
approximate invariant-pair residual produces an out-of-realization component of
order `quadrature_error × realization_error`, followed by a stable Hankel
re-extraction. The Hankel singular gap, `I+Aᴺ` conditioning, and common-overlap
conditioning necessarily enter the constant.

Tangential compression also has a sharper state-level condition. For a right
direction Ω, the exact criterion is controllability of
`[Ω,SΩ,…,Sᵈ⁻¹Ω]`; the adjoint state gives the left criterion. A Jordan block of
length two is controllable with one direction at depth two, whereas two
duplicate diagonal modal columns require two directions. The prototype now
checks the actual controllability rank rather than inferring it from clustered
eigenvalues. This retains the `d·p≥m` capacity bound but replaces the overly
conservative `p≥cluster multiplicity` rule.

On `T(z)=z²`, width-one state compression preserves the two-state invariant
pair to the residual tolerance, while the two reported scalar eigenvalues split
at the square root of that error. This is the expected conditioning of a
defective eigenvalue. Acceptance for such cases must be based on the invariant
pair and lifted-space residual, not on treating duplicate scalar Ritz values as
independently well-conditioned outputs.

The common state is reduced to complex Schur form once per outer iteration.
Right and left maps use the same unitary gauge, so the balanced overlap remains
the identity. Every contour denominator is then triangular: the per-node state
work is `O(m²p)` for tangent width p, rather than an `O(m³)` shifted
factorization. This is essential for making a persistent state competitive with
the diagonal modal representation.

Reference: <https://arxiv.org/abs/2606.13357>
