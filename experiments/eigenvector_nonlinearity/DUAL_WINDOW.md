# Dual Accumulated Moments

For non-Hermitian eigenvector nonlinearity, right and left spaces represent one
oblique projector. A valid accumulated method must therefore couple the two
histories rather than update them as independent physical states.

At each node, `dual_windowed.jl` uses one LU to form right moments and
adjoint-left moments. Recent blocks are accumulated separately, then an
overlap-SVD selects a common rank and establishes the biorthogonal gauge before
the reduced nonlinear solve. Depth one with a wide probe reduces numerically
to the existing dual-FEAST step.

On the similarity-transformed `n=48,p=3` control:

```text
direct mixed dual FEAST     33 refreshes   3168 factors   19008 RHS
three-block d=3,ℓ=1         7 refreshes     672 factors    1344 RHS
growing d=3,ℓ=1             6 refreshes     576 factors    1152 RHS
```

The three-block form also converges in 6 refreshes on `n=40` controls with
similarity condition numbers from about `3` through `1.7×10³`. Final
biorthogonality errors remain below `4×10⁻¹⁵`. The gain comes from both memory
and the higher-moment width reduction; the single node-factor cache is retained.
