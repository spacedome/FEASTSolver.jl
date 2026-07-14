# Two-Timescale Realizations

The projected eigenvector-nonlinear methods have two independent choices.

```text
inner schedule     one closure step | fixed small count | converged | leakage-forced
FEAST repair       occupied filter  | full-q filter     | response action
space memory       p only           | fixed q           | window of prior spaces
```

These choices must be compared separately. The existing `reduced.jl` control
uses a fixed `q`, converges a restarted reduced solve to a fixed tolerance, and
then filters all `q` vectors inside a contour containing `q` states.

The 2013 eigenvector-nonlinearity NLFEAST algorithm is different. It appends
recent contour subspaces, solves the nonlinear problem in their accumulated
span, and uses only a small fixed number of inner DIIS steps. The paper reports
three inner steps and recommends retaining several recent subspaces.
`raw_windowed.jl` implements that pre-extraction update with `d` moment blocks.
`windowed.jl` is its exact-rank post-extraction counterpart. The two must be
compared directly; `reduced.jl` is not a surrogate for either.

This memory result is not new to the present experiment. Figure 5 of the 2013
paper reports that one retained contour space can fail, three or four retain
high convergence, and unbounded accumulation is fastest. It also caps the
reduced DIIS solve at three steps. The controls here reproduce those qualitative
results while adding higher-moment width, residual, and operation-count data.

`two_timescale.jl` implements the fixed-memory occupied-enrichment candidate.
For an orthonormal `U ∈ ℂⁿˣq`, it solves the reduced closure only until

```text
‖D(UY)−ρ‖ / ‖ρ‖ ≤ η ‖H(ρ)UY−UYΛ‖ / ‖H(ρ)UY‖.
```

The right side is the error that no solve confined to `U` can remove. FEAST
then filters only the `p` occupied Ritz vectors, projects their new component
outside `U`, appends that correction, and Ritz-truncates back to `q`. Thus each
refresh uses `N` factorizations and `Nℓ` RHS columns with `dℓ≥p`, independent
of `q`. At `q=p`, the repair is exactly the direct moment-projector step.

`η=1` is the current balanced default, not a theorem. `forcing_sweep.jl` shows
that it minimizes or ties refresh count on the tested multi-state controls.
The difficult `g=10,p=1,q=3` case instead needs `η=4`, while either increasing
to `q=5` or enabling adaptive growth restores convergence at `η=1`.

Current controls show:

```text
g=1, p=3, q=5:  8 refreshes for d×ℓ = 1×3, 2×2, and 3×1
                 RHS columns 1536, 1024, and 512 at N=64
g=5, p=3:       q=5, 6, 7, 8 converge in 18, 13, 12, 12 refreshes
                 q=3 is plain projector SCF and diverges
                 q=4 also lacks enough response space
adaptive q:     q=3 grows once to q=6 and needs 15 refreshes
                 known q=6 needs 13 refreshes
windowed g=5:   1 block diverges; 2, 3, and 4 blocks converge
                 3 fixed inner steps give 28, 12, and 11 refreshes
                 with 2 blocks, 2 inner steps improve 28 to 15
growing g=5:    5 inner steps give 8 refreshes and dimension 24
                 d×ℓ = 1×3, 2×2, 3×1 preserve the 8 refreshes
                 while RHS columns fall from 1536 to 1024 to 512
rank-one g=10:  fixed q=5 converges at η=1; q=3 needs η=4
                 adaptive q=1→5 converges in 16 refreshes
```

Both fixed-memory and windowed forms pass convergence and cost-accounting
tests. The raw closure/leakage norm ratio is a useful diagnostic but not yet a
final forcing law: the two residuals require scaling, and the best inner depth
depends on how much subspace history is retained. Growing memory, adaptive
window size, and a trust-region interpretation of Hamiltonian drift remain
open algorithm choices.

The physical nonlinear state is always warm-started across a contour refresh,
and the windowed forms also retain prior contour blocks. Raw Anderson secants
are different: they approximate a reduced closure map that changes when the
space changes. `history_reuse_sweep.jl` compares restarting them with retaining
them in physical density coordinates. Retention increases refresh counts from
7 to 8, 12 to 18, and 15 to 31 on the `g=1,p=3`, `g=5,p=3`, and `g=10,p=1`
controls, while increasing inner work much more sharply. Cross-refresh secants
therefore remain a rejected policy unless a future transport or acceptance
test accounts for the map change.

The pre-extraction and post-extraction windows currently agree to roundoff for
depth-one wide FEAST and within about `10⁻¹⁰` for the `d=3,ℓ=1` higher-moment form. Pre-extraction
uses only `d` moments and no Hankel realization before the nonlinear reduced
solve. It is therefore the more faithful and cheaper 2013 control; the
post-extraction form remains useful because it isolates the exact occupied
range from finite-quadrature leakage.

Pre-extraction oversampling also exposes a finite-quadrature performance axis.
At `N=32`, widening a three-block depth-one probe from `ℓ=3` to `ℓ=5` reduces
12 refreshes to 9 but increases RHS from 1152 to 1440. The narrow `d=3,ℓ=1`
form uses only 384 RHS at the same 12 refreshes. Thus oversampling is useful
only when a saved factorization is worth hundreds of RHS solves; its extra
directions vanish in the exact-projector limit.

The reduced solve need not use Anderson. `windowed.jl` also implements the
exact occupied–virtual derivative inside the current basis and Newton–CG on the
reduced closure. Its derivative agrees with finite differences to `6×10⁻¹⁰`.
On sparse `n=256,p=4,g=5`, one inexact Newton step per refresh matches growing
Anderson's 8 factor sets while replacing 37 inner Hamiltonian builds by 8 builds
and 24 derivative actions. This is a cost-dependent inner variant, not the
default for the cheap contact model.

Primary references are the original eigenvector-nonlinearity NLFEAST paper
(<https://arxiv.org/abs/1211.4261>) and the linear FEAST subspace-iteration
analysis (<https://doi.org/10.1137/13090866X>).
