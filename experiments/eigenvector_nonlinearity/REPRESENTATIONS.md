# State And Realization Choices

The physical nonlinear state and its numerical representation are separate
choices.  For the current local mean field, density is the minimal state and
occupied orbitals are only a low-rank way to obtain it.

```text
physical state       minimal representative       natural contour output
density ρ            n real values                diag(P)/h
orthogonal projector X with gauge quotient        P=XXᴴ
oblique projector    dual factors X,Y             P=X(YᴴX)⁻¹Yᴴ
orbital-specific     tuple of rank-one factors    (xᵢxᵢᴴ)ᵢ
finite-temperature   selected density-matrix data diag(fβ(H−μ))
```

## Density Versus Orbitals

For `H(ρ)`, iterating density is coordinate-free and does not retain unused
off-diagonal projector information.  Gauge-aligned orbital mixing is valid:
align the new basis by a polar factor, mix the bases, and retract to the
Stiefel manifold.  It is nevertheless a different map from density mixing.

On the `n=72`, `p=4`, `g=5` control:

```text
density mixing α=0.40       39 contour updates
orbital mixing α=0.40       50 contour updates
Anderson density iteration  18 contour updates
```

Orbital mixing becomes relevant when `H` needs the full projector or when the
problem is only locally unitarily invariant after basis alignment.  It is not
the preferred representation for a purely local-density closure.

## Level-Shifted Projector

Level-shifted SCF filters

```text
H(P)−σP.
```

At a solution its occupied eigenvalues shift by `−σ` while its invariant
subspace is unchanged.  It can therefore stabilize plain SCF without
derivatives or response solves.  It necessarily carries `P`, however, and the
rank-`p` term can spoil a sparse node matrix unless a low-rank factorization
update is used.

The experiment confirms both sides of the tradeoff.  For a weak `g=1` problem,
a tuned `σ=0.25` reduces 28 plain updates to 15.  For the difficult `g=5`
problem, the best dense-eigensolve calibration near `σ=1.4` takes 61 updates,
versus 18 for Anderson and 7–8 for response Newton.  Excessive shift drives the
contraction factor back toward one.  Level shifting is therefore a legitimate
derivative-free fallback, not a universal accelerator.

## Reduced Nonlinear Solve

`reduced.jl` is a fixed-`q` full-filter control: it filters a contour containing
`q>p` states, solves the density problem inside that space to a fixed tolerance,
and discards the reduced history on refresh. It is not a faithful implementation
of the 2013 accumulated-subspace eigenvector NLFEAST algorithm.

Observed behavior is regime dependent:

```text
g=1, p=3, q=5…8       9–11 outer filters
g=5, p=3, q=8         31 outer filters
g=5, p=3, q=10        24 outer filters
g=5, p=3, q=6         incomplete response/count failure
g=5, p=3, q=12        moment-rank conditioning failure
```

This control trades response RHS for a larger contour realization and a reduced
nonlinear solve.  It is attractive when `q` stays small and reduced operator
formation is cheap.  It is fragile when the virtual-response dimension is
unknown, which is precisely the case where direct response Newton avoids an
arbitrary buffer.

`two_timescale.jl` instead stops the reduced solve when its closure defect
reaches the full-space leakage floor. It filters only the occupied block and
uses its component outside the current basis to repair a fixed `q` space. Its
contour RHS width therefore depends on `p`, not `q`. The strong `g=5,p=3`
control converges for `q=8` and `q=10`, but not for `q=5`; a windowed or growing
space remains the natural way to remove that fixed-response-dimension choice.

The accumulated controls retain recent filtered occupied ranges instead of a
fixed virtual eigenspace. `raw_windowed.jl` appends the `d` moment blocks before
extraction, matching the 2013 algorithmic order; `windowed.jl` appends the
realized occupied range and isolates the exact-rank mechanism. On the strong
control they agree, and three or four stored ranges converge in 12 or 11
refreshes. Unbounded growth reaches 8 refreshes at reduced dimension 24. This
removes the guessed virtual contour but replaces it with an explicit memory
and reduced-cubic-cost policy.

## Non-Hermitian State

For non-Hermitian `H(P)`, right and left bases are not two physical states.
They represent the single oblique Riesz projector

```text
P=X(YᴴX)⁻¹Yᴴ.
```

The experiment applies a diagonal similarity transform to the Hermitian model.
This preserves its spectrum and exact density while making the right and left
spaces distinct. The original direct control converges in 42 mixed updates
across its condition-number sweep from `3` to `3×10³`. At condition number
`3×10²`, replacing the oblique density by the
Euclidean projector of right vectors alone gives a `34%` density error.  The
left realization is therefore essential, while the common projector remains
the natural nonlinear state.  Each node uses one factorization for both the
right solve and its adjoint left solve, so the dual form retains a single
factor cache.

For a general non-Hermitian closure the response Jacobian is not self-adjoint;
the response solve returns to GMRES.  Higher-moment narrow probes use the same
independent right/left realization and divided-overlap common gauge already
tested in `fused_nlfeast`.

`dual_windowed.jl` extends this to coupled accumulated spaces. Right and
adjoint-left moment blocks share each node LU, while an overlap-SVD supplies one
retained rank and common gauge before reduced extraction. On the separate
`n=48`, tighter-tolerance cost control, three-block and growing schedules reduce
33 direct refreshes to 7 and 6 without losing biorthogonality.

## Finite Temperature And Large Occupation

When a sharp occupied count is discontinuous or `p` is very large, replace the
Riesz projector by the Fermi density matrix

```text
Pβ,μ = fβ(H(ρ)−μI),
ρ = diag(Pβ,μ)/h,
tr(Pβ,μ)=p.
```

This is the natural density-only branch.  Pole expansion retains FEAST's
independent shifted factorizations, while selected inversion obtains only the
diagonal entries needed by the closure.  It avoids `p` right-hand sides and all
Hankel realization work.  The chemical potential must be solved together with
the density, preferably by maintaining rigorous upper and lower bounds rather
than fully converging it inside every SCF step.

The response is the Fréchet derivative of the Fermi matrix function and can be
written with the same double-resolvent contour formula.  At zero temperature,
provided a gap persists and `μ` lies inside it, this branch tends to the sharp
projector map.  It does not return occupied eigenvectors; a separate local
FEAST extraction is needed when they are requested.

## Orbital-Specific Boundary

A closure that changes under `X→XU` cannot be represented by one occupied
projector.  The natural state is a tuple of rank-one projectors, possibly with
an assignment/permutation convention.  The equations live on a product of
Grassmann manifolds and may have different effective Hamiltonians per orbital.

Some optimization-derived problems are locally unitarily invariantizable by
aligning the basis to a reference.  Gauge-aligned SCF then applies locally.
Truly orbital-specific systems instead require either separate contour filters
with a coupled orthogonality step or a block-diagonal lifted operator.  This is
a real extension, not a mode of the current density solver.

Primary analysis for these branches includes the level-shifted SCF convergence
work of Bai, Li, and Lu (<https://doi.org/10.1137/20M136606X>), locally
unitarily invariantizable NEPv analysis by Lu and Li
(<https://arxiv.org/abs/2212.14098>), and bounded chemical-potential iteration
for pole expansion and selected inversion by Jia and Lin
(<https://arxiv.org/abs/1708.04323>).
