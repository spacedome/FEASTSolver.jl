# Fused NLFEAST Experiment

This experiment studies a cache-native, two-sided, higher-moment NLFEAST
iteration. It does not import `moment_rii`. The mathematical update is now
coherent, but the production algorithm is not finalized.

## Core state

The persistent mathematical object is a right/left invariant-pair state, not
a list of scalar Ritz columns. For right pair `(X,S)` and residual
`Eᴿ=T(X,S)`, the corrected node response is

```text
Q(z;X,S) = [X − T(z)⁻¹Eᴿ](zI−S)⁻¹.
```

The left response is its adjoint-contour counterpart with `(Y,R,Eᴸ)`. Moment
blocks use an affine chart coordinate `μ=(z−c)/r`:

```text
Qₖ = Σq wq μqᵏ Q(zq;X,S) Ωᴿ,
Pₖ = Σq wq* (μq*)ᵏ P(zq;Y,R) Ωᴸ.
```

At an exact pair the residual term vanishes. On a circular midpoint rule with
`A=(S−cI)/r`, `k<N`, the discrete moments are exactly

```text
X Aᵏ(I+Aᴺ)⁻¹.
```

Finite quadrature therefore changes the realization input map, not its state.
For `T(z)=zI−A₀`, the corrected response reduces identically to
`(zI−A₀)⁻¹X`, which is the FEAST filter.

## Realization and coupling

Block Hankel/Ho-Kalman extraction realizes the right and left moment
sequences independently. A streaming TSQR implementation provides the
one-sided tall-Hankel realization without materializing an `nd×pd` matrix.
Projected Hankels remain the fast main path when their visibility is sound.

Initial extraction must be broad enough to expose exterior quadrature states.
An ordered Schur restriction is then applied to the target chart. If a
reliable algebraic count is available, exactly that many interior states are
retained; a deficient realization is rejected rather than padded.

Right and left restrictions are dual. A right invariant pair retains the
leading selected Schur block. A left invariant pair retains the trailing block
after unselected states are ordered first. A common state may be restricted on
both sides only after a Sylvester block diagonalization supplies dual gauges.

For polynomial `T(z)=Σₖ Aₖzᵏ`, the state divided overlap is

```text
G = Σₖ≥1 Σⱼ₌₀ᵏ⁻¹ Rʲ(YᴴAₖX)Sᵏ⁻¹⁻ʲ.
```

It obeys the algebraic identities

```text
RG−GS = EᴸᴴX−YᴴEᴿ,
C = GS−YᴴEᴿ = RG−EᴸᴴX.
```

For `T(z)=Σₗ fₗ(z)Aₗ`, the same construction uses the bivariate matrix divided
difference of each `fₗ`, computed by the upper-right block of
`fₗ([R M; 0 S])` or by a specialized Fréchet action.

The structured form is an optimization, not a restriction on the theory. For
any operator holomorphic inside a chart, right and left actions determine all
state data through one Cauchy formula:

```text
Eᴿ   = ∮ T(z)X(zI−S)⁻¹ dz/(2πi),
Eᴸᴴ = ∮ (zI−R)⁻¹YᴴT(z) dz/(2πi),
G    = ∮ (zI−R)⁻¹YᴴT(z)X(zI−S)⁻¹ dz/(2πi).
```

These identities agree with the polynomial formulas numerically to roundoff.
They require `T` to be holomorphic on the enclosed sheet; meromorphic poles or
branch cuts contribute extra residues and invalidate the invariant-pair data.
Black-box action quadrature is refined independently from the solve contour to
avoid same-grid fixed points. It remains uncertified unless the caller supplies
an independent resolution certificate; structured matrix-function actions are
treated as exact for the declared representation.

If `G=UΣVᴴ`, the balanced common realization is

```text
Xᵦ = XVΣ⁻¹ᐟ²,
Yᵦ = YUΣ⁻¹ᐟ²,
Sᵦ = Σ⁻¹ᐟ²UᴴCVΣ⁻¹ᐟ².
```

This common gauge is the exact unified realization when `G` is nonsingular.
Numerically it remains a candidate: poor overlap conditioning or worse
two-sided backward error rejects it in favor of independent states.
Before forming `G`, each independently extracted pair is normalized by the QR
gauge of its affine lifted map. This makes the overlap singular gap invariant
under the separate similarities of the two Ho-Kalman realizations.

In the continuous-moment limit, one cross-Hankel SVD also yields a common
state, right map, and dual left map. At finite quadrature the physical left
moments contain an additional filter factor. The direct one-pencil form is
therefore an optional normalized candidate, not the current primary update.

## Numerical representation

The evidence now favors one canonical persistent representation:

- The iterator retains a small Schur invariant pair for every spectral block.
- Well-separated blocks may be diagonalized as a local arithmetic
  optimization, but modal eigenpairs are output data rather than the state.
- Independent right/left states remain the fallback when the divided overlap
  cannot support a stable common gauge.
- Full tangent width is canonical. Compression is enabled only after observed
  contraction, requires controllability/observability, and widens on
  stagnation.

This distinction is necessary. Scalar modal iteration falsely converges at an
exact double root with split values near `10⁻⁸`, while the Schur state retains
algebraic rank two to roundoff. The previous apparent loaded-string advantage
for modal iteration was caused by selecting a leading rather than trailing
left Schur block. With the dual restriction fixed, the common-state path
converges both sides to roundoff on every seed with a complete initial rank.

## Computational model

Physical contour solves remain dominant for large sparse problems. Per outer
iteration the state path uses:

```text
right/left residual factorizations:  O(nm²),
triangular state shifts:             O(Nm²p),
small realization/coupling work:     O(m³),
live response storage:               O(Nn(p+r)).
```

Residual blocks are compressed before solves. Compressed correction is fused,
so no `n×m` intermediate is formed when tangent width is `p<m`. Old response
blocks can be evicted after moments are formed; a sparse `n=128` control drops
from 3.75 MiB to 0.20 MiB of live cache at 12 iterations with identical output.

Structured divided overlaps cost small matrix actions, not `m²` full
operator-valued divided differences. Factorization persistence remains the
solve plan's responsibility and is not duplicated here.

On the `n=9956` gun cavity problem, 16 nodes, moment depth three, and probe
width eight retain all 17 target states. Four common-state updates reduce the
invariant-pair residual from `2.0×10⁻⁵` to `2.1×10⁻¹¹`; the worst two-sided
modal backward error is `7.1×10⁻¹¹`. Lift-normalizing the extracted independent
pairs before common coupling improves the overlap ratio from about
`3.6×10⁻³` to `3.2×10⁻¹`. Cached sparse factorizations take 3.8 s and each
update averages 4.3 s on the current machine.

The integrated driver also converges in four updates, selects the common state
throughout, and reaches `1.5×10⁻¹¹` lift-normalized error. After removing
duplicate Hankel SVDs, its warmed 25.1 s total for the initial filter plus four
updates is consistent with the manual loop's 4.4 s update time. Julia
compilation dominates an unwarmed first extraction and must be excluded from
production timing comparisons.

## Completeness and charts

Residual convergence is not completeness. Supported count backends are:

- derivative argument principle using `tr(T⁻¹T′)` when quadrature resolution
  has an independent error certificate,
- determinant-phase winding when direct factors expose determinants and phase
  resolution is independently bounded,
- caller-supplied or probabilistic estimates explicitly marked uncertified.

Determinant winding can derive rather than merely assert its phase certificate
when the operator supplies a segment-wise bound on `‖T′(z)‖`. Endpoint
singular values and that Lipschitz bound prove boundary invertibility;
`n‖T′‖/σmin(T)` then bounds argument change by less than `π` between phase
samples. This rejects eigenvalues on internal partition cuts.

Integrality and agreement between two node counts establish only a stable
estimate. They do not certify resolution: symmetric unresolved poles just
inside and outside a contour can alias to the same integer at both levels.

A requested count and a matched count are recorded separately from count
certification. Residual convergence plus an uncertified count may stop an
iteration, but the result remains explicitly uncertified.
Residual-evaluation certification is independent again: a black-box Cauchy
residual cannot certify the result merely because its value is small.

Chart containment is also not enough for a nonnormal state. The driver records
a continuous-boundary lower bound for `σmin((zI−S)/r)` on both sides. It
subtracts the quadrature-node covering radius from the sampled singular-value
minimum using the 1-Lipschitz property. A leaf may require a positive lower
bound before it is accepted.

For meromorphic `T`, determinant winding is zeros minus poles. The count API
therefore adds a caller-certified sum of pole partial multiplicities to the
signed argument index. A scalar pole inside the chart but disjoint from the
eigenvalue state is handled by the common-state iterator to roundoff. If a pole
and eigenvalue coincide in different partial directions, resolvent moments can
still expose the eigenvalue but the naive invariant-pair action is singular;
that case requires holomorphic clearing, a pole-cancelled action, or rational
linearization. Branch functions require a declared single-valued analytic
sheet on the chart.

Large or ill-conditioned realizations must be partitioned. Overlapping circles
recover the spectrum but duplicate work badly. Rectangular Gauss-Legendre
contours form a disjoint recursive partition. Internal cuts are accepted only
when certified child counts add to the certified parent count; shifted cuts
avoid boundary eigenvalues. A failed leaf first doubles quadrature nodes while
preserving its certified count. The exact-count control uses 17 visited charts
and eight solved leaves, performs two leaf refinements, rejects two unsafe
cuts, and recovers every root to `3.4×10⁻¹³`. With derivative-certified
determinant winding, the same 31-root problem needs no root oracle: it visits 15
charts, performs six count refinements, rejects one unsafe cut, and recovers
every root to `1.4×10⁻¹³`.

This is a numerical necessity, not only load balancing. On one circular sine
chart, double-precision monomial Hankels saturate near rank 16; requested ranks
31 through 127 remain unrecoverable even when the known rank is forced and the
node count is raised to 1,984. A multipoint Loewner realization shows the same
effective limit. Large-chart recovery therefore requires partitioning, higher
precision, or a future non-power basis.

Exact-residue arbitrary-precision diagnostics quantify that alternative. The
31-root sine power Hankel has condition about `10²⁰`, the 63-root case about
`10⁵⁶`, and the 127-root case about `10¹⁷¹`. Roughly 128, 256, and 512 or more
bits respectively recover the controlled recurrences. This does not rescue a
large sparse double-precision solve path: reduced high precision cannot restore
digits already lost in contour solves and moment accumulation. Partitioning is
therefore canonical; precision escalation is an optional local policy.

## Current boundary

The following are established primitives:

- linear FEAST, `K=1` NLFEAST-Beyn, polynomial, SS/Hankel, and high-moment
  reductions;
- state fixed point, state divided overlap, common gauge, multiplicity, and
  shared-eigenvector controls;
- a common-Schur research driver with independent fallback, lift-normalized
  backward error, rollback, compression retry, and initial probe augmentation;
- structured and black-box Cauchy-action interfaces for holomorphic operators;
- entire delay, rational loaded-string, meromorphic, branch-domain, inexact
  solve, analytic-scaling, visibility, sparse Schrodinger, and full gun probes;
- cache eviction, streaming TSQR, determinant counting, and rectangular
  contours.

The following still prevent a finished algorithm:

- public-driver integration of the experiment's adaptive node refinement,
  disjoint partitioning, and block invariant-pair assembly;
- structured failure diagnostics for incomplete visibility and
  under-resolution; invalid analytic domains and solve failures are typed,
  while stagnation is currently only a termination symbol;
- main-repository integration of the sparse action/solve/state-function driver
  and persistent factorization ownership;
- arithmetic genericity beyond the currently hard-coded `ComplexF64` path;
- tolerance-aware solve plans and adaptive inner-solve retry/tightening after
  rejected outer steps;
- meromorphic pole metadata and analytic-sheet contracts, including an
  explicit policy for coincident positive and negative partial multiplicities;
- attachment of modal sensitivity and solve diagnostics to the primary
  invariant-pair result object;
- a local convergence proof for lifted invariant-pair spaces modulo
  similarity, including inexact solves and Ho-Kalman conditioning.

Run `just test-fused-nlfeast` inside the development shell.
