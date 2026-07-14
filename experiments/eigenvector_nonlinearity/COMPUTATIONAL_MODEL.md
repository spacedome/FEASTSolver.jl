# Computational Model

Let `N` be the contour-node count, `n` the physical dimension, `p` the target
count, `d` the moment depth, and `ℓ` the probe width, with `dℓ≥p`.

One projector evaluation costs

```text
N shifted factorizations,
Nℓ triangular-solve RHS columns,
O(Nnℓd) moment accumulation,
O((dℓ)³) reduced realization work.
```

At fixed capacity `dℓ=p`, moment storage and accumulation stay `O(np)` while
solve width falls from `p` to `p/d`.  Higher moments are therefore a real
performance parameter, not extra outer stages.  Their limitation is numerical:
deeper power-moment Hankels lose visibility and singular separation.  Local
affine coordinates, chart partitioning, and eventually non-power realizations
are the remedies; increasing depth indefinitely is not.

The outer methods add different costs:

```text
mixing/Anderson   no response solves; one new factor set per outer step
response Newton  kNp response RHS columns; usually fewer outer factor sets
reduced NLFEAST   a q-state realization plus repeated reduced operator builds
two-timescale     a p-state realization plus repeated q-dimensional reduced builds
windowed moments  a p-state realization plus reduced builds up to hp dimensions
level shift       no response RHS, but a rank-p nonlocal node update
```

On the dense `n=72`, `p=4`, `g=5` control, the measured count crossover is:

```text
factorization worth < 11.6 RHS solves     Anderson wins
11.6 … 126 RHS solves                     inexact response wins
factorization worth > 126 RHS solves      tightly solved response wins
```

The exact thresholds are problem and backend dependent.  They make a useful
runtime policy: benchmark one node factorization and representative block
solves, then choose the forcing strategy rather than hard-coding it.

The occupied-enrichment two-timescale form decouples contour width from reduced
space size: each refresh costs `Nℓ` RHS columns with `dℓ≥p`, while each cheap
inner step forms and diagonalizes a `q×q` projected Hamiltonian. Its forcing
rule prevents those inner steps from continuing below the full-space leakage
floor. A full-`q` filter instead requires `dℓ≥q` and a contour containing `q`
states, so it pays for virtual response both in shifted solves and realization.

Accumulated moment windows retain the same `Nℓ` contour cost but grow the
reduced dimension to at most `hp` for `h` stored occupied ranges. Growing memory
replaces `h` by the refresh count. Its dense reduced eigensolve and `O(nhp)`
storage can eventually dominate, so it is a factorization-minimizing endpoint,
not a production default for large `p`. Thick-restarted adaptive `q` is the
memory-bounded counterpart.

Pre-extraction oversampling retains nonzero exterior components of the rational
filter. These can reduce nonlinear refresh count, but disappear as quadrature
approaches the exact projector. On the strong `N=32,h=3` control, moving from
`d=3,ℓ=1` to depth-one `ℓ=5` saves 3 factorizations and adds 1056 RHS columns,
a crossover of 352 RHS solves per factorization. At `N=64` the crossover rises
above 1200. Narrow higher moments remain the default unless factorization is
overwhelmingly dominant.

The sparse implementation preserves sparse shifted matrices and reuses every
node factorization across the moment extraction and all response actions of an
outer step. With the same moving-chart policy on the warmed one-dimensional
`n=256,p=4,g=5` control, `memory_costs.jl` reports:

```text
Anderson          4.616 s   672 factors   1344 RHS
inexact response  2.341 s   336 factors   5088 RHS
growing Anderson  2.660 s   384 factors    768 RHS   q=32
growing response  2.592 s   384 factors    768 RHS   q=32
```

This is not a scaling claim, but it confirms the cost model even where sparse
factorizations have very little fill. Growing moments dominates Anderson in
both counted operations. Response wins this wall-time control; their count
crossover is 90 RHS-column solves per additional factorization. The current
inertia policy densifies, so these timings also charge a known nonproduction
count backend. Symbolic reuse and persistent numeric factor ownership remain
backend optimizations; the experiment does not duplicate them.

The two growing rows differ only in the reduced nonlinear solver. Five-step
Anderson uses 37 reduced operator builds. One inexact Newton–CG step uses 8
operator builds and 24 analytic reduced-response actions. They have similar
wall time for the cheap contact Hamiltonian; reduced response becomes plausible
when constructing `H(θ)` is substantially more expensive than applying `DH`.

For very large `p`, occupied-width response is no longer attractive.  The
density-only finite-temperature branch replaces orbital RHS with pole
factorizations plus selected inversion.  If eigenvectors are needed, a local
FEAST extraction can be run only around the states of interest.

For small `n` and many nonlinear eigenvalues, such as `sin(z)=0`, physical
factorizations are cheap and the reduced realization dominates.  Large power
Hankels are then the wrong global representation: disjoint chart partitioning
keeps each realization small and is part of the algorithm, as established in
`fused_nlfeast`.

Response compression below `p` RHS is not generally available.  For a local
direction, `DH[δρ]X` can have rank `p`; numerical compression is opportunistic.
Anderson is the meaningful derivative-free compressed-response alternative,
while selected inversion is the structural alternative when only density is
required.
