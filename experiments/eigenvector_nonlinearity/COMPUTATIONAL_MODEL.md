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

The sparse implementation preserves sparse shifted matrices and reuses every
node factorization across the moment extraction and all response actions of an
outer step.  On the warmed one-dimensional `n=256` control:

```text
Anderson          14 outer steps   3.94 s
inexact response   7 outer steps   2.01 s
```

This is not a scaling claim, but it confirms the cost model even where sparse
factorizations have very little fill.  Symbolic reuse and persistent numeric
factor ownership remain backend optimizations; the experiment does not
duplicate them.

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

