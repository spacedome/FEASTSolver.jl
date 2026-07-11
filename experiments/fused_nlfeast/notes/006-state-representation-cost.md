# 006: State Representation Cost

The invariant-pair state is the correct mathematical object, but a dense Schur
state should not automatically be the numerical representation used at every
node. A diagonal modal denominator applies in `O(mp)` and preserves the sparse
tangential partition. A Schur denominator costs `O(m²p)`, and multiplying its
dense result by an n-by-m physical map costs `O(nmp)`.

A warmed synthetic shape diagnostic used 2d circular nodes, moment depth d,
and tangent width p with `d·p=m`. It excluded physical linear solves:

```text
n=1,    m=64,  d=8,  p=8: modal 0.00018 s, state 0.00066 s
n=1024, m=64,  d=8,  p=8: modal 0.0090 s,  state 0.036 s
n=1,    m=128, d=16, p=8: modal 0.00067 s, state 0.0043 s
n=1024, m=128, d=16, p=8: modal 0.051 s,   state 0.226 s
```

State allocations were about 5–9 times larger in this unoptimized experiment.
These are not package benchmarks, but the scaling difference is real. For
large sparse NEPs the contour solves should still dominate. For scalar or very
small physical operators with many roots, the reduced state work can dominate.

The intended representation is therefore one invariant pair exposed through
spectral blocks:

```text
well-separated, well-conditioned simple state  -> scalar modal blocks
clustered, repeated, or ill-conditioned state   -> small Schur blocks
```

The full state remains available for analysis, residuals, and common-gauge
construction. After coupling, a reordered Schur form and well-conditioned
Sylvester block separation can expose the numerical blocks. Node denominators
then mix columns only inside a block. This recovers the modal cost for ordinary
simple spectra while retaining the `z²` two-state realization and other unsafe
clusters without explicitly constructing Jordan chains.

The separation test cannot be removed: deciding that a Schur block may be
diagonalized is a conditioning decision. It should be reported through block
separation and eigenvector-condition diagnostics rather than a bare distance
threshold. Tangential widths should be certified by controllability within each
retained block.

Decision: retain the full local Schur invariant pair as the canonical
representation. The gun control is solve dominated at rank 17, while scalar
high-count charts must be partitioned for realization conditioning regardless
of arithmetic cost; partition capacity therefore also bounds dense state work.
Modal or separated Schur-block arithmetic remains a useful leaf optimization,
not an alternate algorithm or a production-readiness gate. Modal output still
requires the conditioning test because it cannot replace the state at
multiplicity.
