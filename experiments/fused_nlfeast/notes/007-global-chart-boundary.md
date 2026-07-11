# 007: Global Chart Boundary

The single-chart scalar sine problem exposes a hard completeness boundary. With
four to eight contour nodes per expected root, the current monomial Hankel path
behaves as follows:

```text
7 roots:  7 returned, converged
15 roots: 13 returned, reported converged
31 roots: 17 returned, reported converged
```

The retained roots have tiny residuals. Residual convergence therefore says
nothing about completeness. At 31 roots the target Hankel singular ratio is
about `10⁻¹⁷`; the information needed to distinguish the full realization is
below double-precision rank resolution.

Moving the contour midway between adjacent roots improves argument-principle
quadrature but does not repair the realization. With 120 nodes for 15 roots,
the full Hankel rank can still be detected at a `10⁻¹²` threshold, yet two
realized values move outside the chart. With 248 nodes for 31 roots, only 19
states are numerically visible.

A direct cache-native Loewner extractor was tested as the rational-coordinate
alternative. It exactly recovers controlled rational realizations and a cached
nonnormal linear problem. On the large sine chart it is also ill-conditioned:
the fifteenth Loewner singular value is around `10⁻⁹`, and finite quadrature
errors produce missing or inaccurate roots until hundreds of nodes are used.
Even then the raw values need refinement. Loewner is an interchangeable
extractor and useful agreement diagnostic, not a universal cure for an
under-observed global chart.

This is consistent with realization theory: one physical input/output channel
must identify a high-order state from a long scalar sequence. The conditioning
can deteriorate super-polynomially with state order relative to channel width.
Higher moments remove the algebraic `m≤n` limitation; they do not remove this
finite-precision information limit.

The algorithm therefore needs all of the following:

```text
an independent reliable algebraic count,
Hankel/Loewner rank and conditioning diagnostics,
extractor or layout agreement when conditioning is marginal,
chart splitting when the local realization is too large or under-observed,
support merging across overlapping charts.
```

The experiment now distinguishes matched from certified counts, augments weak
initial probes, refines failed leaves, shifts internal cuts until certified
child counts are additive, and assembles accepted leaves as one block invariant
pair. These operations remain an orchestrator around rather than inside the
single-chart state driver. Local convergence can only be claimed after assuming
a well-conditioned minimal realization on each accepted leaf chart.

Eigenvalue distance to the boundary is insufficient for a nonnormal state.
The driver also records a continuous lower bound on the normalized singular
separation of `zI−S` along the contour, obtained from samples and the chart-node
covering radius.

References: <https://doi.org/10.1137/20M1389303> and
<https://doi.org/10.1002/nla.70072>.
