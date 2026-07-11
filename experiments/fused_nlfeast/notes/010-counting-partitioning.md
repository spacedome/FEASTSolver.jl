# 010: Counting and Partitioning

Residual-small roots do not certify completeness. Direct argument-principle
counting is exact in theory but may require `n` derivative right-hand sides per
node. Determinant-phase winding is preferable when direct factors expose a
stable determinant. Stochastic trace counts are estimates only: on an oblique
projector with Frobenius norm about 45, 256 probes rounded correctly in only 9
of 200 trials.

Determinant phase does not certify its own sampling resolution. For example,
`z¹⁵⁰` aliases to winding 22 on both 64 and 128 samples. The implementation
therefore reports it as uncertified unless phase resolution is supplied by the
caller or a derivative bound.

The same limitation applies to derivative argument-principle quadrature.
Symmetric unresolved poles just inside and outside a contour can give the same
integer at two refinement levels. Both backends now report numerical stability
separately from independently certified resolution.

For determinant winding, a segment-wise bound on `‖T′‖` can construct the
missing certificate. Endpoint singular values plus the Lipschitz bound give a
continuous lower bound on `σmin(T)`; the resulting logarithmic-derivative bound
must permit less than `π` argument change per segment. This detects a zero on a
shared partition boundary, which count additivity alone cannot do.

For meromorphic functions, winding counts zeros minus poles. The scalar
`sin(z)/(z−p)` control has three eigenvalues and index two. A
two-by-two coincident zero/pole example has a rank-one contour realization and
zero winding. Pole metadata or a holomorphic clearing is mandatory.

Naive overlapping circular subdivision recovered all 31 sine roots but visited
2,413 charts, solved 326 leaves, and left 646 boundary charts unresolved.
Gauss-Legendre rectangular contours permit disjoint subdivision. Certified
child counts must add to the parent count, and shifted cuts avoid boundary
eigenvalues. Failed leaves first refine quadrature without changing their
certified count. The controlled sine partition now visits 17 charts, solves
eight leaves, performs two leaf refinements, rejects two cuts, and recovers all
roots to `3.4×10⁻¹³` with the exact-count control. The derivative-certified
determinant backend needs no root oracle, visits 15 charts, performs six count
refinements, rejects one cut, and recovers all roots to `1.4×10⁻¹³`.

The branch-domain probe separates validity from conditioning. A valid
principal-square-root disk counts and solves correctly. A branch point just
outside the disk yields accurate roots but an uncertified count until high
quadrature. A disk crossing the branch cut produces a noninteger index and
spurious states, so it must be rejected by the operator-domain contract.

Reference: <https://doi.org/10.1002/nla.70072>.
