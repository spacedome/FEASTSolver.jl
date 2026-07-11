# 013: Meromorphic Boundary

Contour extraction samples `T(z)⁻¹`, so poles of `T` inside the contour do not
automatically invalidate it. If `T=G/q` with `G` holomorphic, the moments are
those of `G⁻¹` with filters multiplied by `q`. This explains why the loaded
string and the scalar `sin(z)/(z−p)` control work with an interior pole.

Counting needs more information. The argument index is the sum of positive
partial multiplicities minus the sum of negative ones. The implemented
meromorphic count adds a caller-certified pole multiplicity to that index. For
`sin(z)/(z−1.3)`, index two plus one pole gives the three eigenvalues
`−π,0,π`; the common-state iteration reaches `1.8×10⁻¹⁶` residual. For a
two-by-two coincident zero/pole example, the index is zero while the resolvent
moment rank and corrected count are one.

The coincident example also marks the unsupported boundary. A termwise
meromorphic invariant-pair action attempts to evaluate a singular scalar
function at the state even when its coefficient annihilates the target root
direction. Supporting this requires a holomorphic clearing without accepted
spurious states, a local pole-cancelled action, or a strong rational
linearization. The generic Cauchy residual/overlap formulas remain restricted
to holomorphic charts because interior operator poles add residues.

References: <https://m.mathnet.ru/php/archive.phtml?jrnid=sm&option_lang=eng&paperid=3169&wshow=paper>,
<https://eprints.maths.manchester.ac.uk/2538/1/main.pdf>.
