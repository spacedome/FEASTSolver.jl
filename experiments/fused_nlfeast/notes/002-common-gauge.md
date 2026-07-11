# 002: Divided-Difference Common Gauge

Question: what is the nonlinear analogue of dual-FEAST
biorthogonalization, and can it couple higher-moment right and left filters
without collapsing the state to YᴴX?

Result: use the divided-difference overlap G and common action C. The balanced
construction reduces exactly to dual FEAST for zB-A, to the two-sided Newton
step for one mode, and remains full rank for distinct scalar roots when m>n.
The later invariant-pair formulation replaces scalar Loewner modes with a
common Schur state, retaining repeated and defective states.

The overlap rank decision must follow lift normalization of the independent
right and left pairs. Otherwise separate similarity scalings can make an exact
overlap arbitrarily ill-conditioned. On the gun problem normalization improves
the overlap ratio from about `3.6×10⁻³` to `3.2×10⁻¹` and improves the modal
backward error.

Decision: the divided-difference common Schur state is the primary candidate,
gated by normalized overlap and two-sided backward error. Independent states
are the fallback. Call G a divided-difference overlap or biorthogonality matrix;
do not use packet terminology. Modal Loewner coupling and the direct cross
Hankel remain diagnostics.

An equal-solve A/B comparison on the loaded-string problem now isolates the
iteration effect. Across ten fixed probes at 16, 24, and 32 contour nodes, the
common iteration converges in 30/30 runs, with median update counts 7, 4, and
3. The independent right/left iteration converges in 27/30 runs, with medians
8, 6, and 3. Its worst eigenvalue errors remain near `10⁻⁸`, compared with
`10⁻¹³` for the common iteration. This is evidence of improved convergence on
this problem, not a universal contraction theorem. On the analytic nonnormal
control the common candidate gives a smaller residual iteration count but not
a smaller forward eigenvalue error; the acceptance guard correctly retains
the independent candidate on many updates.
