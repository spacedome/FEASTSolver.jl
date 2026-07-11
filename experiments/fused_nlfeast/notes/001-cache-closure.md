# 001: Cache Closure

Question: can residual responses stored at the original contour nodes feed a
two-sided linear re-extraction without a projected nonlinear solve?

Result: yes on the initial controls. With 32 nodes, the scalar `sin(z)` case
retains rank seven in dimension one and reduces the physical root residual
`abs(sin(lambda))` from `4.26` to below `1e-8` in six cache updates. The same implementation passes the
linear filter identity, canonical `K=1`, polynomial companion, and nonnormal
left/right checks.

Decision: keep rank-compressed discrete corrected moments as the reference
update. Treat inverse-Laurent reduction, defective state matching, and
meromorphic charts as separate later questions.

The underresolved noncommuting quadratic is the first matrix closure control:
its interior count changes from five to the correct four after residual-cache
augmentation, then agrees with the companion reference.
