# 003: Pole-Part Loewner Hermite Boundary

Question: can Q₀/P₀ form the exact pole-part Loewner pencil and replace the
modal divided-difference coupling?

Result: algebraically yes, but it is an alternate realization rather than an
accelerating update. The retained Ho-Kalman input reproduces Q₀/P₀ to 1e-11
initially and about 1e-15 after filtering. Scalar and resolved 2-by-2 controls
recover the exact common pencil to roundoff. On the eight-node canonical case
it reproduces independent convergence exactly. Near converged poles,
κ(μI-Â) reaches 1e14--1e17; sine becomes slower, nonnormal residuals degrade,
and quadratic rank pruning is inconsistent. A residual gate rejects every
materially inferior candidate and gives no iteration-count improvement.

Decision: do not promote this construction. Possible continuations are
locking/deflation, disjoint left and right interpolation points, or a coupled
derivative recovery that never solves at the converged Ritz value.
