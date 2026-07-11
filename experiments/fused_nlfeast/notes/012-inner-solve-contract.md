# 012: Inner Solve Contract

The scalar multiplicative-error probe was too favorable. A new nonnormal
control gives every contour solve a residual of prescribed relative norm and
places it in the least singular direction of `T(z)`. At contour condition
`κΓ≈6.8×10³`, `η=10⁻⁴` converges while `η=10⁻²` stagnates. At
`κΓ≈9.4×10⁷`, `η=10⁻⁴` no longer reaches the outer tolerance. At
`κΓ≈9.4×10¹⁵`, dense direct solves themselves achieve only about `4×10⁻⁷`
linear residual; the invariant pair has small backward error while modal
values remain wrong by about `5×10⁻⁶`. The gauge-invariant modal condition
`‖x‖‖y‖/|yᴴT′(λ)x|` grows from about `33` to `5×10³` to `5×10⁷` across these
controls and accounts for the loss of forward accuracy.

Relative inner residual remains residual-proportional after the cached solve is
multiplied by the residual factor, so it does not introduce an intrinsic fixed
outer error floor. It is nevertheless insufficient as a standalone policy:
the contour resolvent and realization conditioning control its amplification.

The cache now accepts a solve result containing achieved residual, convergence
status, and iteration count. The state driver can require these reports and
reject a node and side that fails or exceeds a configured residual. Diagnostics
survive response eviction.

Remaining decision: rollback or stagnation must tighten and retry the same
outer step through a tolerance-aware solve plan. The result must attach
invariant-pair/eigenvalue conditioning separately from backward convergence. A
solver-provided forward-error estimate or defect correction is preferable on
severely nonnormal nodes; no universal fixed `η` is part of the mathematical
algorithm.
