# 008: Adversarial Numerics

The persistent Schur state is necessary at multiplicity. For
`T(z)=z²`, the modal iterator reports two values about `10⁻⁸` from zero with
`10⁻¹⁶` residuals and declares convergence. The invariant-pair state retains a
rank-two nilpotent realization to roundoff.

Equivalent analytic representations are not numerically equivalent. For
`T(z)=e^{αz}sin(z)` on the same three-root chart, the third Hankel singular
ratio falls from `1.7×10⁻¹` at `α=0` to `1.0×10⁻¹⁴` at `α=5`; rank drops to
two. Rectangular subdivision recovers all three roots to machine precision.

Cross-projected right and left Hankels describe one transfer realization. A
mode weak in either probe is weak on both crossed rank tests. Streaming
one-sided TSQR distinguishes which probe side is deficient, enabling targeted
probe augmentation.

Relative inexact solves preserve the fixed point because solve error is
multiplied by the current pair residual. On the seven-root sine control the
observed contraction is approximately `0.66η`. There is no fixed accuracy
floor, but large `η` needs more outer iterations.

The residual norm must be representation aware. For
`T(z)=Σ fⱼ(z)Aⱼ`, the modal backward error uses
`‖T(λ)x‖/(Σ αⱼ|fⱼ(λ)|‖x‖)`. A contour-wide maximum operator norm can understate
root error by many orders of magnitude.
