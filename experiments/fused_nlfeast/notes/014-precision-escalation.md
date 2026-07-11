# 014: Precision Escalation

Analytic residue moments for scalar `sin(z)` remove quadrature and solve error
from the high-count experiment. The infinity-norm condition of the monomial
Hankel is about `10²⁰` for 31 roots, `10⁵⁶` for 63 roots, and `10¹⁷¹` for 127
roots. The Prony recurrence is recovered at 128, 256, and 512 or more bits,
respectively. At insufficient precision the linear recurrence residual can be
tiny while its coefficients are completely wrong.

High precision is therefore a mathematically valid single-chart alternative,
but not a practical default for large FEAST problems. Promoting only the small
Hankel algebra cannot recover digits lost in double-precision contour solves or
moment summation; promoting sparse factorizations changes the dominant cost and
backend requirements. Decision: keep arithmetic genericity as a production
goal, use partitioning to bound local state count, and treat precision
escalation as an optional leaf policy rather than a fallback that certifies an
underresolved chart.
