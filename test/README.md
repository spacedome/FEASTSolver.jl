# Tests

`runtests.jl` is the maintained automated test entrypoint.

The current test philosophy is narrow: given a matrix or matrix-valued function and a contour, the solver should recover the eigenvalues inside that contour to an appropriate tolerance. Small dense problems should use analytic eigenvalues or dense QR/QZ references. Harder non-normal or experimental problems may use residual convergence as the primary check when exact reference eigenvalues are not reliable or available.

Older research scripts and experiments live in `experiments/legacy_tests/`. They contain useful candidate problems, but they should not be treated as automated tests until converted into deterministic `@testset`s with clear target eigenvalues or residual tolerances.

Current automated coverage:

- Dense standard FEAST on a diagonal problem with exact eigenvalues.
- Generalized FEAST and dual generalized FEAST on a diagonal generalized problem with exact eigenvalues.
- Circular and rectangular contour variants on a sparse Laplacian with exact eigenvalues.
- Nonlinear FEAST on a linear pencil with exact eigenvalues.
- Dual generalized FEAST on a small MatrixDepot Grcar problem using dense eigenvalues as the reference and a looser residual tolerance.
- Standard FEAST on a small MatrixDepot Poisson problem using dense eigenvalues as the reference.

MatrixDepot is kept as a test-only dependency in `Project.toml` under `[extras]` and `[targets]`. Use `just test`, which runs Julia's package test harness.
