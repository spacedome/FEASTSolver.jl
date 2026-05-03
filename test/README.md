# Tests

`runtests.jl` is the maintained automated `TestItemRunner.jl` entrypoint. Test
items are split by purpose:

- `fast/`: default `Pkg.test()` coverage. These are small, deterministic usage
  examples that should run quickly while covering solver variants and API modes.
- `torture/`: flagged numerical stress cases. These use MatrixDepot/NLEVP style
  problems that are larger, more non-normal, or more expensive. They are
  correctness tests, not performance benchmarks.

The test philosophy is narrow: given a matrix or matrix-valued function and a
contour, the solver should recover the eigenvalues inside that contour to an
appropriate tolerance. Small dense problems should use analytic eigenvalues or
dense QR/QZ references. Harder non-normal or experimental problems may use
residual convergence as the primary check when exact reference eigenvalues are
not reliable or available.

Older research scripts and experiments live in `experiments/legacy_tests/`. They contain useful candidate problems, but they should not be treated as automated tests until converted into deterministic `@testitem`s with clear target eigenvalues or residual tolerances.

Current fast coverage:

- Dense standard FEAST on diagonal, symmetric, normal, and non-normal problems.
- Sparse standard FEAST direct and BiCGSTAB solver policies.
- Generalized FEAST and dual generalized FEAST on small diagonal generalized problems.
- Circular and rectangular contour variants on a sparse Laplacian with exact eigenvalues.
- Nonlinear FEAST on a linear pencil with exact eigenvalues.
- Sparse nonlinear FEAST on a linear polynomial.
- Nonlinear FEAST on a small quadratic matrix polynomial with exact roots.
- Nonlinear FEAST on the butterfly matrix polynomial using companion linearization as the reference.
- Distributed FEAST/NLFEAST smoke coverage for dense, sparse, stored, no-store, and persistent-plan paths.
- Gallery operator materialization/action consistency.

Current torture coverage:

- Dual generalized FEAST on a small generated Grcar problem using dense eigenvalues as the reference and a looser residual tolerance.
- Standard FEAST on a larger generated Poisson problem using dense eigenvalues as the reference.
- Sparse nonlinear FEAST on the NLEVP gun cavity problem as an explicit slow residual-convergence test.

The automated tests intentionally do not import MatrixDepot. MatrixDepot's
package initialization verifies remote index files and can download them into a
fresh depot; the generated Grcar and Poisson tests mirror MatrixDepot's Higham
generators locally instead.

TestItems and TestItemRunner are kept as test-only dependencies in `Project.toml`
under `[extras]` and `[targets]`. Use `just test`, which runs Julia's package
test harness through `Pkg.test(test_args=...)`. Use `just test REGEX` to run only
test items whose names match `REGEX`. Slow tests are tagged `:slow`; include them
with `just test --slow REGEX` or `just test-slow REGEX`. Torture tests are tagged
`:torture`; use `just test-torture` to run only that flagged stress suite.
Use `just test --help` or `just test --list-presets` to see the maintained
test loops without running them.
Use `just test --preset moment-core` for the focused higher-moment
algorithm-family checks; this excludes `:moment_heavy` and `:distributed`, and
is the shorter loop for the linear FEAST/SS, polynomial companion, dual reduced
extraction, no-oracle Beyn/SS extractor agreement, canonical NLFEAST, and
residual-Laurent-update reductions.

The maintained test options are:

- `--slow`: include tests tagged `:slow`.
- `--torture`: include tests tagged `:torture`.
- `--only-torture`: run only tests tagged `:torture`; this also enables slow tests.
- `--tags TAGS`: require comma-separated tags, e.g. `just test --tags moment_heavy "moment RII"`.
- `--exclude TAGS`: exclude comma-separated tags, e.g. `just test --exclude moment_heavy "moment RII"`.
- `--preset NAME`: apply a maintained test loop. Current presets are
  `moment-core`, `moment-heavy`, `moment-count`, and `torture`.
- `--list-presets`: print the maintained preset names and exit.
- `--help`: print the test-harness help and exit.
