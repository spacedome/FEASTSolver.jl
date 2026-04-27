# TODO

- [ ] Make agent and developer iteration fast and reliable
  - Keep `nix develop` as the single default local development shell.
  - Maintain simple `just` commands for package loading, tests, docs, instantiate, update, and status.
  - Keep repo-specific agent notes in `just notes` so future sessions know the shell, test entrypoint, and current caveats.
  - Do not expand the dev shell into a Julia-version compatibility matrix. That belongs in CI.

- [ ] Revamp the tests, add more and better test problems
  - Use <https://github.com/JuliaLinearAlgebra/MatrixDepot.jl>
  - Sources the above pulls from:
    - <https://sparse.tamu.edu/VLSI/stokes>
    - <https://math.nist.gov/MatrixMarket/>
  - We're pulling nonlinear problems from julia NLEP?
  - In particular each variant of FEAST has different test problems suited for it. Some may have analytic/exact eigenvalues known in MatrixDepot.
  - For small dense problems, compare against dense QR/QZ solvers when analytic eigenvalues are not convenient.
  - Use residual-only checks deliberately for harder or less well-conditioned problems where reference eigenvalues are not reliable or not known.

- [ ] Evaluate which FEAST variants we want to support (probably most of them), we can look at `~/Code/feast_julia` for the FORTRAN reference implementation.
  - Iterative is important but not worth doing until we cover all the bases with dense variants.
  - FORTRAN reference inventory from `~/Code/feast_julia`:
    - Dense linear FEAST in `src/dense/dzfeast_dense.f90`: standard and generalized dense eigenproblems, with expert custom-contour variants. Families are real symmetric `SY`, complex Hermitian `HE`, real/complex general `GE`, and complex symmetric `SY`.
    - Dense polynomial FEAST in `src/dense/dzfeast_pev_dense.f90`: polynomial eigenvalue problems `PEV` for the same symmetric/Hermitian/general families, again with expert custom-contour variants.
    - Banded linear FEAST in `src/banded/dzfeast_banded.f90`: standard and generalized banded eigenproblems for real symmetric banded `SB`, complex Hermitian banded `HB`, real/complex general banded `GB`, and complex symmetric banded `SB`.
    - Sparse direct FEAST in `src/sparse/dzfeast_sparse.f90`: standard and generalized CSR problems for real symmetric `SCSR`, complex Hermitian `HCSR`, real/complex general `GCSR`, and complex symmetric `SCSR`. These paths are built around direct sparse factorizations, historically MKL/PARDISO.
    - Sparse iterative IFEAST in `src/sparse/dzifeast_sparse.f90`: same CSR problem families as sparse direct FEAST, but shifted systems are solved by iterative kernels such as BiCGSTAB/Arnoldi rather than direct factorization.
    - Sparse polynomial FEAST in `src/sparse/dzfeast_pev_sparse.f90` and `src/sparse/dzifeast_pev_sparse.f90`: direct and iterative CSR polynomial eigenvalue variants, with several real/Hermitian wrappers reducing to complex general or complex symmetric kernels.
    - Parallel sparse FEAST in `src/sparse/pdzfeast_sparse.f90`, `src/sparse/pdzifeast_sparse.f90`, `src/sparse/pdzfeast_pev_sparse.f90`, and `src/sparse/pdzifeast_pev_sparse.f90`: MPI-prefixed sparse direct, sparse iterative, and sparse polynomial variants.
    - RCI kernels in `src/kernel/dzfeast.f90`: reverse-communication interfaces for real symmetric, complex Hermitian, real/complex general, complex symmetric, and polynomial kernels. These are the algorithmic core behind most wrappers.
    - Contour utilities in `src/kernel/feast_tools.f90`: interval contours, general complex circular/elliptic contours, custom contour nodes/weights, and rational filter evaluation.
    - Sparse search helpers in `src/sparse/dzfeast_sparse.f90`: stochastic interval search for real symmetric and complex Hermitian sparse standard/generalized problems.
  - Proposed Julia support target:
    - First-class: dense standard `feast!`, dense generalized `gen_feast!`, and dense non-Hermitian/generalized dual `dual_gen_feast!`, all with clean contour handling, stable tests, and low allocation per iteration.
    - First-class nonlinear: keep `nlfeast!`/Beyn-style nonlinear variants in scope, but test and document them separately from linear dense FEAST.
    - API boundary: export the first-class dense serial family, the canonical nonlinear prototype, and the explicit distributed FEAST plan/stat types. Keep unfinished IFEAST, iterative nonlinear, and moment/SS experiments available as qualified `FEASTSolver.*` names rather than treating them as the default user interface.
    - Near-term: make sparse matrices work through the same dense-facing API only where the shifted solve abstraction is clean; do not copy the FORTRAN CSR API shape into Julia unless performance forces it. First passes exist for standard `feast!(X, A::AbstractSparseMatrix)`, generalized `gen_feast!(X, A::AbstractSparseMatrix, B)`, and sparse-operator `nlfeast!` with `SparseDirectSolver()` and an experimental `SparseBiCGSTABSolver()`.
    - Sparse direct profiling note: Julia's UMFPACK wrapper stores reusable solve workspace on `UmfpackLU` and supports `lu!(F, A; reuse_symbolic=true)`, but numeric factorization still allocates a new UMFPACK numeric object. Sparse standard/generalized FEAST and sparse-operator NLFEAST now avoid sparse shift-structure allocation and reuse symbolic analysis in `store=false` when an in-place materializer supplies fixed sparsity; remaining direct-solver allocation is dominated by numeric factorization unless `store=true` caches factors.
    - Later: dual sparse generalized FEAST should reuse the sparse solver policy once the direct sparse shifted-solve path has enough tests and allocation/performance measurements.
    - Later: iterative FEAST/IFEAST, after dense and sparse-direct variants have shared workspace abstractions and meaningful convergence diagnostics.
    - Later: contour-level parallelism using Julia mechanisms rather than MPI-first PFEAST compatibility.
    - Defer by default: banded-specific APIs, unless a real benchmark shows band storage/factorization is worth the extra public surface.
    - Defer/bind instead of rewrite: full upstream FEAST compatibility wrappers, MPI PFEAST, and the historical PARDISO-specific sparse interfaces belong with resurrected binary bindings, not the pure-Julia core.

- [ ] There is now a BLAS interface that will allow us to do iterative eigenvalue problems without re-allocating. Being able to fully pre-allocate is a large performance concern here.
  - <https://github.com/DynareJulia/FastLapackInterface.jl>
  - Nonlinear FEAST allocation note: `nlfeast!` now accepts FEAST-native nonlinear operator objects. The preferred contract is `operator_prototype(op)`, `materialize!(M, op, z)` for explicit shifted matrices, and `mul!(Y, op, z, V)` for action-style application. The old `T(z)` closure path remains available through `matrix_operator(T, prototype)` or direct callable compatibility, but it is not the performance-oriented interface.
  - Residual action note: `nlfeast!` and `distributed_nlfeast!` now accept an expert `residual_update=(res, X, R, Λ) -> ...` hook for NEPs where residuals should use matrix-vector actions instead of materializing `T(λ)`. This is useful for the gun problem. NEP-PACK does have `compute_Mlincomb!`, but in the current API it does not take caller-owned output storage; the NLEIGS experiment now uses our native gun gallery action for residual checks instead.

- [ ] Revisit parallelism
  - We have BLAS level parallelism, but once we solve the above we need to investigate how we can parallelize at the contour level. Each node of the quadrature is essentially a separate problem. The difficulty is in managing memory across iterations; we must save LU factorizations for example in the variant where this is feasible to save.
  - We don't want MPI.jl, but the Julia `Distributed` standard library had some issues years ago, so we should investigate.
  - We also need to consider if these have to be totally separate implementations and leave versions that just have BLAS parallelism.
  - We should disable BLAS threading on the calls where we have `Distributed` parallelism.
  - We mostly want local-to-the-node parallelism. Multi-node is interesting but not worth it right now.
  - First-pass design decision: keep the normal `feast!` serial/BLAS-threaded path separate from an explicit `distributed_feast!` path. The distributed path has worker lifecycle, shared-memory, and BLAS-thread policy semantics that should not be hidden behind a boolean keyword on the main API.
  - Dense standard, dense generalized, and dense dual generalized FEAST now have explicit distributed variants and corresponding persistent plan types.
  - Distributed layering: keep separate public plan types for each solver family because they own different subspaces and work buffers, but share worker preparation, fixed node assignment, shared matrix construction, cleanup, stats logging, and the contour-worker execution model.
  - Use fixed process-to-contour-node assignment so each worker can keep node-local memory and, for `store=true`, cached LU factors across iterations.
  - Use `SharedArray` for dense `A`, iteration inputs `X`/`R`, and worker output partial sums on one host. This intentionally does not target multi-node MPI-style distribution.
  - Naming/architecture cleanup: distinguish BLAS-threaded serial execution from local-process shared-memory execution (`Distributed` workers plus shared buffers/local worker state). We are not targeting multi-node or MPI-style distributed-memory execution in the pure-Julia core.
  - For nonlinear FEAST, large-problem experiments should prefer a worker-local `T(z)` path instead of sending dense contour matrices from the master. The current fallback can serialize closures or materialized node matrices for convenience, but the research/performance path should move toward explicit worker-loaded problem definitions.
  - Persistent ownership is required for the FEAST execution model. `DenseDistributedFeastPlan`, `DenseDistributedGeneralizedFeastPlan`, and `DenseDistributedDualGeneralizedFeastPlan` own shared buffers, fixed worker assignments, and worker-local workspaces/cached factors. The convenience `distributed_*feast!` wrappers still exist, but they are intentionally one-shot allocate-use-cleanup paths.
  - Process 1 is currently treated as the coordinator/driver, not a contour worker. In this Julia `Distributed` design it handles QR/Ritz/residual/reduction and remote dispatch, so assigning contour nodes to it would serialize part of the solve unless we redesign the driver around a true MPI-rank style loop.
  - Lightweight profiling is part of the distributed abstraction. `DenseDistributedFeastStats` records setup phase totals, solve phase totals, and a per-iteration log with convergence counts, max residual inside the contour, and QR/Ritz/residual/shared-copy/worker/reduction timings. This is useful both for development and for diagnosing user problems where FEAST is sensitive to contour/subspace/conditioning choices.
  - Benchmark target: compare serial `feast!`/`gen_feast!`/`dual_gen_feast!` against their distributed variants with 1, 2, 4, and 8 worker processes while forcing worker BLAS threads to 1.
  - Initial benchmark result: small diagonal/near-diagonal cases around `N=256`/`512` are dominated by overhead and can misleadingly make 4/8 processes look worse. A full perturbed dense Hermitian case with `N=4096`, `M0=16`, 16 contour nodes, `store=false`, one forced FEAST iteration, and BLAS threads set to 1 produced serial `66.45s`, distributed 1 worker `65.40s`, 2 workers `39.45s`, 4 workers `25.00s`, and 8 workers `19.56s`.
  - Repeated-iteration benchmark result: `N=2048`, `M0=16`, 16 contour nodes, `store=false`, three forced FEAST iterations gave serial `25.06s`, distributed 1 worker `25.35s`, 2 workers `15.56s`, 4 workers `10.32s`, and 8 workers `7.74s`.
  - Stored-factor benchmark result after parallelizing worker initialization: `N=2048`, `M0=16`, 16 contour nodes, `store=true`, three forced FEAST iterations gave serial `4.70s`, distributed 1 worker `10.51s`, 2 workers `6.68s`, 4 workers `4.51s`, and 8 workers `3.58s`. The one-worker distributed path is still slower than serial due process/shared-memory overhead, but multi-process now beats serial.
  - Persistent-plan benchmark result: `N=1024`, `M0=16`, 16 contour nodes, perturbed dense Hermitian, `store=false`, two forced FEAST iterations gave serial `2.28s`; plan setup/solve/total were 1 worker `0.04s/2.41s/2.45s`, 2 workers `0.04s/1.48s/1.52s`, 4 workers `0.09s/0.95s/1.04s`, and 8 workers `0.10s/0.71s/0.81s`.
  - Persistent stored-factor benchmark result: `N=1024`, `M0=16`, 16 contour nodes, perturbed dense Hermitian, `store=true`, four forced FEAST iterations gave serial `1.08s`; plan setup/solve/total were 1 worker `1.30s/0.67s/1.97s`, 2 workers `0.82s/0.43s/1.25s`, 4 workers `0.60s/0.30s/0.91s`, and 8 workers `0.58s/0.22s/0.81s`. This confirms setup/factorization must be separated from the iteration loop for meaningful scaling analysis.

- [ ] Library ergonomics: it should be easy to use, easy to understand, have good diagnostics, and have no unpleasant surprises.
  - For example, when we want debug output there are lots of statistics to follow in each iteration.
  - Dense serial FEAST and the canonical `nlfeast!` now accept `stats=DenseFeastStats()` for lightweight iteration diagnostics and phase timing, mirroring the distributed diagnostics without exposing distributed planning details.

- [ ] Contour abstraction
  - The default should always be circular contour with trapezoidal rule quadrature.
  - First pass: dense serial FEAST, canonical `nlfeast!`, and distributed dense FEAST entrypoints accept explicit `Contour` objects, while the old `c`/`r`/`nodes` convenience path remains circular trapezoidal.
  - `CustomContour(nodes, weights; inside=z -> ...)` now supports custom quadrature rules with explicit eigenvalue classification. This is intentionally required because arbitrary quadrature nodes and weights do not define a region by themselves.
  - The question is how to do other shapes: we could copy big FEAST, or we could think about conformal mappings on the circle. A review paper may cover this, possibly the nonlinear eigenvalue review paper in SIAM.
  - Either way we want a clean abstraction for custom contours that gets completely out of the way for people who just want it to work.
  - May have to revisit stochastic estimation for auto contour finding. Big FEAST may have subspace resizing.

- [ ] Implement SS, Sakurai subspace iteration algorithms
  - We need a working good implementation of this for comparison.
  - In particular we must implement the nonlinear variants, as there is an unexplored relation to the nonlinear FEAST variants.
  - If we can get this, revisit the concept of a unified algorithm uniting Beyn, FEAST, and SS. This seems possible: Beyn-FEAST is already done, and the higher order moments aspect possible with polynomial expansion solutions via the companion problem naturally leads to Sakurai if we want everything to work well.
  - Current research hypothesis: the higher-moment/RII closure probably needs an invariant-pair state `(X, S)` rather than a diagonal eigenpair list `(X, Lambda)`. Then the moment-RII kernel becomes `(X - T(z) \ T(X, S)) * inv(zI - S)`, which closes as `Q_k = X*S^k` for an exact invariant pair. See `experiments/moment_rii/README.md`.
  - First experiment result: lifted invariant-pair normalization and contour-scaled moments are essential. The local refinement is better viewed as invariant-pair/block Newton, with contour SS/Beyn providing the filtered initializer and possible globalization step.
  - Second experiment result: after the initial Hankel/SS realization discovers the state, corrected moments can be used through the shift relation `[Q_0; ...; Q_{K-1}] * S ≈ [Q_1; ...; Q_K]`. This improves the scalar many-eigenvalue radius-10 sine stress case, but radius-20 exposes rank/gauge conditioning problems. A Chebyshev recurrence basis helps organize the multiplication operator but does not solve observability by itself. Next theoretical target is a balanced or orthogonal-basis realization of the multiplication operator, not a larger brute-force Newton system.
  - Third experiment result: two-sided SS-style projected Hankel extraction works as the right realization geometry and avoids forming `K*n` row Hankel matrices. Spurious retained states can seriously contaminate the invariant-pair residual even when wanted scalar eigenpairs look converged, so NLFEAST and moment variants need an explicit deflation/replacement policy rather than treating spurious values as only a reporting issue.
  - Linear control result: projected SS-FEAST recovers ten diagonal eigenvalues with four right-hand sides and `K=3`, matching ordinary FEAST with twelve right-hand sides. A nonnormal Grcar control also refines to machine precision. This validates the two-sided realization/extraction stages for `T(z)=zI-A`; the remaining research problem is the nonlinear corrected-moment update plus deflation/replacement policy.
  - Nonlinear update result: on deficient quadratic and butterfly, repeated projected extraction, shifted realization, and projected extraction plus local invariant-pair Newton all recover the wanted polynomial eigenpairs. Newton gives the cleanest invariant-pair residual when the state dimension is not larger than the physical dimension.
  - Low-dimensional many-eigenvalue result: a nonnormal degree-eight polynomial with `n=4`, four probes, and twenty interior eigenvalues is recoverable with `K=5`; the shifted-realization update converges all twenty even with only eight contour nodes, while `K=4` fails because the Hankel capacity is only sixteen. The experiment harness now avoids incorrectly capping nonlinear realization rank by physical dimension `n`.
  - Gauge result: the scalar `sin(z)` radius-20 stress case exposed that small invariant-pair residuals can hide unusable scalar Ritz values when the small multiplication matrix is extremely nonnormal. Applying diagonal similarity balancing to the shifted-realization state during iteration converges all thirteen interior scalar roots with max residual around `1e-10`.
  - Infinite-root analytic insight: `cos(z)` on a radius-20 contour is a counterexample to "one large contour plus one monomial moment basis." Gauge-balanced shifted moments recover the outer roots, but roots near the contour center are buried; a union of smaller nested gauge-balanced contour solves recovers all twelve interior roots.
  - Companion-polynomial insight: polynomial companion linearization supplies a finite global coordinate system where all polynomial roots live in one enlarged state. General analytic NEPs with infinitely many roots should instead be treated as local finite realizations, so the next direction is gauge-controlled local rational coordinates, nested contours, moment offsets, or other rational bases rather than scalar ad-hoc deflation.
  - Polynomial layer checkpoint: the moment experiment now compares the degree-eight `n=4` many-eigenvalue polynomial against FEAST on the companion pencil; both the companion solve and polynomial-native moment updates recover all twenty finite target roots. A degree-deficient control also confirms that polynomial infinite roots are handled naturally by reversing the polynomial and solving in the `nu=1/lambda` chart.
  - Adaptive-chart checkpoint: scalar chart prototypes grow nested contours for `cos(z)` on radius 20, choose between moment offsets by newly converged roots, and recover all twelve known roots. The first version used exact scalar root counts to size local realizations; the rank-adaptive version now estimates local state size from Hankel singular values and uses the outer-contour rank estimate as the stopping target. The rank threshold is algorithmically significant, so production code needs visible count/rank diagnostics rather than a hidden magic tolerance.
  - Rank-estimation stress result: scalar tests show that near-contour exterior roots can overestimate rank, while large contours can underestimate rank by burying weak interior states. Adaptive charts recover all tested scalar roots for radii 10 and 20, but some `sin`-type radius-20 cases recover all roots despite an underestimated outer target count. Do not stop purely because an estimated count was reached; the outer contour should be run as a final consistency chart.
  - Non-scalar analytic toy result: a diagonal `sin/cos` NEP exposed that pair residual alone is unsafe after scalar residual normalization; physical eigenvectors can become nearly unobservable through the first block `X`. Chebyshev shifted moments plus observable-eigen gauge and best-history stopping recover the radius-10 chart, but radius 20 remains a real failure. Nested rank-adaptive charts recover only the central thirteen roots tightly. A generic lifted Newton cleanup and balanced Hankel scaling did not fix a poor radius-20 pair.
  - Dual-FEAST insight: projected Hankel is not the same thing as dual FEAST. The true dual analogue for higher-moment NLFEAST should carry left/right filtered physical spaces, biorthogonalize them, and extract with the reduced NEP `Y' * T(lambda) * X` rather than relying solely on eigenvalues of the small multiplication matrix `S`.
  - Reduced-NEP extraction result: the first dual/Petrov-Galerkin determinant prototype solves the diagonal `sin/cos` radius-20 failure. Left/right moment bases reduce the problem to `2 x 2`; argument-principle power sums for `det(Y' T(lambda) X)` give the exact count 25, and determinant Newton refinement recovers all 25 roots. Reduced polynomial controls recover all twenty roots of the nonnormal degree-eight `n=4` problem, and an ill-conditioned polynomial control shows a concrete dual/Galerkin split: dual extraction returns the twelve target roots with no residual-small extras, while one-sided Galerkin admits a spurious residual-small root.
  - Iterative dual-RII result: a deliberately bad dual-sensitive polynomial chart starts with residual-small but wrong Ritz values. One two-sided scalar RII step, mirroring dual linear FEAST with right solves by `T(z)` and left solves by `T(z)'`, recovers all twelve target roots and removes spurious values after physical SVD compression. The residual solves can be low-rank compressed from the number of Ritz values to the numerical residual rank, giving a plausible bridge to the compact realization update we actually want.
  - Moment-compressed dual-RII result: on a circular contour the scalar RII factor has a Laurent expansion, so the correction can be represented by residual-solve moments instead of persistent Ritz-vector columns. On the bad dual-sensitive polynomial chart, one residual Laurent moment recovers all twelve target roots to machine precision. On the low-dimensional many-root polynomial control, a rank-deficient initial basis has zero valid target matches, but one residual Laurent moment recovers all twenty target roots. A non-polynomial diagonal-similarity analytic control with `sin`, `cos`, and `sin(z)-0.3` likewise expands a rank-one initial chart to rank three and recovers all twenty radius-10 roots. This is the strongest link so far between higher Hankel moments and a genuinely iterative dual NLFEAST update.
  - Reduced-extractor checkpoint: for generic analytic reduced NEPs, use the argument principle for count only, then use SS/Hankel on the small reduced NEP for roots/vectors. This counted-SS extractor avoids the unstable determinant power-sum root reconstruction and removes the hidden Hankel-rank threshold. It recovers all 38 roots of the three-function radius-20 analytic control after one residual Laurent update. The four-function `exp(z)-1` control remains a chart stress case, pointing to local contours or rational coordinates for stiff analytic functions.
  - SS extraction checkpoint: NEP-PACK's generalized Hankel pencil and the Beyn-style SVD-similarity extraction agree on the successful counted-SS controls and fail similarly on the stiff `exp(z)-1` mixture. The next issue is chart/scaling/residual reliability, not this algebraic choice.
  - Current direction: prioritize robust chart policy over ad-hoc scalar deflation. Next implementation target is a real chart abstraction with diagnostics: contour, coordinate map, moment basis or offset, left/right extraction, gauge, rank/count estimate, physical observability, reduced NEP solver, and merge policy. If this stalls, inspect NEP-PACK and adjacent NEP/SS/Beyn literature for deflation techniques as a fallback.

- [ ] Benchmarking
  - There is a lot to benchmark.
  - Benchmark tooling: `benchmark/` already uses `BenchmarkTools.jl` for local
    dense serial/distributed FEAST timing. Keep using it for repeatable
    micro/meso benchmarks where `@benchmarkable`, sample counts, medians, and
    allocation estimates are useful. Keep publication-style algorithm
    comparisons, such as NLEIGS vs NLFEAST, under `experiments/` with explicit
    markdown logs because those runs need tuned regions, warmup policy, solver
    metadata, and interpretation beyond a single timing distribution.
  - The NLEIGS comparison should stay curated rather than collecting every
    available NEP gallery problem. Current direction: one small dense sanity
    problem (`butterfly`), one large dense polynomial problem (`pep0`, default
    size 3000, center 0, radius 0.095, m roughly 2x the observed interior count),
    and one sparse NLEVP benchmark problem (`gun`, fixed size 9956, center
    140000, radius 30000, m=36, nodes=8) now that sparse NLFEAST has a first
    direct-solver path.
  - Current `pep0` NLEIGS observation: even with BLAS threading and larger block
    sizes, NLEIGS is sensitive to the target radius. At radius 0.1, FEAST
    recovers a wider 32-eigenpair region while NLEIGS recovers only a subset; at
    radius 0.095, both methods recover 27 interior eigenpairs under the
    no-store/no-reuse memory policy. We tried target radii 0.08/0.09/0.095/0.1/0.12,
    `blksize=32`, `blksize=96`, `maxit=150`, `maxdgr=300`, `leja=1/2`, and
    `reusefact=0/1/2`. Aligned 32-point NLEIGS at radius 0.1 improves to 18
    eigenpairs but still misses FEAST's wider 32-eigenpair region; aligned
    32-point NLEIGS at radius 0.095 with no reuse recovers only 7 eigenpairs, so
    the robust NLEIGS default remains the unaligned 96-point polygon. Treat this
    radius/node sensitivity as a comparison result to explain, not as an
    automatic benchmark failure.
  - Future contour-abstraction work should let NLFEAST accept explicit custom
    contours, so we can test FEAST on exactly the same polygonal target sets
    where NLEIGS struggles.
  - Sparse NLFEAST now has a first sparse-direct path. `gun` is part of the
    default NLEIGS comparison set using a native FEAST gallery operator and a
    low-rank-factorized NLEIGS representation. The scalable native
    `schrodinger_movebc` gallery problem remains available as an explicit
    exploratory selector.

- [ ] Resurrect Julia bindings for the upstream FEAST library
  - Revisit the old Julia BinaryBuilder bindings work.
  - Packaging FEAST with Nix would likely help local development and make the binding work more reproducible.
  - Keep this separate from the minimal FEASTSolver.jl dev shell until the binding work is active.

- [ ] Clean up
  - Docs, CI, module layout, types.
  - Important note: some very important historical "test" files are experiments showing significant results. These now live under `experiments/legacy_tests/` so they are preserved but no longer confused with automated tests.
  - Keep dense serial FEAST implementations readable as educational/research references while preserving the preallocated LAPACK paths.
  - Keep distributed FEAST focused on making persistent contour-node ownership and synchronization explicit. Users can read the serial implementation for the mathematical algorithm.
  - Keep `src/feast_experimental.jl` in-tree for now as the unfinished IFEAST/inexact-FEAST prototype.
  - Keep `src/nlfeast_experimental.jl` in-tree for now as nonlinear moment/Beyn/Sakurai-Sugiura research code.
  - Canonical nonlinear path: `nlfeast!` in `src/nlfeast.jl`, the NLFEAST-Beyn hybrid that applies residual inverse iteration directly to Beyn-style contour moments and reduces to linear FEAST in the appropriate sense.
  - Next module-layout cleanup: when a real hierarchy exists, consider moving experimental implementations into an `src/experimental/` folder while keeping them importable.
  - Next nonlinear cleanup: separate the canonical nonlinear implementation from moment/SS experiments enough that the canonical variant can be tested, documented, and profiled independently.

- [ ] Now that package extensions exist, split things into subdirectories and have optional dependencies where useful, for example plots and pseudospectra.
  - <https://discourse.julialang.org/t/quick-tutorial-on-package-extensions/130923>
  - <https://github.com/pebeto/julia_extensions_example>

- [ ] Plots
  - We want to visualize the contour and eigenvalues, optionally pseudospectra.
  - Stretch goal is an interactive contour tool. This is probably not worth it now and seems better suited to a WASM-style interface.

- [ ] Use FEAST contours to get local pseudospectra faster by reducing to small problems, approximately.
  - This is worth exploring in the future.

## Initial Execution Plan

This ordering is based on the current code layout:

- [ ] Stabilize the local iteration loop
  - Use `just smoke` for package load checks.
  - Use `just test` for the current automated test entrypoint.
  - Use `just docs` for documentation checks.
  - Keep `just notes` current with repo assumptions and caveats.

- [ ] Reconcile the public API with the module layout
  - Previously, `src/FEASTSolver.jl` exported `nlfeast_opt!`, but the implementation lived in `src/nlfeast_lapack.jl`, which was not included.
  - Removed obsolete `nlfeast_opt!`; the old LAPACK-specific prototype is obviated by the shared FastLapackInterface helpers.
  - The exported API is now limited to first-class dense serial FEAST, the canonical nonlinear prototype, explicit distributed FEAST, stable research utilities, contours, and diagnostics. Experimental solvers remain in-tree but are no longer exported by default.

- [x] Convert the current test directory into a deliberate test/experiment split
  - `test/runtests.jl` is the only automated test entrypoint right now.
  - Historical standalone research scripts with useful problems and observations now live under `experiments/legacy_tests/`.
  - First pass: classify each file as automated test candidate, benchmark candidate, or experiment.
  - Second pass: move the highest-value deterministic cases into `runtests.jl` or included testsets.

- [ ] Build a test problem catalog before changing algorithms
  - Dense Hermitian standard eigenproblem cases for `feast!`.
  - Dense generalized cases for `gen_feast!` and `dual_gen_feast!`.
  - Non-Hermitian cases with known or reference eigenvalues.
  - Nonlinear eigenvalue cases from `NonlinearEigenproblems.jl`.
  - MatrixDepot / MatrixMarket sparse cases once standard sparse `feast!` direct-solver behavior is measured beyond the current deterministic smoke tests.
  - Prefer MatrixDepot cases with useful metadata for the real medium-sized test set.

- [ ] Clean up contour behavior before broadening the algorithm surface
  - `CustomContour` has `in_contour` support when constructed with an explicit predicate.
  - Rectangular contour constructors have known real-coordinate type bugs.
  - The default user path should remain circular contour plus trapezoidal quadrature.
  - Custom contours should be possible without making the simple case harder.

- [ ] Inventory FEAST variants against the FORTRAN reference
  - Compare the local implementation to `~/Code/feast_julia`.
  - Identify dense standard, dense generalized, non-Hermitian, nonlinear, iterative, and stochastic-estimation variants.
  - Support dense variants first; iterative variants should wait until allocation and test coverage are under control.

- [ ] Make allocation behavior measurable before optimizing it
  - Add benchmarks around current dense standard/generalized/nonlinear paths.
  - Track allocations per iteration and per contour node.
  - Evaluate `FastLapackInterface.jl` where it replaces repeated LAPACK workspace allocation.

- [ ] Revisit parallelism after the dense implementation is stable
  - Separate BLAS-threaded implementations from contour-node parallel implementations if needed.
  - Investigate Julia `Distributed` for local-node contour parallelism.
  - Disable BLAS threading around distributed contour work if the implementation goes that direction.

- [ ] Defer optional surfaces until the core shape is clearer
  - FEAST BinaryBuilder/Nix packaging belongs with upstream FEAST binding resurrection.
  - Plotting, pseudospectra, and optional visualization should use package extensions or separate environments.
  - CI should eventually own Julia 1.10, 1.11, and 1.12 compatibility checks.
