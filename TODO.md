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
    - Near-term: make sparse matrices work through the same dense-facing API only where the shifted solve abstraction is clean; do not copy the FORTRAN CSR API shape into Julia unless performance forces it.
    - Later: iterative FEAST/IFEAST, after dense and sparse-direct variants have shared workspace abstractions and meaningful convergence diagnostics.
    - Later: contour-level parallelism using Julia mechanisms rather than MPI-first PFEAST compatibility.
    - Defer by default: banded-specific APIs, unless a real benchmark shows band storage/factorization is worth the extra public surface.
    - Defer/bind instead of rewrite: full upstream FEAST compatibility wrappers, MPI PFEAST, and the historical PARDISO-specific sparse interfaces belong with resurrected binary bindings, not the pure-Julia core.

- [ ] There is now a BLAS interface that will allow us to do iterative eigenvalue problems without re-allocating. Being able to fully pre-allocate is a large performance concern here.
  - <https://github.com/DynareJulia/FastLapackInterface.jl>
  - Nonlinear FEAST allocation note: `nlfeast!` now reuses LAPACK workspaces, but the function-valued nonlinear operator API still allocates because users pass `T(z)` and dense shifted/factored matrices must be materialized. Consider an optional expert hook `T!(M, z)` later, with `T(z)` remaining the canonical mathematical interface and `T!` only used when supplied. This is especially relevant for user-defined dense nonlinear problems; `NonlinearEigenproblems.jl` generally exposes allocating full-matrix `compute_Mder(nep, λ)` plus matrix-vector-style `compute_Mlincomb`/`compute_MM`, not a universal `compute_Mder!(M, nep, λ)` convention we can rely on.
  - Residual action note: `nlfeast!` and `distributed_nlfeast!` now accept an expert `residual_update=(res, X, R, Λ) -> ...` hook for NEPs where residuals should use matrix-vector actions instead of materializing `T(λ)`. This is useful for the NEP-PACK gun problem. NEP-PACK does have `compute_Mlincomb!`, but in the current API it does not take caller-owned output storage and the native gun type falls back through a sum implementation, so it is not a general nonallocating `T!(λ, V)` replacement.

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
  - The question is how to do other shapes: we could copy big FEAST, or we could think about conformal mappings on the circle. A review paper may cover this, possibly the nonlinear eigenvalue review paper in SIAM.
  - Either way we want a clean abstraction for custom contours that gets completely out of the way for people who just want it to work.
  - May have to revisit stochastic estimation for auto contour finding. Big FEAST may have subspace resizing.

- [ ] Implement SS, Sakurai subspace iteration algorithms
  - We need a working good implementation of this for comparison.
  - In particular we must implement the nonlinear variants, as there is an unexplored relation to the nonlinear FEAST variants.
  - If we can get this, revisit the concept of a unified algorithm uniting Beyn, FEAST, and SS. This seems possible: Beyn-FEAST is already done, and the higher order moments aspect possible with polynomial expansion solutions via the companion problem naturally leads to Sakurai if we want everything to work well.

- [ ] Benchmarking
  - There is a lot to benchmark.

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
  - MatrixDepot / MatrixMarket sparse cases once the basic deterministic tests are organized.
  - Prefer MatrixDepot cases with useful metadata for the real medium-sized test set.

- [ ] Clean up contour behavior before broadening the algorithm surface
  - `CustomContour` exists but lacks `in_contour`.
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
