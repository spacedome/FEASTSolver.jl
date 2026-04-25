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
    - Near-term: make sparse matrices work through the same dense-facing API only where the shifted solve abstraction is clean; do not copy the FORTRAN CSR API shape into Julia unless performance forces it.
    - Later: iterative FEAST/IFEAST, after dense and sparse-direct variants have shared workspace abstractions and meaningful convergence diagnostics.
    - Later: contour-level parallelism using Julia mechanisms rather than MPI-first PFEAST compatibility.
    - Defer by default: banded-specific APIs, unless a real benchmark shows band storage/factorization is worth the extra public surface.
    - Defer/bind instead of rewrite: full upstream FEAST compatibility wrappers, MPI PFEAST, and the historical PARDISO-specific sparse interfaces belong with resurrected binary bindings, not the pure-Julia core.

- [ ] There is now a BLAS interface that will allow us to do iterative eigenvalue problems without re-allocating. Being able to fully pre-allocate is a large performance concern here.
  - <https://github.com/DynareJulia/FastLapackInterface.jl>

- [ ] Revisit parallelism
  - We have BLAS level parallelism, but once we solve the above we need to investigate how we can parallelize at the contour level. Each node of the quadrature is essentially a separate problem. The difficulty is in managing memory across iterations; we must save LU factorizations for example in the variant where this is feasible to save.
  - We don't want MPI.jl, but the Julia `Distributed` standard library had some issues years ago, so we should investigate.
  - We also need to consider if these have to be totally separate implementations and leave versions that just have BLAS parallelism.
  - We should disable BLAS threading on the calls where we have `Distributed` parallelism.
  - We mostly want local-to-the-node parallelism. Multi-node is interesting but not worth it right now.

- [ ] Library ergonomics: it should be easy to use, easy to understand, have good diagnostics, and have no unpleasant surprises.
  - For example, when we want debug output there are lots of statistics to follow in each iteration.

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
  - Important note: some very important test files are not really test files but experiments showing significant results. We should think how to lay things out so these have their own space. Making this a real library is not important enough to pull them out too aggressively.

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
  - `src/FEASTSolver.jl` exports `nlfeast_opt!`, but the implementation is in `src/nlfeast_lapack.jl`, which is not currently included.
  - Decide whether `nlfeast_lapack.jl` is supported code, an experiment, or dead code.
  - Make the supported and experimental exports explicit.

- [ ] Convert the current test directory into a deliberate test/experiment split
  - `test/runtests.jl` is the only automated test entrypoint right now.
  - Many other files in `test/` are standalone research scripts with useful problems and observations.
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
