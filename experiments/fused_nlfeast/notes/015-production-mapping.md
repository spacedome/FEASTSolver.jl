# 015: Production Mapping

The nonlinear state iterator is compatible with the current `FEAST.rs` linear
execution abstractions. `ContourSlice` already gives stable node ownership,
`FactorizationPolicy` already selects stored versus scratch factors, and the
parallel sparse dual workspaces already own worker-local factor slots and
right/left residual buffers.

One worker should own each contour node and use the same factorization for
primal and adjoint solves. Residual bases are broadcast per outer update;
workers accumulate right and left corrected moments locally and reduce only
the moment blocks. Ho-Kalman extraction, lift normalization, divided-overlap
coupling, and Schur restriction remain small central work. Responses need not
survive moment formation, so the Rust workspace can preallocate bounded
scratch rather than reproduce the Julia research cache.

Partition leaves require separate contour workspaces because shifted cuts and
node refinement change the nodes; this is orchestration rather than a change
to the solve kernel. The current Rust code already tests cached/uncached and
parallel/sequential equivalence. Missing production work is a nonlinear
operator/action trait, variable local state/moment workspace layout,
tolerance-aware solve reports, cooperative cancellation, and end-to-end
benchmarks. None of these requires changing the moment algorithm.
