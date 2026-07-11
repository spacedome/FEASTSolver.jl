# Acceptance Boundary

The algorithm is complete only when every mandatory gate below passes without
problem-specific root oracles in the solver path.

## Mathematical gates

- PASS: finite circular fixed point and linear FEAST identity.
- PASS: one-moment NLFEAST-Beyn and higher-moment SS/Hankel reductions.
- PASS: polynomial companion agreement and structured analytic state action.
- PASS: algebraic multiplicity, defective state, and shared physical vectors.
- OPEN: local contraction theorem on lifted invariant-pair spaces.
- OPEN: perturbation theorem for common/independent extraction and inexact
  solves.

## Numerical gates

- PASS: residual-proportional inexact solves show no fixed accuracy floor.
- PASS: analytic scaling and weak probes are detected as rank/count failures.
- PASS: invalid branch charts are rejected, and argument/determinant counts no
  longer self-certify from integrality plus one refinement comparison.
- PASS: rectangular subdivision repairs the 31-root and analytic-scaling
  controls; derivative-certified determinant winding completes the sine case
  without a root oracle and rejects spectrum on shared cuts.
- PARTIAL: an experiment driver integrates common-Schur iteration, independent
  fallback, overlap gating, rollback, and initial probe augmentation. Main-repo
  API integration remains open.
- PASS: a certified-count experiment orchestrator refines and bisects
  rectangles, shifts cuts until child counts are additive, retains and assembles
  invariant-pair leaves, and conserves the 31-root sine count. Public/main-repo
  integration remains a production gate; a single large power-moment
  realization is not an accepted fallback.
- PASS: cluster and defective acceptance uses the persistent invariant pair;
  split modal values are not a convergence certificate.
- PASS: independent pairs are lift-normalized before common coupling, making
  the overlap gap invariant under separate realization similarities.
- PASS: a continuous-boundary lower state-resolvent separation, rather than
  eigenvalue margin alone, can gate nonnormal chart candidates.
- PASS: black-box Cauchy actions use quadrature independent of the solve grid
  and cannot certify convergence without an explicit resolution certificate.
- PASS: worst-direction inexact-solve controls expose resolvent amplification;
  a requested relative residual alone is explicitly not accepted as an inner
  accuracy guarantee.
- PARTIAL: a certified signed-index plus supplied pole-partial-multiplicity
  count works for separated poles. Coincident positive/negative partial
  directions still require clearing, a pole-cancelled action, or linearization.

## Production gates

- PASS: live cache memory can be independent of iteration count.
- PASS: streaming TSQR avoids tall-Hankel materialization.
- PASS: a structured sparse state path solves the `n=128` Schrodinger control
  to at most `2.4×10⁻¹¹` backward error.
- PASS: the corrected common-state path retains all 17 modes of the `n=9956`
  gun problem and reaches `7.1×10⁻¹¹` worst two-sided modal backward error.
- OPEN: public sparse action/solve interface with no dense conversion.
- PARTIAL: common-state research drivers pass loaded string, sparse
  Schrodinger, gun, delay, polynomial, multiplicity, and high-count controls.
  These still need one public driver and fixed production seed sweeps.
- OPEN: arithmetic types and tolerances are generic rather than hard-coded to
  `ComplexF64` and `Float64`.
- PASS: exact-moment precision sweeps quantify power-Hankel conditioning and
  rule out reduced-only precision escalation as a substitute for partitioning.
- PARTIAL: the state algorithm maps directly to `FEAST.rs` contour ownership,
  factorization policy, and dual sparse workspaces. Nonlinear workspace
  implementation, bounded-allocation benchmarks, and cancellation remain open.
- PARTIAL: solve results can report achieved residual/failure for every node
  and side, and the driver can require and bound them. Rejected outer steps do
  not yet tighten and retry tolerance-aware solves; modal conditioning is a
  separate structured diagnostic but is not attached to the result object.
- PARTIAL: result objects distinguish matched from certified counts and
  certified from heuristic residual convergence. Candidate, invalid-domain,
  and solve failures are typed; public diagnostics still need consolidation.

No production-ready claim is allowed while any mandatory item is `OPEN`.
