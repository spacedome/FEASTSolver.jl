# Cache Contract And Policy

The cache is identified by `(operator state, chart, node set, solve tolerance)`.
Changing any component creates a new generation. Numerical factors and response
samples cannot cross generations. Physical state, invariant pairs, and selected
moment blocks may cross generations as iteration data.

There are two implementation layers. A factor cache owns node-local numerical
solve plans. The existing `FusedNLFEAST.ContourSampleCache` owns sampled probe
responses and their moment algebra. A bridge test feeds the sample cache from
the frozen factor cache and obtains identical moment sequences. Production may
stream and evict responses, but these two ownership roles should remain
separate.

One frozen generation may provide:

```text
base right moments             Nℓ RHS
base right/left moments        2Nℓ RHS
residual-corrected moments     Nr RHS after residual compression
Hermitian closure response     Np RHS per Krylov action
dual closure response          2Np RHS per Krylov action
```

For `T(ρ,z)=zI−H(ρ)`, corrected moments reduce identically to the ordinary
FEAST filter. The cache must select that identity rather than spend `Np` RHS on
an explicit residual solve. For a general holomorphic `T(θ,z)`, the explicit
corrected transfer is required.

The polynomial cache now compresses the physical invariant-pair residual before
node solves and records every realized rank `r`. A constructed rank-one
residual reproduces the uncompressed corrected moments while reducing correction
width from `p` to one. This compression is opportunistic; full-rank residuals
retain width `p`.

The current policy state machine is:

```text
build frozen cache
→ form corrected moment block
→ update bounded accumulated space
→ optionally spend cache on one full response correction
→ invalidate cache when θ changes
→ perform reduced closure iterations without the stale cache
→ classify the phase as closure-limited or leakage-limited
→ choose the next refresh policy
```

Periodic response is a tuning control. Adaptive response requires the previous
closure defect to exceed both the final tolerance and a multiple of invariant
leakage, and enforces a minimum refresh interval. This prevents response work
when missing physical directions, rather than closure convergence, limit the
current space.

For simultaneous spectral/state nonlinearity, repeated invariant-pair repair
uses the same frozen factors. It is stopped by the physical invariant-pair
residual. Current evidence treats this as accuracy forcing, not a default
performance step: tighter spectral repair has not reduced closure refreshes.

## GPU Mapping

Each contour node is an ownership unit. A worker or device keeps its node
factor and processes base, residual, adjoint-left, and response RHS blocks in
sequence. Responses are reduced into moment blocks immediately and need not be
retained. Only moment reductions and Krylov scalars cross workers.

Block width is a hardware parameter. Narrow higher moments minimize RHS count;
wider depth-one or oversampled probes may deliver higher sparse/dense triangular
solve throughput. A production policy must measure factorization and block
solve costs at representative widths rather than infer wall time from column
counts alone.

Response Krylov introduces synchronization after each action. On multiple GPUs,
its factor-reuse benefit must exceed both additional RHS work and reduction
latency. Short-recurrence CG is available only for self-adjoint positive
closure equations; general dual or combined response requires GMRES or another
nonsymmetric method.

## Prototype Boundary

The experiment deliberately keeps density, polynomial, and dual cache types
separate while their algebra is being tested. A later common interface should
own nodes, factors, validity generation, solve diagnostics, RHS accounting, and
streaming moment reducers. It should not encode one closure representation or
assume that right and left responses are both present.

The following remain open:

- a contraction/cost predictor better than periodic or defect/leakage forcing;
- response for a general simultaneous `T(θ,z)` with left residue normalization;
- a low-rank Krylov realization for full nonlocal projector response; each
  tested Jacobian action has the expected rank bound `2p`, but no iterator yet
  preserves that factorization across Krylov combinations;
- inexact-solve tolerance forcing and rejected-step cache reuse;
- stored, streamed, and distributed factor policies with measured GPU costs;
- safe fixed-contour updates for strong non-Hermitian closure drift.
