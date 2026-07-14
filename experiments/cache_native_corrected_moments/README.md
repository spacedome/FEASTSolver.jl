# Cache-Native Corrected Moments

This experiment tests whether one frozen contour factor cache should serve both
moment-space construction and selected closure-response actions.

`CACHE_CONTRACT.md` records the validity generation, policy state machine, GPU
execution mapping, and the operations that are not yet justified.
`FINDINGS.md` records the accepted, conditional, and rejected conclusions.

For fixed nonlinear state `θ`, the cache owns the factors of `T(θ,zⱼ)`. It is
valid only until `θ` changes. Within that lifetime it can form positive moments,
explicit residual-corrected moments, adjoint actions, and contour Jacobian
actions without another factorization.

The first control uses `T(ρ,z)=zI−H(ρ)`. Here explicit corrected moments reduce
exactly to the ordinary FEAST filter. The specialized path therefore spends
`Nℓ` base RHS rather than `Np` residual RHS. Tests pin this identity, exact
agreement with the established pre-extraction window, response-action reuse,
and cache invalidation.

The integrated schedule is:

```text
freeze ρ and build node factors
→ form 2d corrected moments
→ append the first d blocks to a bounded window
→ optionally spend the same factors on one full response step
→ invalidate factors after changing ρ
→ continue reduced iteration in the accumulated space
→ refresh when leakage remains
```

On the `n=40,p=3,g=5,N=48` control, four-block windowing with two reduced
steps takes 12 refreshes and 576 RHS. A response action every third refresh
takes 9 refreshes, 432 base RHS, and 1296 response RHS. It saves 144 node
factorizations for 1152 additional total RHS, a crossover of 8 RHS solves per
saved factorization. Growing memory moves from 8 to 7 refreshes with a crossover
of 17 RHS per saved factorization.

These are operation-count results, not timing or scaling claims. The response
period is an experimental upper-envelope policy. The adaptive policy spends a
response only when the previous reduced phase remains closure-limited rather
than leakage-limited and a minimum refresh interval has elapsed. It is more
conservative than the best fixed period and correctly spends no response work
when the reduced solve is already adequate.

For simultaneous `ρ,z` nonlinearity, the cache supports repeated corrected
invariant-pair extraction before changing `ρ`. On the quadratic control this
can reduce the frozen-state spectral residual by several orders without new
factors, but does not reduce closure refreshes. Residual-forced repair is
therefore a correctness/accuracy policy; fixed repeated repair is rejected as
oversolving.

For the non-Hermitian similarity control, the same LU supplies right,
adjoint-left, and oblique-projector response actions. The dual response agrees
with finite differences. It reduces refreshes when the moment window or reduced
inner solve is deliberately constrained, but adds no value once a three-block
space is adequately solved. A general dual response uses GMRES; the present
similarity control retains the Hermitian positive-definite closure and uses CG.

The warmed sparse `n=256,p=4` control does not show a universal timing win.
Cached response reduces an under-solved four-block schedule from 13 to 10
refreshes and approximately `3.8–4.1 s` to `3.0–3.2 s`; taking one more reduced
step uses the same 10 refreshes and approximately `2.9–3.1 s`. Growing five-step
moments remain fastest at approximately `2.5–2.8 s`. These timings support a
runtime policy, not a new fixed default.

The separate two-dimensional sparse kernel probe measures one factorization at
roughly 30–34 RHS-column solves on the current CPU. That scalar ratio would
favor the periodic growing-cache policy in the operation-count model, while the
end-to-end one-dimensional benchmark still favors window-only growing moments.
Reduced work, chart construction, allocation, and synchronization therefore
belong in any production selector; factor/RHS ratio alone is insufficient.

Run:

```sh
nix develop --command julia --project=. --startup-file=no \
  experiments/cache_native_corrected_moments/test/runtests.jl

julia --project=. --startup-file=no \
  experiments/cache_native_corrected_moments/policy_sweep.jl

julia --project=. --startup-file=no \
  experiments/cache_native_corrected_moments/representation_sweep.jl

julia --project=. --startup-file=no \
  experiments/cache_native_corrected_moments/sparse_benchmark.jl

nix develop --command julia --project=. --startup-file=no \
  experiments/cache_native_corrected_moments/kernel_cost_probe.jl
```
