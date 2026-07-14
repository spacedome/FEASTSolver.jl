# Eigenvector-Nonlinear Moment FEAST

This experiment studies Hermitian eigenvector-dependent problems

```text
H(P)X = XΛ,   XᴴX = I,   P = XXᴴ.
```

The nonlinear state is the occupied projector or its density. One iteration
freezes `H(P)` only long enough to evaluate its Riesz projector with contour
moments, then feeds the projected density back into `H`. FEAST is therefore
the nonlinear fixed-point map itself, not an inner eigensolver.

The experiment now keeps several solver realizations separate: direct closure
iteration, full response Newton, fixed/adaptive thick restart, and pre/post
extraction moment windows with finite or growing memory. `run.jl` remains the
original response-Newton control; `categorical_sweep.jl` is the current
cross-realization comparison.

The test problem is a finite-difference one-dimensional quantum Hamiltonian
with a harmonic trap and repulsive contact mean field,

```text
H(ρ) = −½Δ + ½x² + gρ(x),
ρ(xᵢ) = diag(XXᴴ)ᵢ / h.
```

With one occupied orbital this is the stationary Gross–Pitaevskii equation.
The main run uses four occupied orbitals to exercise genuine higher-moment
recovery with a probe narrower than the occupied space; the tests include both
forms.

`MASTER_FORM.md` separates spectral and eigenvector nonlinearities and records
the exact reductions to linear FEAST, dual FEAST, Beyn/NLFEAST-Beyn,
SS/higher-moment realization, projector SCF, and response Newton.
`REPRESENTATIONS.md` compares density, orbital, level-shifted, reduced,
non-Hermitian dual, finite-temperature, and orbital-specific branches.
`COUNT_POLICY.md` separates fixed-region count certification from oracle-free
moving occupied windows.
`TWO_TIMESCALE.md` separates inner stopping, FEAST repair, and subspace-memory
choices and records the fixed-memory occupied-enrichment candidate.
`VARIANT_LEDGER.md` assigns stable names and evidence boundaries to every
retained or still-promising realization.
`COMPUTATIONAL_MODEL.md` records the factorization/RHS cost envelope, sparse
evidence, high-moment width trade, and large-occupation branch.
`COMBINED_NONLINEARITY.md` exercises a problem nonlinear in both `ρ` and `z`
and verifies its linear-pencil reduction.
`NONLOCAL_PROJECTOR.md` checks the surviving schedules on a Hamiltonian that
depends on the full off-diagonal projector rather than its density.
`DUAL_WINDOW.md` records the coupled right/left accumulated realization and its
single-factor-cache evidence.
The sibling `../cache_native_corrected_moments` experiment tests a factor cache
that supplies both moment construction and selected full closure-response
actions before the nonlinear state invalidates it.

Run:

```sh
nix develop --command julia --project=. --startup-file=no \
  experiments/eigenvector_nonlinearity/run.jl

nix develop --command julia --project=. --startup-file=no --check-bounds=yes \
  experiments/eigenvector_nonlinearity/test/runtests.jl

nix develop --command julia --project=. --startup-file=no \
  experiments/eigenvector_nonlinearity/variant_costs.jl

nix develop --command julia --project=. --startup-file=no \
  experiments/eigenvector_nonlinearity/moment_width_sweep.jl

nix develop --command julia --project=. --startup-file=no \
  experiments/eigenvector_nonlinearity/categorical_sweep.jl

nix develop --command julia --project=. --startup-file=no \
  experiments/eigenvector_nonlinearity/memory_costs.jl

nix develop --command julia --project=. --startup-file=no \
  experiments/eigenvector_nonlinearity/history_reuse_sweep.jl

nix develop --command julia --project=. --startup-file=no \
  experiments/eigenvector_nonlinearity/nonlocal_projector_sweep.jl

nix develop --command julia --project=. --startup-file=no \
  experiments/eigenvector_nonlinearity/dual_window_sweep.jl

nix develop --command julia --project=. --startup-file=no \
  experiments/eigenvector_nonlinearity/combined_window_sweep.jl
```

Primary context:

- Gavin and Polizzi introduce the accumulated-subspace nonlinear-eigenvector
  NLFEAST schedule: <https://arxiv.org/abs/1211.4261>.
- Cai, Zhang, Bai, and Li formulate the unitarily invariant NEPv as
  `H(P)X=XΛ`, `P=XXᴴ`: <https://doi.org/10.1137/17M115935X>.
- Upadhyaya, Jarlebring, and Rubensson analyze SCF as a density-matrix
  fixed-point map through its Jacobian: <https://doi.org/10.3934/naco.2020018>.
- Jarlebring, Kvaal, and Michiels develop nonlinear inverse iteration and a
  Gross–Pitaevskii application: <https://doi.org/10.1137/130910014>.
