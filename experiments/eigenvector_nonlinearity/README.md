# Eigenvector-Nonlinear Moment FEAST

This experiment studies Hermitian eigenvector-dependent problems

```text
H(P)X = XΛ,   XᴴX = I,   P = XXᴴ.
```

The nonlinear state is the occupied projector or its density. One iteration
freezes `H(P)` only long enough to evaluate its Riesz projector with contour
moments, then feeds the projected density back into `H`. FEAST is therefore
the nonlinear fixed-point map itself, not an inner eigensolver.

The response-Newton variant differentiates that contour projector using the
same node factorizations and occupied-width right-hand sides. It solves the
density fixed-point correction with a matrix-free Krylov iteration. This is
the main algorithm in `run.jl`; plain moment-projector SCF remains as a control.

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
`COMPUTATIONAL_MODEL.md` records the factorization/RHS cost envelope, sparse
evidence, high-moment width trade, and large-occupation branch.
`COMBINED_NONLINEARITY.md` exercises a problem nonlinear in both `ρ` and `z`
and verifies its linear-pencil reduction.

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
```

Primary context:

- Cai, Zhang, Bai, and Li formulate the unitarily invariant NEPv as
  `H(P)X=XΛ`, `P=XXᴴ`: <https://doi.org/10.1137/17M115935X>.
- Upadhyaya, Jarlebring, and Rubensson analyze SCF as a density-matrix
  fixed-point map through its Jacobian: <https://doi.org/10.3934/naco.2020018>.
- Jarlebring, Kvaal, and Michiels develop nonlinear inverse iteration and a
  Gross–Pitaevskii application: <https://doi.org/10.1137/130910014>.
