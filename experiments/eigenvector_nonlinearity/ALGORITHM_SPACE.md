# Algorithm Space

## Core

`MASTER_FORM.md` gives the coupled residual system and the exact reductions.

The common object is a state-dependent holomorphic pencil

```text
T(θ,z),
```

where `θ` is physical nonlinear data such as a density, projector, or tuple of
orbital projectors. Contour moments of the corrected transfer realize an
invariant pair `(X,S)`, and a closure map `D` returns new nonlinear data:

```text
θ → T(θ,·) → corrected contour realization `(X,S)` → D(X,S) → θ⁺.
```

For `T(ρ,z)=zI−H(ρ)` and `D(X,S)=diag(XXᴴ)/h`, this is the nonlinear density
projector map. If `T` does not depend on `θ`, it is moment NLFEAST. If `T` is
linear in `z`, the corrected response is ordinary FEAST. Depth one gives the
usual wide-probe FEAST/NLFEAST-Beyn limit; greater depth gives SS/Hankel
realization from a narrow probe.

The distinction between spectral nonlinearity in `z` and eigenvector
nonlinearity in `θ` is essential.  The implementation exercises each
separately and together: `combined.jl` applies the same corrected realization
to a quadratic `T(ρ,z)` and reduces to the dedicated projector step when the
quadratic coefficient vanishes.

## Outer Updates

`projector SCF`

```text
θ⁺ = D(X,S).
```

This is the cheapest iteration and the correct baseline. It may diverge when
the projector-map Jacobian has spectral radius above one.

`mixed projector SCF`

```text
θ⁺ = (1−α)θ + αD(X,S).
```

It needs no derivative and is robust for a suitable `α`, but can require many
new contour factorizations.

`Anderson/DIIS projector iteration`

This builds a multisecant inverse from previous density defects. It adds only
small dense least-squares work and no contour solves. It is the preferred
default when factorizations are cheap relative to repeated solves or no
Hamiltonian derivative is available.

`contour-response Newton`

This applies the exact derivative of the Riesz projector with occupied-width
contour solves and solves `(I−DΦ)δθ=Φ(θ)−θ` by Krylov iteration. It minimizes
new factorizations and is attractive when node factorizations dominate. Its
cost is additional right-hand sides and a derivative action `DH[δθ]`.
For repulsive local-density models `I−DΦ` is positive definite, so conjugate
gradients provide a short-recurrence response solve.  General nonsymmetric
closures retain GMRES.

`hybrid`

Anderson or mixing supplies globalization; response Newton is enabled when the
count is stable and the defect is small. This is the likely robust production
policy, but the core remains the projector map and its response.

As an optional fixed-chart safeguard, a local-density step can be capped by the
Weyl-safe condition
`g‖δρ‖∞ < s mΓ`, where `mΓ` is the current occupied-to-boundary margin and
`0<s<1`.  This prevents the outer accelerator from ejecting a currently
captured state.  It does not prevent an unobserved exterior state from entering;
that requires an exterior buffer/count policy.  The worst-case Weyl cap is too
conservative to enable by default; it is a retry mechanism after chart
containment becomes doubtful.

The three useful update policies form a cost envelope rather than a strict
ranking.  Anderson minimizes response right-hand sides, tight Newton minimizes
new factorizations, and adaptively inexact Newton lies between them.  The
reproducible comparison in `variant_costs.jl` reports the crossover in units of
RHS-column solves per new node factorization.

`level-shifted projector FEAST`

Filter `H(P)−σP` and retain its lowest occupied projector.  Every solution of
the original problem is preserved, with its occupied eigenvalues shifted by
`−σ`; the added gap can make the projector iteration contractive without
response right-hand sides.  The tradeoff is a new parameter, slow convergence
when the shift is excessive, a low-rank nonlocal term in each node matrix, and
the need to carry the full projector rather than density alone.  This is a
genuinely different state representation and is evaluated separately from
density mixing.

## Alternative Nonlinear States

`density`: minimal for local-density Hamiltonians and compatible with selected
inversion. It loses information needed by nonlocal exchange.

`projector or density matrix`: gauge invariant and sufficient for Hartree-Fock
and general unitarily invariant NEPv problems. Storing it densely is usually
impossible; low-rank orbitals or selected entries represent it.

`occupied orbitals`: computationally compact but carries an arbitrary unitary
gauge. Algorithms must work on a Grassmann quotient or fix a gauge.

Gauge-aligned orbital mixing is a valid Grassmann-retraction variant: align the
new occupied basis to the old one by a polar factor, mix, then reorthogonalize.
At full step it reduces to projector SCF.  At fractional step it generally
differs from density mixing and carries information that a local-density
Hamiltonian does not need; it is mainly relevant when the closure itself needs
orbitals or the full projector.

`tuple of rank-one projectors`: required for orbital-specific nonlinearities.
It is a product-Grassmann problem and generally does not admit one common
Hamiltonian projector.

`finite-temperature density matrix`: replace the sharp Riesz projector by the
Fermi function `fβ(H−μ)`. This removes the spectral-gap/count discontinuity and
connects directly to pole expansion and selected inversion. Moment extraction
is unnecessary if only density and energy are required.

## Realization Choices

Wide depth-one FEAST minimizes reduced realization work but uses one RHS per
state. Higher moments trade RHS width for moment depth. Polynomial basis
changes are exact pencil equivalences; Loewner coordinates are a distinct
rational realization. Density-only selected inversion and stochastic diagonal
estimation avoid occupied orbitals, but they no longer provide invariant pairs
without an additional recovery step.

## Boundaries

- A sharp zero-temperature projector requires a persistent gap and fixed
  occupied count.
- Metallic or changing-count problems favor finite-temperature occupation and
  chemical-potential iteration.
- Non-Hermitian problems require left/right Riesz projectors and the dual
  response.
- Orbital-specific nonlinearities require a product state or a lifted block
  operator; one density projector is insufficient.
- Response compression, selected inversion, sparse factor reuse, and
  preconditioning change cost but not the mathematical core.

## Current Synthesis

The common coupled system is unambiguous:

```text
physical state θ
  → state-dependent operator T(θ,z)
  → residual-corrected local moment realization (X,S[,Y])
  → gauge-invariant physical closure D
  → accelerated update of θ.
```

For Hermitian `H(P)` this collapses further to the occupied Riesz-projector map
`P⁺=ΠΓ(H(P))`. Moment depth and probe width are realization coordinates of
that same map. The best schedule for coupling reduced nonlinear work to FEAST
refreshes is not yet settled. In particular, the 2013 accumulated-subspace
algorithm, fixed-`q` full filtering, and leakage-forced occupied enrichment are
different realizations. `VARIANT_LEDGER.md` keeps their evidence separate.

The recommended profiles are:

```text
factorization dominated, DH available
    shallow higher moments + hybrid Anderson/inexact response Newton

factorization dominated, reduced memory available
    pre-extraction growing/windowed moments + a bounded inner solve

cheap factors or no DH
    shallow higher moments + Anderson/DIIS

small known virtual-response dimension q
    leakage-forced occupied enrichment, with direct residual verification

non-Hermitian
    dual moment realization + common gauge + oblique-projector closure

large occupation, density only, or closing gap
    finite-temperature pole expansion + selected inversion + μ iteration

simultaneous θ and z nonlinearity
    persistent invariant pair + corrected moments + closure acceleration
```

`cache_native_corrected_moments` now tests the integration between the memory
and response profiles. One frozen factor cache supplies the moment block and,
when scheduled, one full closure-response correction before reduced iteration.
The integration creates a measurable factor/RHS envelope but does not establish
one universal policy: accumulated moments remain the sparse timing leader on
the current cheap-Hamiltonian control.

Level-shifted and gauge-aligned orbital SCF are valid fallbacks but are not the
default on the tested density problem.  Deep power moments are also not a
default: use the shallowest depth that meets the RHS-width budget and partition
before the Hankel singular gap collapses.

## Competing Methods

Several important methods do not define missing FEAST variants:

- Nonlinear inverse iteration and the J-method are strong single-state or
  shift-selective solvers.  They are recovered conceptually at one state and
  one shift, but they do not replace a many-state contour realization.
- Riemannian gradient/Newton and normalized gradient flow are attractive when
  the NEPv is the stationarity condition of a known energy.  They offer global
  safeguards for that variational subclass but do not cover arbitrary interior
  contours or general holomorphic `T(θ,z)`.
- Full coupled Newton on `(X,S,θ)` is a legitimate local solver.  Response
  Newton is its closure Schur complement in the linear-pencil density case and
  preserves the contour factorization architecture with fewer gauge variables.
- Kerker, dielectric, and other physical preconditioners compose with density
  mixing, Anderson, or Krylov response.  They change the closure linear algebra,
  not the moment algorithm.
- Polynomial purification, Fermi-operator expansion, and stochastic diagonal
  estimation are density-matrix backends.  They become preferable when
  invariant vectors are not required.

## Evidence Boundary

The experiment establishes the algorithm shape, not a production library or a
universal convergence theorem.  Evidence now covers:

- exact linear FEAST and zero-quadratic reductions;
- depth-one Beyn and higher-moment SS/Hankel reductions in `fused_nlfeast`;
- direct Hermitian density/projector iteration and Gross–Pitaevskii rank one;
- Anderson, mixed, inexact/tight response, CG, level-shift, orbital, and reduced
  outer branches;
- a single-cache dual non-Hermitian closure;
- coupled right/left accumulated moments with a common overlap gauge;
- simultaneous spectral and eigenvector nonlinearity, including accumulated
  corrected-moment spaces and an exact zero-quadratic reduction;
- a nonlocal full-projector closure that cannot be represented by density;
- full-operator count mismatch detection and oracle-free moving occupied charts;
- sparse shifted solves and a measured factorization/RHS cost crossover.

The remaining work is deliberately separated:

- analytic local contraction and finite-realization perturbation theorems;
- a scale-aware inner forcing law and principled history transport/restart rule;
- adaptive memory/restart policies beyond leakage-contraction growth;
- large-occupation and multidimensional sparse cost evidence;
- certified phase resolution for fixed non-Hermitian contours;
- a sparse inertia backend and production symbolic-factor reuse;
- finite-temperature selected-inversion integration;
- product-Grassmann algorithms for truly orbital-specific operators;
- public FEASTSolver/FEAST.rs API design and tolerance ownership.

These are theorem, backend, or broader-problem projects. Across local density,
nonlocal projector, oblique projector, and simultaneous `θ,z` controls, none
currently points to a different core corrected moment update. The meaningful
variation remains scheduling, memory, reduced solve, and physical-state
representation.
