# Master Contour Form

There are two independent nonlinearities which should not be conflated:

```text
spectral nonlinearity:     z ↦ T(θ,z),
eigenvector nonlinearity:  θ = D(X,S,Y).
```

The complete problem is the coupled invariant-pair system

```text
Tθ(X,S) = 0,
θ − D(X,S,Y) = 0.
```

For a representation `T(θ,z)=∑ⱼ Aⱼ(θ)fⱼ(z)`, the right invariant-pair
residual is

```text
E = Tθ(X,S) = ∑ⱼ Aⱼ(θ)Xfⱼ(S).
```

The equivalent Cauchy definition applies to a general holomorphic operator.
Given a current pair `(X,S)`, its corrected node response is

```text
Qθ(z;X,S) = [X−T(θ,z)⁻¹E](zI−S)⁻¹.
```

Moments of `Qθ`, followed by a finite realization, produce the next invariant
pair.  A closure map then produces the next physical nonlinear state:

```text
(θ,X,S) → corrected contour moments → (X⁺,S⁺) → D(X⁺,S⁺,Y⁺) = θ̂.
```

The unaccelerated master iteration sets `θ⁺=θ̂`.  Mixing, Anderson, and
response Newton alter only this last update.  They are not different spectral
algorithms.

For non-Hermitian problems the left response is retained as well.  The state
divided overlap `G` supplies the common gauge.  When `T(z)=zI−H`, it reduces to
the usual left/right overlap, so common-gauge biorthogonalization is exactly
the dual-FEAST coupling rather than an unrelated nonlinear construction.

Equivalently, the single coordinate-free nonlinear state is the oblique Riesz
projector

```text
P = X(YᴴX)⁻¹Yᴴ.
```

Separate right and left spaces are numerical representatives of this one
projector.  Their dual filtering and common gauge improve conditioning without
introducing a second physical closure variable.

## Exact Reductions

`linear FEAST`

Set `θ` absent and `T(z)=zB−A`.  Since

```text
E = BXS−AX,
Q(z;X,S) = (zB−A)⁻¹BX,
```

the corrected response is precisely the FEAST filter.  A depth-one wide probe
is ordinary block FEAST; left/right responses give dual FEAST.

`Beyn and NLFEAST-Beyn`

Set `θ` absent, allow holomorphic nonlinear `T(z)`, and use a depth-one
realization with a probe at least as wide as the local algebraic count.  The
zeroth and first corrected moments give the Beyn realization.  Repeating the
residual-corrected construction is NLFEAST-Beyn.

`SS and higher-moment NLFEAST`

Keep the same corrected transfer but use depth greater than one and a narrower
probe.  Block Hankel realization trades probe width for moment depth.  Thus SS
is a realization choice, not a different contour correction.

`Hermitian eigenvector-nonlinear FEAST`

Set

```text
T(ρ,z)=zI−H(ρ),
D(X)=diag(XXᴴ)/h.
```

The linear identity above removes the old `(X,S)` from the node response, so
one frozen-`ρ` contour realization evaluates the exact occupied projector up
to quadrature and realization error.  The resulting map

```text
ρ⁺ = diag(ΠΓ(H(ρ)))/h
```

is the coordinate-free core of the present experiment.  One occupied state is
the Gross–Pitaevskii case; several occupied states give the density/projector
NEPv.

`response Newton`

Apply Newton to the closure residual `F(ρ)=ρ−D(X(ρ))`.  The spectral derivative
is evaluated by differentiated contour solves.  This is the Schur complement
of a coupled Newton step after eliminating the invariant-pair response; it is
therefore a natural acceleration of the same master system.

`canonical projected NLFEAST`

Replace corrected full-space realization by: filter a larger trial space,
solve `YᴴT(θ,z)X` as a reduced nonlinear problem, then lift its solutions.
This is another consistent block factorization of the coupled system, but it
is not the same map.  It can be attractive when a small fixed trial space
captures both occupied states and their nonlinear response.  It deteriorates
when important virtual response lies outside that space.

## Natural Scheduling Choices

There is a unique natural core for a Hermitian unitarily invariant
`H(P)`: evaluate its occupied spectral projector and close the projector or
density.  The meaningful alternatives are schedules for solving the coupled
system:

```text
projector SCF       one frozen spectral realization, then closure
inner-converged SCF solve the frozen spectral problem accurately, then closure
Anderson/DIIS       multisecant acceleration of the closure residual
response Newton     eliminate spectral response and Newton-correct the closure
full coupled Newton solve simultaneously for pair, gauge, and physical state
```

For `T(θ,z)=zI−H(θ)`, inner-converged SCF and one accurate contour realization
coincide mathematically.  Repeated inner eigensolves are therefore redundant.
For a problem nonlinear in both `θ` and `z`, one corrected spectral step and
one closure step form a natural block Gauss–Seidel method, but inner-converged
and fully coupled Newton schedules become genuinely distinct.

`combined.jl` exercises that simultaneous case with a quadratic `T(ρ,z)`.
Residual-corrected higher moments update `(X,S)` and the density closure
directly.  Its zero-quadratic limit agrees with the dedicated projector FEAST
step to roundoff, while its nonzero form converges without a projected inner
NEP.  Details are in `COMBINED_NONLINEARITY.md`.

The full coupled Newton form is useful as a local reference or for strongly
coupled spectral/eigenvector nonlinearities.  It is not the default FEAST form:
it introduces gauge equations, large tangent variables, and a harder
preconditioner, while response Newton retains the contour solver as the
elimination mechanism.

## Representation Boundary

The core state should contain only the physical information required by
`T`:

```text
local density                 for local mean fields
low-rank projector `XXᴴ`      for unitarily invariant nonlocal fields
left/right projector          for non-Hermitian invariant subspaces
tuple `(xᵢxᵢᴴ)ᵢ`              for orbital-specific nonlinearities
```

Orbital matrices are efficient representatives, but their unitary gauge is
not physical.  A method whose update changes under `X→XU` is not a valid
projector algorithm unless the problem itself is orbital-specific.

Finite-temperature density-matrix iteration is a smooth sibling obtained by
replacing the sharp occupation projector with `fβ(H−μ)`.  It shares the same
closure and response structure, but it is not a moment invariant-pair method
unless orbitals are also requested.

## Claim Boundary

The controls now make growing or short-window pre-extraction moment spaces the
strongest derivative-free candidate by contour work. This ordering survives
local-density, nonlocal-projector, non-Hermitian dual, and simultaneous
spectral/state tests. Higher moments change solve width and the retained
response space; they do not change the exact fixed point. Adaptive thick
restart is the bounded-memory alternative when indefinite growth is
unacceptable. Response Newton remains faster on the current sparse timing and
uses fewer factor sets there, at substantially greater RHS width. Inner
forcing, memory, reduced cubic work, and the factorization/RHS crossover still
prevent selection of one universal schedule.

The master form also identifies real extensions rather than bookkeeping:

- general response Newton for simultaneous spectral and eigenvector
  nonlinearity `T(θ,z)`;
- non-Hermitian left/right response;
- orbital-specific product-Grassmann states;
- finite-temperature or changing-count density matrices.

Direct and accumulated simultaneous iteration, a coupled dual accumulated
closure, and a nonlocal full-projector closure now have focused implementations.
Their general response equations, orbital-specific states, and finite-temperature
selected-inversion backend should not be claimed solved from these controls.
