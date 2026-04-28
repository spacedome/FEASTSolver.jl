# Literature Review Notes

This file tracks external and local references relevant to higher-moment
NLFEAST, SS/Beyn contour methods, reduced NEP extraction, and invariant-pair
updates. The goal is to prevent the experiment from drifting into an ad-hoc
design when established language or machinery already exists.

## Immediate Local Code: NEP-PACK

NEP-PACK is useful as a nearby reference, but it should not dictate our API.
Its implementation points to several concepts we should reuse deliberately.

- `method_block_SS.jl` implements block Sakurai-Sugiura by forming moments
  `Shat[:,:,0:2K-1]`, observing them with a left random probe `U'`, building
  reduced block Hankel matrices, truncating by SVD, then solving the small
  generalized pencil. This matches our current reduced counted-SS direction:
  the natural object is a two-sided compressed Hankel realization, not a large
  one-sided pile of physical vectors.
- `nep_transformations.jl` provides `shift_and_scale` and `mobius_transform`.
  This is strong evidence that our "chart" language is standard NEP machinery:
  local coordinates, scaling, and rational coordinate changes are first-class
  transformations, not implementation hacks.
- `compute_MM(nep, S, V)` is exactly the abstraction for invariant-pair
  residuals `T(V,S)`. We should keep this idea, but avoid depending on
  NEP-PACK's allocation-heavy interface for performance-critical code.
- `nep_deflation.jl`, `method_jd.jl`, and `method_nlar.jl` use invariant-pair
  deflation/restart machinery to avoid repeated convergence. That is useful as
  a fallback or diagnostic layer, but our experiments still suggest scalar
  deflation is not the primary geometry for moment-NLFEAST.

## Contour NEP Foundations

- Beyn's integral method is the canonical contour-integral NEP reference. It
  uses Keldysh's theorem to reduce the holomorphic NEP inside a contour to a
  finite-dimensional linear eigenproblem, and explicitly discusses the case
  where the number of eigenvalues exceeds the physical matrix dimension.
  This supports our low-dimensional many-root tests as central, not exotic.
  Source: https://doi.org/10.1016/j.laa.2011.03.030
- Sakurai-Sugiura / SS-RR methods are the direct moment-Hankel relatives.
  Yokota and Sakurai extend SS with Rayleigh-Ritz projection to NEPs by
  converting the contour region to a smaller problem. This is close to our
  "reduced NEP extraction" view.
  Source: https://doi.org/10.14495/jsiaml.5.41
- NLFEAST is explicitly described as a contour generalization of residual
  inverse iteration using fixed shifts and fixed subspace dimension. This is
  the correct interpretation for `K=1`; our higher-moment work should reduce
  to it, but not force higher moments into a diagonal scalar-RII state.
  Source: https://doi.org/10.1016/j.jocs.2018.05.006
- The existing NLFEAST-Beyn hybrid paper already states the open problem we are
  attacking: higher moments can be iterated by replacing the moments in the
  Beyn/SS extraction, but this requires deflating from `K*m` back to `m`, which
  blocks cases with more eigenvalues than physical dimension. Our current
  residual-Laurent-moment and dual reduced-NEP experiments are a direct attempt
  to solve that limitation without ad-hoc truncation.
  Source: https://doi.org/10.48550/arXiv.2007.03000

## Adjacent Ideas That Matter

- The 2023 SIAM Review systems-theoretic contour paper is probably the most
  important adjacent reference. It recasts contour methods as realization and
  rational-interpolation problems, and replaces standard Hankel pencils with
  Loewner pencils that interpolate at many points rather than effectively at
  infinity. This is very close to our chart/local-coordinate problem and should
  be read carefully before finalizing a moment API.
  Source: https://doi.org/10.1137/20M1389303
- NLEIGS builds a rational interpolant of the nonlinear operator and solves a
  structured companion-type linearization with rational Krylov. This is not a
  contour method, but it is the main competing answer to "use rational local
  coordinates instead of monomial moments." It should influence our reduced
  solver and benchmark interface.
  Source: https://doi.org/10.1137/130935045
- Invariant pairs are the established nonlinear analogue of invariant
  subspaces. Szyld and Xue analyze multiplicity, Jordan chains, perturbation,
  and two-sided block Rayleigh functionals for invariant pairs. This gives the
  right theoretical vocabulary for our `(X,S)` state, gauge choices, and
  multiplicity diagnostics.
  Source: https://doi.org/10.1093/imanum/drt026
- Van Barel and Kravanja reinterpret Beyn's algorithm through filter functions
  and Hankel tensor decompositions. This is a reminder that contour placement
  and filter function shape materially affect extraction quality; our local
  chart cover is consistent with that warning.
  Source: https://doi.org/10.1016/j.cam.2015.07.012

## Current Design Implications

- Do not finalize a public higher-moment API around eigenvalues of a small
  multiplication matrix `S` alone. Literature and experiments both point toward
  realization/reduced-NEP extraction as the natural layer.
- Keep the core state language as left/right physical bases plus a charted
  reduced NEP, with invariant-pair data as an internal realization diagnostic.
- Keep residual Laurent moments. They are the cleanest bridge we have found
  between FEAST RII and SS/Beyn moments, and they avoid exposing expanded
  Hankel columns as the iterative state.
- Treat charts as real mathematical objects: contour, coordinate map, scaling,
  moment basis, extraction method, and gauge. NEP-PACK's shift/scale/Mobius
  transformations and the systems/Loewner literature both support this.
- Treat deflation as an escalation rung, not the main algorithm. Deflation is
  established and useful, but our failures so far are better explained by chart
  quality, observability, nonnormality, and reduced extraction.

## Reading Priority Before API Work

1. Brennan, Embree, and Gugercin, contour methods as realization/Loewner
   pencils. This may give a cleaner replacement for monomial Hankel charts.
2. Beyn 2012, especially the larger-than-physical-dimension and multiplicity
   treatment.
3. Yokota-Sakurai SS-RR and block SS followups, especially reduced projection
   and parameter/rank selection.
4. Szyld-Xue invariant-pair theory, especially gauge/conditioning and
   two-sided block Rayleigh functionals.
5. NLEIGS rational interpolation, especially scaling and rational coordinate
   choices for target regions.

## Open Literature Questions

- Is there already a named "residual moment" or "Loewner residual update"
  equivalent to our Laurent-moment correction?
- Can Loewner pencils give a more stable reduced extractor than our current
  counted-SS/Hankel path while reusing exactly the same contour solves?
- Are there practical invariant-pair gauge/balancing algorithms that are more
  principled than our diagonal realization balancing?
- What rank/count estimators do SS-RR and modern contour papers recommend for
  near-contour roots and stiff analytic functions?
- Is there a published two-sided nonlinear FEAST variant beyond the linear
  dual-FEAST analogy, or is our left/right reduced-NEP correction genuinely a
  new synthesis?

## Detailed Read: Brennan, Embree, Gugercin 2023

Reference: Michael C. Brennan, Mark Embree, Serkan Gugercin,
"Contour Integral Methods for Nonlinear Eigenvalue Problems: A Systems
Theoretic Approach", SIAM Review 65(2), 2023.
Preprint: https://arxiv.org/abs/2012.14979.
Journal DOI: https://doi.org/10.1137/20M1389303.

### Core Interpretation

The paper gives a clean systems-theoretic interpretation of contour NEP
algorithms. Under the semisimple Keldysh decomposition, the pole part of
`T(z)^-1` in a contour can be written schematically as

```text
T(z)^-1 = V * inv(zI - Lambda) * W' + analytic_remainder(z),
```

where `Lambda` contains the eigenvalues inside the contour and `V`, `W`
contain right and left eigenvectors. Applying Cauchy's theorem to
`f(z) * T(z)^-1` removes the analytic remainder and leaves

```text
contour_integral f(z) * T(z)^-1 dz = V * f(Lambda) * W'.
```

The key move is to recognize `V * inv(zI - Lambda) * W'` as the transfer
function of a linear dynamical system. Contour algorithms are then realization
algorithms: they recover the poles of this transfer function, hence the NEP
eigenvalues inside the contour.

This matches our experiment almost exactly. Our "moment realization" language
is not merely an analogy; it is the standard systems-theoretic object exposed
by Keldysh plus contour integration.

### Hankel / SS As Ho-Kalman

The standard Beyn/SS moment path computes Markov parameters. With left/right
probes `L` and `R`, the moment blocks have the form

```text
M_k = L' * V * Lambda^k * W' * R.
```

Block Hankel matrices built from these moments factor as

```text
H_0 = observability * reachability
H_1 = observability * Lambda * reachability.
```

The paper explicitly identifies established contour-Hankel methods with the
Ho-Kalman realization algorithm. The usual SVD truncation of `H_0` is therefore
a realization-rank decision, not just a numerical cleanup. This is important
for us because it reframes "moment order" and "probe count" as
observability/reachability design parameters.

Consequences for our work:

- Our counted-SS extractor is a Ho-Kalman realization from contour-computed
  Markov parameters.
- The failure modes we saw as rank collapse, weak physical observability, and
  nonnormality are exactly realization conditioning failures.
- Gauge control is naturally a realization-coordinate choice.
- Underestimating the Hankel rank produces a reduced-order model of the pole
  transfer function, not a faithful invariant representation. That explains why
  reduced solves can return plausible but incomplete or spurious roots.

### One-Sided Data And Eigenvector Recovery

The paper distinguishes two-sided scalar/block samples from one-sided samples.
The basic two-sided sampled realization can expose the poles, but eigenvectors
are obscured by the probes. By collecting one-sided data, the Loewner/system
realization construction can recover an equivalent realization of the full
transfer function and therefore recover eigenvectors.

This is directly relevant to our dual reduced-NEP extraction. We should not
think of left/right probing as only a conditioning trick. It determines whether
we are realizing the sampled transfer function or enough of the full pole
transfer function to recover meaningful physical eigenvectors.

### Single-Point Loewner

The Hankel path expands the transfer function at infinity; equivalently, it
uses monomial/Markov moments. The paper then asks what happens if we expand
about a finite point outside the target domain. This produces a single-point
Loewner realization. The expensive work is unchanged: the contour quadrature
still solves systems with `T(z_q)` at the same quadrature nodes. The difference
is only in how the quadrature data is weighted and assembled.

Important points:

- The interpolation point affects numerical accuracy.
- Multiple candidate interpolation points can be tested cheaply after the
  contour solve data is available.
- In their delay example, a good single-point Loewner choice improves residuals
  versus Hankel.
- In their gun example, single-point Loewner can remain accurate in a case
  where Hankel fails to converge with the same final matrix size and smaller
  probing dimension.

This matches our suspicion that monomial moments are often the wrong chart.
Single-point Loewner is a principled finite-coordinate chart, not an ad-hoc
moment offset.

### Multi-Point Loewner

The strongest algorithmic result for us is the multi-point Loewner method. It
trades high-order samples at one point for first-order samples at several
points. The authors emphasize that this can recover linearly dependent
eigenvectors without relying on large powers. This is very close to our central
problem: higher moments are useful but high powers and large lifted stacks are
numerically fragile.

The multi-point construction uses left interpolation points, right
interpolation points, and associated tangential directions. The Loewner and
shifted-Loewner matrices are assembled from divided differences of tangential
samples of the pole transfer function. Those samples are again computed by
contour integrals, so once the quadrature-node solves are available, trying
many interpolation-point layouts is cheap.

Their gun experiment is especially relevant:

- Hankel and single-point Loewner can numerically under-reveal the realization
  rank when probing dimensions are small.
- Multi-point Loewner reveals the correct rank and converges where the others
  fail.
- The explanation is singular-value/rank visibility, not a different linear
  solve kernel.

For us, this is a direct candidate replacement for large monomial Hankel
moments in hard local charts.

### Matching Points And Derivative Data

The paper allows left and right interpolation points to coincide. In that case
the Loewner divided difference becomes a Hermite/tangential derivative sample,
and the derivative data can also be obtained by contour integration. This may
matter for local charts where symmetric interpolation layouts are natural, or
where repeated points give better conditioning.

We should not implement this first, but the design should not make it hard.

### Direct Rational Approximation Of `T(z)^-1`

The paper also describes a cheaper non-contour route: sample `T(z)^-1`
directly at selected interpolation points and build a Loewner rational
approximant whose poles approximate eigenvalues. This is not guaranteed as a
black-box eigensolver, but it can provide good initial estimates, contour
counts, interpolation points, or probing directions.

This suggests a possible escalation layer:

1. Use cheap Loewner interpolation samples to estimate where roots are.
2. Use those estimates to choose contours, interpolation points, and directions.
3. Run the contour method for validated recovery.

### Filter Functions

The paper separates two notions that are easy to conflate:

- Rational filter functions from quadrature design.
- Rational interpolation / Loewner realization.

Quadrature error acts like replacing the ideal contour projector with a rational
filter. The authors summarize Van Barel/Kravanja's filter view for Hankel and
extend the same idea formally to Loewner. For our implementation, this means
contour shape, quadrature nodes, and Loewner interpolation points are separate
knobs, but they interact through the same finite-precision rank diagnostics.

### Design Impact For This Repo

This paper changes the immediate design priority.

- We should add a Loewner extractor prototype before finalizing the
  higher-moment NLFEAST API.
- The extractor should reuse the same contour solve cache as Hankel/SS. The
  data layer should store enough right and left quadrature samples so Hankel,
  single-point Loewner, and multi-point Loewner can all be assembled without
  new solves.
- Our `Chart` abstraction should include interpolation points and tangential
  directions, not just contour, quadrature, moment basis, and scaling.
- Local chart selection should score Hankel and Loewner candidates by residual
  and rank visibility. The paper explicitly supports trying several Loewner
  interpolation points cheaply after quadrature data is computed.
- Multi-point Loewner is the most promising principled fix for cases where
  monomial Hankel moments need large `K`, bury central roots, or expose poor
  rank gaps.
- The dual left/right basis work is aligned with the paper. Left/right
  sampling is part of recovering the full transfer realization and the
  eigenvectors, not merely a dual-FEAST-inspired numerical embellishment.

### Concrete Next Experiment

Implement a prototype inside `experiments/moment_rii` that uses the existing
analytic local-chart quadrature data and compares:

1. counted Hankel/SS extraction;
2. single-point Loewner extraction with several interpolation points outside
   the local contour;
3. multi-point Loewner extraction with low-order samples around a larger circle
   enclosing the local contour.

The first target should be the hard analytic charts we already know:

- diagonal/similar `sin`, `cos`, `sin(z)-0.3`, `exp(z)-1` at radius 20;
- upper-triangular nonnormal versions with coupling `10`;
- a reduced version of the gun/sparse benchmark once sparse NLFEAST is back in
  scope.

Success criteria:

- better rank gaps than monomial Hankel for the same contour solve data;
- equal or fewer residual Laurent updates needed;
- no loss on the low-dimensional many-eigenvalue polynomial controls;
- clear diagnostics for when interpolation-point placement is poor.

### Prototype Status

Implemented first pass in `experiments/moment_rii/run.jl`:

- `contour_transfer_samples` computes Cauchy-filtered samples of the reduced
  pole transfer function at scaled interpolation points `alpha` outside the
  unit chart.
- `block_loewner_pencil` assembles block Loewner and shifted-Loewner pencils
  from full-block left/right samples.
- `reduced_analytic_loewner_extraction` mirrors the counted-SS reduced
  extractor: form a small Loewner realization, recover candidate roots, refine
  them against the reduced determinant, then validate with full right/left
  residuals.
- `extractor=:loewner` and `extractor=:loewner_counted` are accepted by
  `reduced_analytic_extraction`.
- `run_reduced_loewner_extractor_comparison` compares counted SS and counted
  Loewner on the same initial and residual-updated reduced bases.

Initial behavior:

- On scalar `sin(z)` with radius 20, counted Loewner recovers the same thirteen
  roots as counted SS to roundoff.
- On the rank-deficient three-function analytic radius-20 case, counted
  Loewner works as a prototype but does not yet beat counted SS after one
  residual Laurent update. Both are limited by the reduced basis quality; the
  Loewner singular values expose interpolation-point sensitivity.
- This confirms the implementation path is viable, but not yet a replacement
  for counted SS. The next useful work is interpolation-point selection and
  testing Loewner inside the local chart cover, where the paper suggests it
  should help most.
