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

## Focused Pass: Iterating Realizations

The targeted follow-up was to look for the missing iteration theory: not
"another contour extractor", but a principled way to refine a finite
realization or reduced nonlinear subspace after moment extraction.

### What NEP-PACK Implements Locally

NEP-PACK has three nearby pieces, but none of them is exactly our higher-moment
FEAST iteration.

- `method_block_SS.jl` is extraction only. It computes moments, compresses a
  two-sided Hankel pencil, and returns Ritz values/vectors. This validates our
  counted-SS extractor, but it does not define an RII-like residual update.
- `method_rfi.jl` implements two-sided Rayleigh functional iteration for a
  single eigentriplet. This is the scalar local correction analogue of
  nonlinear RII: it alternates right/left inverse solves and updates the scalar
  root through `y' T(lambda) x = 0`. It is useful language, but it is not enough
  for many roots in a compact realization.
- `method_blocknewton.jl` implements Kressner's block Newton method for
  invariant pairs `(S, X)`. This is the closest local model for the tangent
  space of moment-NLFEAST. It explicitly normalizes the lifted block
  `[X; X*S; ...]`, transforms `S` to Schur form, solves a coupled correction
  system, then regauges the pair. This matches our gauge/chart observations.
- `nep_deflation.jl` uses invariant-pair deflation to prevent reconvergence.
  This supports deflation as a useful escalation rung, but not as the primary
  geometry of the moment update.

The local code therefore reinforces a clean separation:

```text
contour moments -> reduced realization/extraction -> residual update/refinement
```

SS/Beyn/Loewner live in the first two stages. Block Newton/RFI/deflation live
in the third stage or in diagnostics.

### Invariant-Pair Refinement Is The Closest Match

Kressner's block Newton paper is the main reference for local refinement of a
nonlinear invariant pair. The core reason it matters here is that it treats
several nonlinear eigenvalues as one object `(X, S)` instead of as unrelated
scalar Ritz pairs. That is exactly what higher moments require.

Source: D. Kressner, "A block Newton method for nonlinear eigenvalue problems",
Numerische Mathematik 114(2), 2009, 355--372.
DOI: https://doi.org/10.1007/s00211-009-0259-x.
Open ETH record: https://doi.org/10.3929/ethz-b-000019530.

Betcke and Kressner's invariant-pair refinement paper is the polynomial layer
we need below the fully analytic case. It studies perturbation, extraction from
linearizations, and refinement procedures directly on the polynomial
formulation. This is the right reference for comparing our polynomial-native
moment method against companion-pencil FEAST.

Source: T. Betcke and D. Kressner, "Perturbation, extraction and refinement of
invariant pairs for matrix polynomials", Linear Algebra and its Applications
435(3), 2011, 514--536.
DOI: https://doi.org/10.1016/j.laa.2010.06.029.
Preprint page: https://eprints.maths.manchester.ac.uk/1291/.

Szyld and Xue provide the broader nonlinear invariant-pair theory:
multiplicity, Jordan structure, conditioning, and two-sided block Rayleigh
functionals. This is the theoretical vocabulary for our gauges and for the
cases where scalar residuals and pair residuals disagree.

Source: D. B. Szyld and F. Xue, "Several properties of invariant pairs of
nonlinear algebraic eigenvalue problems", IMA Journal of Numerical Analysis
34(3), 2014, 921--954.
DOI: https://doi.org/10.1093/imanum/drt026.

### Contour Integral Invariant Pairs Connect SS To Newton

The contour-integral invariant-pair paper is unusually close to our exact
question. It adapts Sakurai-Sugiura moments to compute invariant pairs,
including some multiple-eigenvalue cases, and studies Newton-type refinement.
This is evidence that the natural continuation of SS moments is not scalar
deflation but invariant-pair refinement.

Source: M. Barkatou, P. Boito, and E. Segura Ugalde, "A contour integral
approach to the computation of invariant pairs", Theoretical Computer Science
681, 2017, 3--26.
DOI: https://doi.org/10.1016/j.tcs.2017.03.024.

This still does not appear to contain the specific FEAST-style residual
Laurent-moment update we have prototyped. The paper supports the same
mathematical state `(X, S)`, but our update is more FEAST-like: it repairs
left/right physical spaces through residual inverse contour moments before
resolving the reduced NEP.

### SS-RR Is Extraction, Not Iteration

Yokota and Sakurai extend SS with Rayleigh-Ritz projection to NEPs by reducing
the contour region to a smaller problem. This is exactly aligned with our
reduced NEP extraction layer. It does not answer how to iterate a bad
left/right physical basis, which is the role of our residual Laurent update.

Source: S. Yokota and T. Sakurai, "A projection method for nonlinear
eigenvalue problems using contour integrals", JSIAM Letters 5, 2013, 41--44.
DOI: https://doi.org/10.14495/jsiaml.5.41.

### Loewner/IRKA Is A Chart And Realization Tool

The Brennan-Embree-Gugercin review makes Loewner methods relevant, but not as
a direct FEAST iteration. Loewner pencils improve the realization/extraction
stage by changing interpolation points and rank visibility. IRKA-style ideas
suggest how interpolation points could be adapted. Neither is a residual
inverse update on left/right physical trial spaces.

The actionable conclusion is to keep Loewner as:

- an alternate reduced extractor when monomial Hankel/SS has poor singular
  value separation;
- a chart diagnostic for rank visibility;
- a future adaptive interpolation-point policy.

It should not replace the dual residual Laurent update as the core iteration
unless experiments show that Loewner-refined reduced spaces can repair bad
physical trial/test spaces without residual inverse solves.

### Deflation Is An Escalation Rung

Effenberger's successive-computation paper and NEP-PACK's deflation machinery
show that invariant-pair deflation is the established way to avoid
reconvergence in Newton/Jacobi-Davidson style solvers. That is useful once
individual roots or small clusters have converged. It is not the clean answer
to moment-NLFEAST's rank/observability failures, because those failures happen
before we have trustworthy scalar Ritz values to deflate.

Source: C. Effenberger, "Robust Successive Computation of Eigenpairs for
Nonlinear Eigenvalue Problems", SIAM Journal on Matrix Analysis and
Applications 34(3), 2013, 1231--1256.
DOI: https://doi.org/10.1137/120885644.

## Actionable Design After Focused Pass

The literature pass supports the direction of the current experiment rather
than replacing it. The most defensible algorithmic boundary is:

1. Build local right and left trial spaces from contour moments.
2. Solve a reduced Petrov-Galerkin NEP `Y' * T(lambda) * X`.
3. Use counted SS/Hankel, polynomial companion/QZ, or Loewner as interchangeable
   reduced extractors depending on the chart.
4. Apply the FEAST-style residual Laurent-moment update to expand/repair
   `X` and `Y` when the reduced extraction misses roots or returns spurious
   residual-small values.
5. Treat invariant-pair/block Newton as the local refinement model and possible
   refinement rung, especially for polynomial problems or small dense reduced
   NEPs.
6. Use invariant-pair deflation only after roots/clusters are reliable enough
   that reconvergence is the actual problem.

This clarifies the likely "generalized NLFEAST iteration" answer:

```text
generalized moment-NLFEAST = local dual contour realization
                           + reduced NEP extraction
                           + residual Laurent correction of left/right spaces
                           + chart/rank/gauge diagnostics
```

For `K=1` and diagonal states this reduces toward ordinary NLFEAST/RFI. For
linear problems it reduces to FEAST/dual-FEAST on the Petrov-Galerkin reduced
problem. For higher moments it avoids forcing a large Hankel realization back
into a diagonal scalar-RII form.

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
