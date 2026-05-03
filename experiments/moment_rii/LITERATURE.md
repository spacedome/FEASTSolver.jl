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
- Gopalakrishnan, Parker, and Vandenberge's optical-fiber polynomial FEAST work
  is a useful application-side control. It applies contour integration directly
  to a polynomial eigenproblem from a frequency-dependent PML model while
  avoiding the large inversions of a full linearization. This aligns with our
  polynomial-native invariant-pair experiments and companion-pencil checks:
  polynomial contour methods should be compared against linearizations, but not
  forced to expose the linearized state as the public algorithmic state.
  Source: https://arxiv.org/abs/2104.13527
- Li and Polizzi's 2024 GW quasiparticle paper confirms that NLFEAST-style
  nonlinear contour methods are still being specialized for domain NEPs with
  expensive analytic evaluations. This is a practical reminder that reduced
  extraction and residual updates must minimize full operator solves and make
  chart-local scaling explicit; domain-specific nonlinearities can have large
  dynamic range and nontrivial complex-contour evaluation rules.
  Source: https://arxiv.org/abs/2409.06119

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
- AAA/CORK-style automatic rational approximation is another rational-coordinate
  baseline. Lietaert, Pérez, Vandereycken, and Meerbergen use AAA to build a
  rational NEP approximation and embed it in a compact rational Krylov
  linearization. This strengthens the design requirement that our chart layer
  support rational coordinates chosen from approximation quality, not only
  monomial moments or fixed contour scaling.
  Source: https://arxiv.org/abs/1801.08622
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
- Liu, Roman, and Shao use infinite GMRES inside contour-integral nonlinear
  eigensolvers to avoid expensive factorizations at every quadrature node. This
  is relevant to sparse/distributed future work and inner linear solves, but it
  does not replace the outer realization/update policy. Our residual Laurent
  update should be compatible with such an inner solve layer because both are
  organized around contour-node linear systems.
  Source: https://doi.org/10.1137/24M1650375
- Sakurai, Futamura, and Tadano discuss parameter estimation and implementation
  for contour-integral eigensolvers, including parallel scalability and the
  practical importance of method parameters. This reinforces that chart radius,
  support threshold, contour-node count, and rank/count estimates are algorithm
  parameters that need diagnostics, not constants hidden in the implementation.
  Source: https://doi.org/10.1260/1748-3018.7.3.249
- Jarlebring, Koskela, and Mele reinterpret residual inverse iteration and
  related NEP methods as quasi-Newton methods through Keldysh theory. This
  supports our boundary that scalar RII is the `K=1` local correction model,
  while the moment version needs a realization or invariant-pair state rather
  than scalar diagonal updates.
  Source: https://doi.org/10.1007/s11075-017-0438-2
- Guo, Huang, and Lin use an extended argument principle and contour integrals
  to determine algebraic multiplicities of NEP eigenvalues in a region. This is
  directly aligned with our distinction between geometric retained support and
  algebraic contour count: local cluster counts are a principled lower-rung
  multiplicity diagnostic, not an ad-hoc duplicate-root patch.
  Source: https://doi.org/10.1016/j.amc.2015.09.024

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
- Separate outer update geometry from inner linear-solve acceleration. NLEIGS,
  rational Krylov, and infinite-GMRES contour work inform sparse/large-scale
  implementation, but the missing moment-NLFEAST piece remains the outer
  charted realization update and retention policy.
- Keep algebraic count separate from geometric support. The extended argument
  principle literature supports using contour counts to determine algebraic
  multiplicity, while retained values should remain geometric clusters unless a
  higher-rung Jordan/derivative state is explicitly needed.

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
6. Infinite-GMRES / rational Krylov contour implementations, but only after the
   dense outer algorithm is stable; these primarily affect solve cost and
   memory, not the moment update geometry.
7. Argument-principle multiplicity papers, to decide when local cluster counts
   are enough and when a derivative/Jordan retained state is justified.

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

### Focused Online Recheck: No Hidden Residual-Moment Method Found

A fresh targeted search for "higher moments residual inverse iteration",
"two-sided nonlinear FEAST", and "residual inverse iteration contour moments"
mostly returns the same nearby references already in this file. Neumaier's
residual inverse iteration is the scalar local correction model used by
NLFEAST, while Brennan/Embree/Gugercin support Loewner as a realization and
interpolation layer. The search did not reveal a named method that performs the
specific residual Laurent-moment repair of left/right physical trial spaces
before reduced NEP re-extraction.

Sources checked:

- A. Neumaier, "Residual Inverse Iteration for the Nonlinear Eigenvalue
  Problem", SIAM Journal on Numerical Analysis 22(5), 1985, 914--923.
  DOI: https://doi.org/10.1137/0722055.
- M. C. Brennan, M. Embree, and S. Gugercin, "Contour Integral Methods for
  Nonlinear Eigenvalue Problems: A Systems Theoretic Approach", SIAM Review
  65(2), 2023, 439--470.
  DOI: https://doi.org/10.1137/20M1389303.

### May 2026 Recheck

A targeted search on 2026-05-03 again found the same immediate neighborhood:
Brennan--Embree--Gugercin for systems/Loewner contour realization, Neumaier for
scalar RII, NLFEAST for contour-generalized RII, Beyn/SS-style contour
extraction, and Barkatou--Boito--Segura Ugalde for contour-computed invariant
pairs. The search did not reveal a method that combines these into the specific
algorithmic loop we are testing:

```text
left/right contour realization
  -> reduced NEP extraction
  -> residual Laurent moment repair of the physical trial/test spaces
  -> re-extraction in local charts
```

The closest conceptual matches remain contour invariant-pair extraction plus
Newton refinement, and systems-theoretic Loewner realization. Those support the
state space and extractor story, but they do not appear to replace the
residual-Laurent update or the chart/count policy.

### May 2026 Targeted Residual-Update Query

A narrower follow-up search used phrases around "residual inverse iteration",
"moments", "finite realization", "Loewner", and "contour nonlinear eigenvalue".
The result set again clustered around:

- Neumaier's scalar residual inverse iteration:
  <https://doi.org/10.1137/0722055>.
- NLFEAST as contour-generalized residual inverse iteration:
  <https://doi.org/10.1016/j.jocs.2018.05.006>.
- Brennan--Embree--Gugercin's systems/Loewner realization framework:
  <https://doi.org/10.1137/20M1389303>.
- Contour-computed invariant pairs and Newton refinement:
  <https://doi.org/10.1016/j.tcs.2017.03.024>.
- Sakurai/SS projection and moment extraction:
  <https://doi.org/10.14495/jsiaml.5.41> and
  <https://doi.org/10.14495/jsiaml.1.52>.

It also rediscovers the NLFEAST-Beyn hybrid preprint from this research line,
which explicitly frames the missing problem as applying residual inverse
iteration to contour moments. That is useful provenance, but it is not an
external algorithm that resolves the higher-moment state/update problem.

Current conclusion: the literature supports the decomposition into contour
realization, reduced extraction, and local residual/Newton correction, but this
search still did not find a published method that iterates a finite left/right
Hankel or Loewner realization by the residual-Laurent physical-space update
tested here. The novelty risk is therefore not "someone already named this exact
loop" based on the targeted search; the larger risk is that an adjacent
invariant-pair or model-reduction method can be specialized to the same formula
with different language.

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
- `run_loewner_interpolation_sweep` varies Loewner interpolation radius and
  phase on the exponential many-root global chart, using the same reduced basis
  and residual Laurent update, so interpolation-point sensitivity is visible as
  a first-class diagnostic. It also clusters residual-small candidates across
  interpolation layouts and reports support thresholds, giving a same-chart
  analogue of the local-chart support diagnostic.
- `run_global_loewner_interior_artifact_diagnostic` focuses that sweep on the
  current in-target spurious case: a single updated Loewner layout returns a
  residual-small candidate inside the requested contour but displaced beyond the
  strict matching tolerance, while cross-layout support removes it.
- `run_exponential_local_chart_loewner_layout_sweep` runs the same
  interpolation-layout question inside supervised local exponential charts. It
  tests whether local charting makes the Loewner extractor insensitive to the
  outside interpolation layout, or whether layout support is still needed after
  chart support merging.
- `run_exponential_grid_chart_loewner_spacing_sweep` removes the supervised
  root-centered chart assumption for the same exponential problem. It uses a
  Cartesian grid cover, residual-scored local-chart selection, and Loewner
  counted extraction, then reports how grid spacing changes union and
  support-threshold recovery.
- `run_exponential_adaptive_grid_loewner_refinement` starts from a coarse grid
  and adds computed weak-support residual-small candidates as new chart centers.
  It is the first prototype of adaptive chart placement for the Loewner/count
  local-chart path.
- `run_triangular_adaptive_grid_loewner_refinement` applies the same intrinsic
  weak-support refinement rule to the nonnormal triangular analytic control. It
  separates two chart-cover tasks: adding centers restores missing support for
  true roots, and final target-contour membership removes local-chart roots that
  are valid nearby candidates but outside the requested solve domain.
- `run_three_function_adaptive_grid_loewner_refinement` applies target-domain
  weak-support refinement to the larger radius-20 `sin/cos/sin(z)-0.3` analytic
  control. It tests whether the same chart policy repairs a case where the
  initial grid misses target roots outright, not just support evidence.
- `run_three_function_adaptive_grid_loewner_layout_sweep` reruns that adaptive
  radius-20 three-function solve across several Loewner interpolation layouts
  and clusters the final globally retained candidates across layouts. It checks
  that the retained set is stable under the reduced Loewner interpolation circle,
  not only under chart support.
- `run_three_function_adaptive_grid_extractor_agreement` reruns the same
  adaptive radius-20 three-function solve with different reduced extractors,
  currently Loewner counted extraction and counted SS/Hankel extraction. It
  clusters final target-domain retained candidates across extractor families,
  checking that the retained set is not an artifact of one reduced realization
  algebra.
- `run_three_function_retention_score_diagnostic` collects the first retention
  scorecard for the same repaired radius-20 cover. It reports target-domain
  support thresholds, weak in-domain clusters, local argument-principle count
  errors, count deficits, maximum local residual, and optional cross-layout and
  cross-extractor agreement evidence.
- `run_three_function_automatic_retention_policy` is the first policy wrapper
  around that scorecard. It escalates exact support-2 retained sets that still
  have local count warnings to Loewner-layout and reduced-extractor agreement;
  after both agreements pass it returns `:accept_with_chart_warnings` rather
  than pretending the count warnings vanished.
- `run_global_loewner_artifact_retention_policy` applies the same idea to the
  focused global in-target Loewner artifact: single-layout residual-small values
  are marked unsafe, and the accepted retained set is the cross-layout
  support-2 cluster.
- `run_delay_count_driven_adaptive_refinement` removes the analytic root oracle
  from the count-driven path. The delay scalar NEP reports no exact roots to
  the harness; completion is decided only by support matching the full-operator
  argument-principle count.
- `run_multi_delay_count_driven_adaptive_refinement` checks the same no-oracle
  completion rule in a nonnormal dimension-three setting: three distinct delay
  components are coupled by an upper-triangular operator and retained by
  support/count evidence alone.
- `run_near_pole_rational_count_driven_adaptive_refinement` checks the same
  no-oracle rule for rational components with exterior poles close to the
  contour. The gap-`0.01` case is a clean regression; the gap-`0.005` probe is
  retained as a reminder that near-contour singularities stress
  argument-principle quadrature reliability. The intended behavior there is
  diagnosis, not forced recovery: the count-driven policy should reject the
  contour as unreliable and ask for a safer chart.
- `run_duplicate_delay_count_driven_adaptive_refinement` applies the same
  oracle-free condition to the algebraic multiplicity branch. Duplicate delay
  components with triangular nonnormal coupling are retained as three
  geometric clusters whose local contour counts sum to the algebraic count six.

Initial behavior:

- On scalar `sin(z)` with radius 20, counted Loewner recovers the same thirteen
  roots as counted SS to roundoff.
- On the rank-deficient three-function analytic radius-20 case, counted
  Loewner works as a prototype but does not yet beat counted SS after one
  residual Laurent update. Both are limited by the reduced basis quality; the
  Loewner singular values expose interpolation-point sensitivity.
- On the exponential many-root radius-10 global chart, the interpolation sweep
  shows that several outside-point layouts all recover the 18 roots, but a
  single residual-updated layout (`rho=1.3`, zero phase) produces one
  residual-small displacement/spurious match failure at the strict `1e-6`
  lattice tolerance. The focused in-target artifact diagnostic records this as
  one bad layout out of six: nearest-root distance about `1.2e-6`, residual
  about `6.9e-11`, and Loewner singular ratio about `3e-8`. Clustering
  residual-small values across Loewner layouts fixes the diagnostic: support
  `>=2` removes the one-off spurious candidate and recovers all 18 roots after
  the residual update. This supports a policy of trying multiple cheap Loewner
  layouts after the contour solves and using cross-layout support as evidence,
  analogous to cross-chart support.
- In supervised local exponential charts, Loewner is much less sensitive to the
  outside interpolation layout. With chart radii `1.2` and `2.0`, all tested
  layouts recover 18/18 roots after support merging. The raw union still carries
  layout-dependent residual-small extras, 20--22 candidates for the six tested
  layouts, but chart support `>=2` returns exactly 18 roots every time. Local
  chart quality and support merging are therefore the first-line robustness
  tools; Loewner layout support remains a useful diagnostic for oversized or
  weak charts.
- An unsupervised Cartesian grid cover now recovers the same radius-10
  exponential Loewner case without root-centered charts. With local radii
  `1.2` and `2.0`, spacing `2.4` uses 57 centers and the raw union already
  matches all 18 roots, but only 12 roots have support `>=2`; spacing `1.8`
  uses 97 centers and support `>=2` returns exactly the 18 roots; spacing `1.2`
  uses 221 centers and even support `>=3` retains all 18 roots. Thus support is
  meaningful only relative to cover density: too sparse a grid can make valid
  roots look weakly supported, while a denser grid gives the desired support
  evidence at higher solve cost.
- A first adaptive chart-refinement rung repairs this weak-support case with far
  fewer solves than the denser grid. Starting from the spacing-`2.4` cover, the
  prototype adds the computed residual-small clusters with support below two as
  new chart centers. It adds five centers, from 57 to 62, and changes support
  `>=2` from 12/18 roots to exactly 18/18. Validation still uses known roots,
  but the refinement rule itself uses only computed candidates and support.
- The adaptive rule was generalized from this clean exponential control to a
  harder nonnormal triangular analytic control at radius 6 with coupling 10. The
  spacing-`2.4` grid starts with 21 centers and a raw union matching all 11
  roots, but support `>=2` keeps only 5/11. Restricting refinement centers to
  weak-support candidates inside the requested target contour gives 26 centers
  and global support `>=2` for 10/11 after one round, then 28 centers and global
  support `>=2` for all 11 after two rounds. The unfiltered support-2 set still
  has 13 candidates, but the extras are roots recovered by local charts outside
  the requested global contour. Adding final global-contour membership to the
  support rule reduces the retained set to exactly 11/11. This gives a concrete
  retention rule for overlapping local charts: support is chart evidence, while
  final target-contour membership defines the solve domain; the same membership
  test should also guide which weak candidates deserve new chart centers.
- The same target-domain weak-support refinement repairs the larger radius-20
  three-function analytic control. The coarse spacing-`3.0` grid with local
  radii `1.5` and `2.4` starts with 137 centers, raw union 36/38, and global
  support `>=2` only 16/38. One refinement round adds 19 target-domain weak
  centers and finds all 38 in the raw union, but global support is still 36/38.
  A second round adds two more target-domain weak centers and gives exactly
  38/38 with final global support, while unfiltered support still contains two
  outside-domain extras. Repeating the full adaptive solve for Loewner radii
  `1.15`, `1.3`, and `1.6` gives exact 38/38 final global support for every
  layout, and cross-layout support `>=2` on the retained values also gives
  exactly 38/38. Rerunning the same adaptive chart policy with counted SS/Hankel
  instead of Loewner also succeeds: Loewner reaches 38/38 with 158 centers,
  counted SS reaches 38/38 with 159 centers, and cross-extractor support `>=2`
  returns exactly 38/38. This demonstrates that adaptive charting can repair a
  genuinely missing-root many-root analytic cover, not only low support on
  already-found roots, and that the repaired retained set is not tied to one
  Loewner layout or one reduced-extractor algebra.
- The first retention scorecard on this case separates evidence that had been
  conflated. Target-domain support `>=1` and `>=2` both retain exactly 38/38,
  while support `>=3` keeps only 30/38 because the cover is not uniformly triple
  overlapping. All in-domain clusters have support at least two after
  refinement, and the maximum local residual among good chart records is about
  `1.4e-13`, but three local charts still have argument-principle count deficits
  with count error above `1e-2`. Optional scorecard checks also confirm the
  already-established cross-layout and cross-extractor agreement. Thus local
  count consistency is a chart-quality warning, not an automatic root-pruning
  rule; it should feed refinement/splitting decisions together with support,
  residual, geometry, and extractor/layout agreement.
- The first automatic policy wrapper makes that escalation explicit. On the same
  radius-20 three-function cover, the initial policy decision is `:escalate`
  because exact support-2 retention coexists with local count warnings. It then
  runs the already-defined Loewner-layout and SS/Loewner reduced-extractor
  agreement checks. Since both pass, the final decision is
  `:accept_with_chart_warnings`, retaining 38/38 while preserving the warning
  for future chart split/shrink logic. The global in-target Loewner artifact
  exercises the opposite branch: single-layout residual-small values are marked
  unsafe, and the policy retains only cross-layout support-2 candidates, giving
  18/18 with no spurious in-target artifact.
- This confirms the implementation path is viable, but not yet a replacement
  for counted SS. The next useful work is to stress the retention rule on cases
  where residual-small extras are not merely outside-contour roots: combine
  chart-local argument-principle counts, Loewner/Hankel rank gaps, residual
  magnitudes, chart overlap geometry, and agreement across interpolation
  layouts.
