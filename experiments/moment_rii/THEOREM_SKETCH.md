# Local Theorem Sketch

This note is the current target for a rigorous statement. It is not a finished
proof. Its purpose is to turn the experiment into a small set of mathematical
claims that can be proved, falsified, or compared against the literature.

## Objects

Let `T(z)` be regular and analytic near a contour `Gamma`, with no eigenvalues
or operator singularities on `Gamma`. Work in a circular chart

```text
z = c + r*zeta,        lambda = c + r*alpha.
```

Assume the target eigenvalues in the chart are semisimple for the first theorem
rung. Keldysh gives a pole realization of the local resolvent:

```text
T(z)^(-1) = X_* (zI - Lambda)^(-1) C_* + analytic remainder,
```

or, after applying left/right probes and chart coordinates,

```text
M_k = integral_Gamma mu(z)^k T(z)^(-1) V dz
    = X_* S_mu^k C
```

up to quadrature and finite-precision error.

The algorithmic state is not the moment stack. It is:

```text
right physical space X,
left physical space Y,
reduced Petrov-Galerkin NEP Y' T(lambda) X,
reduced extractor E,
residual inverse-Laurent enrichment U -> Q(U).
```

## Candidate Local Theorem

Under a local chart where:

- the positive contour moments identify the target transfer realization to
  numerical rank `p`;
- `X` and `Y` are full-rank physical trial/test spaces whose reduced
  Petrov-Galerkin NEP contains a stable `p`-dimensional realization of the
  target pole data;
- the reduced extractor returns Ritz data with residual blocks `R_X`, `R_Y`
  whose compressed bases span the leading physical residual directions;
- quadrature resolves both the positive realization moments and the
  inverse-Laurent residual enrichment moments;

then one residual-Laurent enrichment step

```text
X_+ = orth([X, Q_X(U_X)]),
Y_+ = orth([Y, Q_Y(U_Y)])
```

is invariant under reduced-coordinate changes of the Ritz/residual data and is
the FEAST-style physical-space correction compatible with the lower rungs:

- it vanishes on an exact two-sided realization;
- it reduces to the FEAST/RII contour-filter identity for `T(z)=zI-A`;
- it reduces to canonical NLFEAST in the `K=1`, diagonal scalar-state setting;
- it respects the SS/Beyn/Loewner realization layer by re-extracting rather
  than persisting expanded Hankel columns.

The theorem should not promise a global convergence factor from subspace angle
alone. The sparse diagonal diagnostic shows the extracted right/left subspaces
can already have projection gaps near roundoff while physical residuals are
still too large. The useful local quantity is the quality of the reduced
Ritz/eigenvector data inside the captured realization, measured by physical
right/left residuals and Petrov-Galerkin consistency.

The current controlled correction target is therefore:

```text
if the realization rank/count is correct but extracted physical residuals are
too large, residual-Laurent enrichment should reduce the physical residuals
after re-extraction, even when subspace angle was already a weak diagnostic.
```

The sparse diagonal pipeline pins this lower-rung behavior: the extracted
right/left subspace projection gaps are near roundoff before the update, but
the physical residuals improve by more than `1e4` after one residual-Laurent
enrichment and re-extraction.

The rank-deficient analytic compression diagnostic pins the same behavior in
the nonlinear setting: initial extraction has residuals of order one and no
accepted roots, while one compressed residual-Laurent enrichment improves the
residual scale by more than `1e8` and recovers the full target set. This is the
main numerical clue for the analytic local correction estimate.

## Why This Is Not Just Block Newton

Invariant-pair or block Newton methods are still the right local refinement
language once a small reliable reduced model is available. They are not the
moment update itself.

The experiment separates the roles as follows:

- block Newton acts inside an already chosen reduced realization or invariant
  pair;
- residual-Laurent enrichment changes the physical trial/test spaces before
  re-extraction;
- chart policy decides whether the local realization is trustworthy enough to
  refine, split, or reject.

The `run_analytic_block_newton_boundary_diagnostic` control records this
boundary. On a small, well-localized chart, reduced block Newton can improve
local data. On a large many-root analytic chart, it is not a substitute for
chart policy or residual-Laurent physical-space repair. Thus the remaining
theorem should not be stated as "apply block Newton to the moment matrix." It
should explain why the residual-Laurent update is a FEAST-style outer
correction that supplies better physical spaces for the next reduced
realization.

## Proof Skeleton

1. **Realization layer.** Use Keldysh plus contour integration to show positive
   moments are Markov parameters of the local transfer realization. In the
   linear diagonal control, this is pinned numerically by
   `M_k = X*S^(k-1)*C` and by outside-contour leakage near roundoff.

2. **Linear FEAST convergence language.** In the linear case, view the contour
   operator as a rational filter approximating a spectral projector, followed
   by Rayleigh--Ritz. This is the established FEAST convergence framing. The
   residual identity below explains why the residual-inverse update is the same
   filter in different algebra.

3. **Coordinate invariance.** Show that replacing residual blocks
   `R_X, R_Y` by `R_X C_X, R_Y C_Y`, for nonsingular `C_X, C_Y`, leaves the
   enriched spaces unchanged after residual-subspace compression. This is
   pinned by the residual-coordinate invariance diagnostic.

4. **Closure.** If `X,Y` contain the exact target right/left residue spaces and
   the reduced Ritz data is exact, then `R_X=R_Y=0`; the enrichment adds no
   physical directions. This is pinned by the exact-realization closure
   diagnostic.

5. **Linear reduction.** For `T(z)=zI-A`, use

   ```text
   x - (zI-A)^(-1)(lambda I-A)x = (z-lambda)(zI-A)^(-1)x
   ```

   so scalar residual inverse iteration and the FEAST contour filter are the
   same object. This is pinned by the dual linear RII diagnostic.

6. **Nonlinear local correction.** For analytic `T`, use the local pole
   realization to interpret the residual inverse-Laurent enrichment as a
   correction of physical input/output maps for the reduced realization. This
   is the remaining proof gap. The proof must involve the captured
   realization, reduced residual blocks, and quadrature error. It cannot be a
   denominator-only Laurent tail bound: that path is explicitly rejected by the
   scalar truncation boundary diagnostic.

7. **Re-extraction.** After enrichment, the algorithm returns to the
   realization layer and solves the reduced Petrov-Galerkin NEP again. This is
   why the persistent state remains `X,Y` plus extractor policy, not expanded
   residual or Hankel columns.

The nonlinear correction proof should likely borrow quasi-Newton/RII language:
Neumaier gives scalar nonlinear RII convergence, and Jarlebring--Koskela--Mele
connect RII-type NEP methods to quasi-Newton methods through Keldysh theory.
The missing step is lifting that scalar local-correction language to a
compressed two-sided finite realization without choosing scalar expanded
Hankel columns as the persistent state.

## Non-Claims

- This is not a black-box global convergence theorem for arbitrary analytic
  NEPs.
- It does not guarantee safety for contours near poles or eigenvalues.
- It does not replace chart selection, count diagnostics, extractor agreement,
  or multiplicity handling.
- It does not claim that more inverse-Laurent moments always improve a bad
  chart.
- It does not claim subspace angle alone controls success.
- It does not claim block Newton on a reduced realization is the outer
  moment-update mechanism.

## Current Evidence Map

- Positive realization recurrence:
  `run_positive_moment_realization_recurrence_diagnostic`.
- Residual-coordinate invariance:
  `run_residual_laurent_residual_coordinate_invariance_diagnostic`.
- Exact closure:
  `run_residual_laurent_realization_closure_diagnostic`.
- Scalar truncation boundary:
  `run_residual_laurent_scalar_truncation_boundary_diagnostic`.
- Linear FEAST/RII identity:
  `run_linear_dual_rii_reduction_diagnostic`.
- Canonical NLFEAST limit:
  `run_canonical_nlfeast_limit_diagnostic`.
- Nonnormal repair ladder:
  `run_residual_laurent_update_ladder_diagnostic`.

The unresolved mathematical task is step 5: a clean local correction estimate
for analytic `T` stated in terms of the finite transfer realization and reduced
physical residuals.

## Current Stopping Boundary

At this point the experiment has a coherent candidate algorithm and has ruled
out three tempting but wrong proof simplifications:

- finite denominator-only Laurent truncation;
- pure subspace-angle improvement;
- reduced block Newton as the outer moment update.

The remaining gap is genuinely mathematical: prove a local correction estimate
for analytic `T` where a captured two-sided finite realization has inaccurate
physical Ritz residuals, and show that compressed inverse-Laurent residual
enrichment improves the re-extracted physical Ritz data under explicit local
conditioning/quadrature/rank assumptions.

If that estimate cannot be proved, the algorithm should be reported as a
strong empirical FEAST-family synthesis rather than a solved general
moment-NLFEAST theory.

## Closest Known Proof Language

The closest adjacent proof language now appears to be Jacobi-Davidson and
residual inverse iteration, not block Newton alone.

In JD-style NEP methods, the outer loop:

```text
extracts Ritz data from a projected NEP,
forms a physical residual,
solves a correction equation,
expands the search space,
re-extracts.
```

Moment-NLFEAST has the same high-level correction rhythm but with FEAST
geometry:

```text
extract Ritz data from a two-sided contour realization,
compress physical right/left residual subspaces,
apply contour-averaged inverse solves to those residual subspaces,
expand both physical trial/test spaces,
re-extract.
```

This suggests a proof route:

1. Use positive contour moments to justify the reduced realization.
2. Use JD/RII correction-equation theory as the local residual-correction
   analogue.
3. Use the linear FEAST/RII identity to show that contour-averaged correction
   is the correct FEAST-family replacement for one fixed-shift correction.
4. Prove that the compressed two-sided residual subspace gives the same
   correction space under residual-coordinate changes.

The theorem should therefore be framed as a **FEAST-filtered block correction
equation for a finite contour realization**. That is currently the most concise
description of the candidate algorithm.

## Formal Obstruction

The remaining proof is not blocked by implementation details. It is blocked at
one specific mathematical step.

For one approximate Ritz pair `(lambda, x)`, scalar nonlinear RII studies an
update of the form

```text
x_+ = x - T(sigma)^(-1) T(lambda) x
```

or a variable-shift variant. For linear FEAST, replacing one shift by a contour
average is harmless because the identity

```text
x - (zI-A)^(-1)(lambda I-A)x = (z-lambda)(zI-A)^(-1)x
```

turns the residual correction exactly into a rational spectral filter.

For analytic `T`, there is no matching identity:

```text
x - T(z)^(-1) T(lambda)x
```

does not factor by `(z-lambda)` times a clean resolvent action except in a
local first-order model. Keldysh gives the pole structure of `T(z)^(-1)`, but
the residual term `T(lambda)x` is produced by the reduced Petrov-Galerkin NEP
and depends on the current left/right realization, extractor, and Ritz
coordinate. Therefore a proof must control all of the following at once:

- perturbation of the reduced nonlinear Ritz data from the exact local
  transfer realization;
- physical right/left residual compression error;
- quadrature error for applying `T(z)^(-1)` and `T(z)^(-H)` to those residual
  subspaces;
- the effect of re-extracting after the physical spaces are enriched;
- nonnormal left/right conditioning of the reduced Petrov-Galerkin NEP.

The diagnostics rule out simpler substitutes:

- denominator-only Laurent truncation fails because `T(z)^(-1)` has interior
  poles;
- subspace angle can already be near roundoff while physical residuals are bad;
- block Newton refines a chosen reduced realization but does not supply the
  outer FEAST-style physical-space repair;
- scalar expanded RII chooses a bad coordinate gauge for many-root moment
  charts and can fail while the compressed residual subspace update succeeds.

This is the precise place where a new proof or a known theorem is required.
The candidate algorithm is no longer vague; what is missing is a perturbation
and correction estimate for **contour-filtered compressed residual corrections
of a two-sided finite NEP realization**.
