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
