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

For now this should be treated as an experimental conjecture, not a proved
theorem. The current evidence supports the following narrower claim:

```text
When contour samples capture the correct finite local realization but the raw
realization coordinates produce large physical residuals, a reduced analytic
cleanup of Tred(lambda)=Y^H T(lambda) X and/or residual-Laurent physical-space
enrichment is the correct FEAST-compatible repair layer.
```

The Schrodinger domain-decomposition diagnostic is the strongest current
stress for this claim. The fused realization captures the correct rank and all
13 inside candidates, but the raw physical residuals are large; reduced `Tred`
cleanup recovers all 13 roots once contour quadrature is sufficiently resolved.
The same sweep also shows the boundary: if contour quadrature is still
underresolved, cleanup does not magically complete the solve.

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

## Lean Handoff Note

The current Lean theory suggests the next useful bridge is not another global
algorithm wrapper.  The experiment should be interpreted through a local
certificate of the form

```text
Corr o Rlin = P_packet + E
```

where `P_packet` is the projector/filter induced by the fused contour
realization and `E` is the realized residual-Laurent implementation or chart
defect.  Lean already proves the downstream residual-chart consequences once
this boundary is supplied: the exact correction leaves the packet complement,
while the perturbed correction leaves the packet complement minus `E`; under
FEAST-ratio filtering the persistent obstruction is the projector-visible
component `P_packet(E v)`.

For the next numerical pass, hard cases should therefore report whether the
fused cache and reduced extractor identify the same physical packet projector,
and whether the residual-Laurent update error is mostly projector-invisible.
If `P_packet(E v)` is large, treat that as a chart/transport or extraction
defect rather than as generic moment instability.  Keep the two moment roles
separate: positive powers build the finite realization, inverse-Laurent powers
repair physical residual spaces.

Since the last Lean pass, the most actionable new layer is the fused Laurent
schedule monitor.  It packages a run by outer iterate and asks the numerical
code to expose four quantities:

```text
Biterate[j] >= || Q_packet current_j ||
Brepair[j]  >= || Q_packet E(current_j) ||
innerBudget[j] >= |nu/mu|^innerSteps[j] * (Biterate[j] + Brepair[j])
outerBudget[j] >= contraction^j * || P_packet E(current_0) ||
```

The theorem then says the scheduled normalized repair update is bounded by
`innerBudget[j] + outerBudget[j]`.  In the exact-chart case
`P_packet E(current_j) = 0`, the outer visible-error budget drops out and the
inner FEAST-ratio budget alone controls the update.  This gives the next Julia
checkpoint: for each hard case, log an estimated `||P_packet E(current)||`,
`||P_packet correction(current)||`, `||Q_packet current||`,
`||Q_packet E(current)||`, `innerSteps`, and the observed scheduled repair
norm.  The useful question is whether chart fixes reduce the visible component
`P_packet E(current)` specifically; if they only reduce the total residual, the
Lean certificate does not yet explain the success.

A practical approximation is acceptable at this stage.  Use the fused positive
realization/extractor to build the packet projector used for diagnostics, keep
that projector fixed while comparing candidate corrections, and report the
positive-moment realization evidence separately from the inverse-Laurent repair
evidence.  Do not merge these into a single generic residual score, because the
Lean split is exactly what distinguishes a good chart with invisible repair
error from a bad chart whose correction has a packet-visible defect.

Julia checkpoint: `run_fused_schrodinger_dd_packet_defect_diagnostic` now
implements this as an experiment-level proxy. It builds a high-resolution
fused/refined reference packet projector for the Schrodinger/DD interface NEP,
then compares lower-node raw and `Tred`-refined packet projectors by splitting
the projector defect into reference-packet-visible and invisible parts. The
current run supports the Lean interpretation: at 128 nodes, reduced cleanup
drives the packet-visible defect to roundoff while recovering all 13 roots; at
64 nodes, the refined solve remains underresolved and the packet-visible defect
stays large.

The reusable Julia machinery for this handoff now lives in `packet_monitor.jl`.
Hard cases such as the Schrodinger/DD interface operator supply the packet
states and reference projectors; the monitor then provides the visible/invisible
split, correction-coordinate proxy, acceptance envelope, update-stage steering,
explicit schedule-monitor fields, and product/direct-sum merge policy. The
current schedule monitor records the Lean quantities `Biterate`, `Brepair`,
`innerSteps`, observed repair norm, and the observed outer visible-error budget.
It deliberately leaves the FEAST-ratio inner budget optional because estimating
that ratio requires a problem-specific inside/outside separation, not a generic
packet-projector norm.

The monitor also exposes the trial/test compatibility part of the Lean packet
boundary: right packet vectors should remain in the declared physical trial
space and left packet vectors should remain in the declared physical test
space. In the DD fused diagnostic these are reported as right/left membership
gaps against the moment realization bases.

The first policy use of this information is
`run_fused_schrodinger_dd_packet_policy_diagnostic`. It maps
`packet_visible_defect` to `increase_nodes_or_refine_chart`,
`packet_invisible_acceptance_gap` to `refine_extraction_or_acceptance`, and
`accepted_visible_removed` to `accept`. This is not yet a complete adaptive
parameter controller, but it demonstrates that the Lean-facing split is already
algorithmically actionable rather than only retrospective explanation.

After `ALGORITHM_HANDOFF.md`, the Julia monitor now also reports the
correction-coordinate identity and local visible-error contraction proxies:

```text
visible_contraction =
  ||P_packet(E_after current)|| / ||P_packet(E_before current)||

correction_coordinate_relative_error =
  ||P_packet(correction current) + P_packet(E_before current)||
    / ||P_packet(E_before current)||
```

On the Schrodinger/DD policy sweep, the 64-node case has contraction near one,
so the visible chart defect is not repaired; the 96- and 128-node cases
contract the visible component by many orders of magnitude, with the correction
coordinate matching the predicted negative visible-error bias. This is the
current numerical hook for the Lean `LocalVisibleErrorContractionHypothesis`.

The same handoff says extractor acceptance should not be raw rank thresholding.
The packet policy now therefore carries a small score-envelope certificate:
rank, accepted membership, packet-visible defect, visible-error contraction,
correction-coordinate agreement, and physical residual acceptance must all
pass before the action is `accept`. This keeps the Lean distinction between
realization evidence, repair evidence, and accepted membership visible in the
Julia experiment.

The update-stage steering is now explicit:

- visible defect or missing visible contraction means rebuild the packet/update
  geometry by increasing contour resolution or refining the chart;
- invisible packet defect with failed contraction/correction-coordinate checks
  means continue the local repair schedule;
- invisible packet defect with failed membership or residual checks means
  improve reduced extraction or acceptance;
- a full acceptance envelope accepts the local solve.

This is the main algorithmic use of the Lean geometry so far: not only
diagnosing failure, but selecting which update stage should receive more work.

The product/direct-sum Lean boundary is intentionally handled by an explicit
merge policy rather than by pretending a componentwise certificate supplies one
global schedule. The experimental rule is conservative: any component with a
packet-visible defect rebuilds the packet/update geometry; otherwise local
repair gaps outrank reduced extraction gaps, and only componentwise acceptance
accepts the merged packet.

The DD policy runner now also emits a steering trace. This is still a small
diagnostic, not a production controller, but it exercises the intended loop:
consume a candidate chart/node ladder, classify the packet defect, choose the
update stage that should receive more work, and stop only when the acceptance
envelope passes.

The chart-ladder diagnostic adds a necessary guard to that steering story:
changing the chart radius can make a low-node DD solve acceptable by changing
which spectral packet is being solved. The packet policy therefore treats
accepted changed-count charts as `chart_changes_packet`, not as acceptance for
the original target. For fixed-target steering, the target count/packet identity
has to stay part of the acceptance envelope.

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

The closest reduced-extraction theorem is Jia--Zheng's Rayleigh--Ritz/refined
Rayleigh--Ritz analysis for analytic NEPs. It can likely supply the
post-enrichment extraction side: given a sufficiently good trial space, it
relates subspace deviation, Ritz values, refined Ritz vectors, and residual
norms. It does not supply the enrichment side. The candidate proof therefore
has to be modular:

```text
residual-Laurent enrichment estimate
  + reduced NEP Ritz/refined-Ritz perturbation theory
  + chart/count acceptance policy
```

Only the first line remains genuinely missing.

## Lemma Decomposition

The local theorem can be decomposed into five lemmas. Four are already covered
by derivation, literature, or diagnostics; one is the unresolved mathematical
core.

### Lemma 1: Contour Moments Give A Finite Realization

For a semisimple target spectrum inside a safe chart, positive contour moments
computed from right/left probes factor as Markov parameters of a finite pole
realization:

```text
M_k = X_* S^k C + E_k,
N_k = B S^k Y_*' + F_k,
```

with `E_k,F_k` controlled by quadrature and rank truncation. This is the
standard Keldysh/SS/Beyn/Loewner layer. It is not novel.

Status: established by literature and pinned in the experiment by
`run_positive_moment_realization_recurrence_diagnostic`.

### Lemma 2: Residual Compression Is Coordinate-Invariant

Let `R_X` and `R_Y` be physical residual blocks from reduced Ritz data. If the
same residual subspaces are represented by `R_X C_X` and `R_Y C_Y` for
nonsingular `C_X,C_Y`, then after rank-revealing compression the
residual-Laurent enrichment spans are unchanged up to the compression
tolerance.

Status: elementary linear algebra; pinned numerically by
`run_residual_laurent_residual_coordinate_invariance_diagnostic`.

### Lemma 3: Exact Realizations Are Fixed Points

If `X,Y` contain the exact target right/left residue spaces and the reduced
Ritz data is exact, then `R_X=R_Y=0`, the compressed residual ranks are zero,
and residual-Laurent enrichment adds no new physical directions.

Status: elementary from the NEP residual equations; pinned by
`run_residual_laurent_realization_closure_diagnostic`.

### Lemma 4: The Linear Rung Reduces To FEAST

For `T(z)=zI-A`, the residual-inverse expression is algebraically identical to
the FEAST rational filter:

```text
x - (zI-A)^(-1)(lambda I-A)x = (z-lambda)(zI-A)^(-1)x.
```

Thus the contour-averaged correction is ordinary FEAST/RII on both right and
left Ritz vectors.

Status: exact identity; pinned by `run_linear_dual_rii_reduction_diagnostic`.

### Lemma 5: Local Analytic Enrichment Estimate

This is the missing lemma.

Assume a chart has captured the correct finite two-sided realization rank and
the reduced Petrov-Galerkin NEP produces Ritz data whose physical residual
blocks are nonzero but compressible. A desired estimate would show that one
residual-Laurent enrichment and re-extraction improves the physical Ritz
residuals under explicit local constants:

```text
residual_after <= C_chart * (realization_error + quadrature_error
                            + compression_error + higher_order_terms),
```

or at least that the enriched spaces contain the leading correction directions
needed by a reduced Ritz/refined-Ritz perturbation theorem.

The constants must depend on:

- separation of the target pole realization from exterior poles/singularities;
- conditioning of the two-sided reduced Petrov-Galerkin realization;
- quadrature accuracy for the contour-node solves;
- rank truncation and residual compression tolerances;
- nonnormal left/right eigenvector conditioning.

Status: not proved. The sparse diagonal and rank-deficient analytic diagnostics
show exactly the behavior the lemma should explain, but they are not proof.
This lemma is the current theorem gap.

### Minimal Sufficient Form Of Lemma 5

The theorem does not need to prove monotone convergence of the whole nonlinear
iteration. A weaker, more realistic statement would be enough:

```text
Given a safe chart whose positive moments identify the correct two-sided
realization rank, and given reduced Ritz data with compressed physical
residual bases U_X,U_Y, the residual-Laurent enrichment spaces contain,
up to quadrature/rank/compression errors, the leading local correction
directions that a JD/RII-style correction equation would add for the same
Ritz data.
```

Then the proof can be modular:

```text
contour realization accuracy
  -> residual-Laurent contains useful correction directions
  -> reduced RR/refined-RR perturbation improves re-extracted physical data
```

This weaker form is enough for the algorithm-family story because FEAST is
also a subspace iteration: the contour step supplies a better physical space,
and Rayleigh--Ritz/reduced extraction is responsible for producing new Ritz
data inside that space.

The minimum missing estimate is therefore a **range-inclusion/angle** statement
for correction directions, not a full nonlinear convergence theorem:

```text
dist(correction_space_JD/RII,
     span([X, contour_inverse_residual_moments(U_X)]))
  <= C_chart * (realization_error + quadrature_error + compression_error).
```

The left side should be interpreted side-by-side for the right and left
physical spaces. This also explains why a pure eigenvector-subspace angle
theorem was insufficient: the useful object is the local residual-correction
range inside a captured realization, not merely the angle between `X,Y` and
the exact residue spaces.

This minimal form still appears to be new or at least not covered by the
adjacent sources checked so far. Neumaier/Jarlebring-style RII theory supplies
the scalar correction language, Jacobi--Davidson supplies the residual
correction-equation analogy, Jia--Zheng supplies reduced extraction
perturbation after a good space exists, and SS/Beyn/Loewner supplies the
finite realization. None of those sources, as currently checked, proves this
contour-filtered compressed residual range-inclusion step.

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

Current literature search found reduced Rayleigh--Ritz perturbation theory and
Jacobi-Davidson correction-equation theory adjacent to this obstruction, but
not a theorem that directly covers the residual-Laurent enrichment step.
