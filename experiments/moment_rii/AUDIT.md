# Moment-NLFEAST Research Audit

This audit maps the current goal to concrete experiment artifacts. It is meant
to prevent treating passing tests or a plausible algorithm sketch as proof that
the research objective is complete.

## Objective Restated

Find a decisive, elegant answer to the NLFEAST higher-moment update problem, or
identify why the current path is stuck. The answer should explain the algorithm
family connecting:

- FEAST and dual FEAST;
- SS-FEAST;
- Beyn/SS contour realization methods;
- canonical NLFEAST;
- moment-expanded NLFEAST.

The work should stay in `experiments/moment_rii` until the design is stable,
and claims need both theoretical basis and numerical evidence.

## Prompt-To-Artifact Checklist

| Requirement | Current Artifact | Evidence | Status |
| --- | --- | --- | --- |
| Keep work contained to the experiment | `experiments/moment_rii/run.jl`, `pipeline.jl`, `policy.jl`, `experiment_matrix.jl`, `ALGORITHM.md`, `IMPLEMENTATION.md` | No public API promotion; new algorithm objects are experiment-layer only. | Satisfied for current work |
| Explain unified algorithm family | `ALGORITHM.md`, `DERIVATION.md` | Candidate formula: local dual contour realization + reduced Petrov-Galerkin extraction + residual Laurent repair + chart policy. `ALGORITHM.md` now states the claim boundary directly: the experiment supports two-sided residual-Laurent physical-space repair as the FEAST-style higher-moment update, not scalar RII on expanded moment columns, while chart policy/extractor agreement remain part of the realization and acceptance layer. `DERIVATION.md` records the contour/Laurent argument, reductions, RII compatibility ladder, candidate proof obligations, and the separation between positive realization/extraction moments and inverse-Laurent residual-enrichment moments. Linear FEAST/SS-FEAST have a true RII/filter identity, canonical NLFEAST is the `K=1` Keldysh-local rung, polynomial moments can be checked against companion/invariant-pair structure, and fully analytic moment-NLFEAST uses residual-Laurent repair rather than literal scalar RII on expanded moments. | Satisfied as candidate |
| Reduce to linear FEAST / SS-FEAST | `README.md`, `run_linear_ss_feast_control`, `run_linear_dual_rii_reduction_diagnostic`, slow tests `linear SS-FEAST` and `dual linear RII` | Same-probe FEAST returns 4/10, wide FEAST returns 10/10, SS-FEAST with `K=3` returns 10/10 using four physical probes. The dual RII reduction diagnostic verifies that scalar residual-inverse iteration equals the FEAST contour filter on extracted right/left Ritz vectors for diagonal and nonnormal Grcar controls. | Numerically pinned |
| Pin true dual extraction/update | `run_dual_reduced_polynomial_control`, `run_dual_residual_laurent_two_sided_update_control`, slow tests `dual reduced extraction` and `dual residual Laurent update`, `ALGORITHM.md` | On a dual-sensitive polynomial, true dual and biorthogonal dual extraction recover all 12 target roots. One-sided Galerkin extraction returns 13 inside reduced Ritz values with reduced residual near `1e-15`, but zero values satisfy the original residual tolerance. On the weak-basis variant, two-sided residual-Laurent repair expands both physical spaces from dimension three to six and recovers all 12 target roots; one-sided repair requires truncating back to a square reduced problem and recovers zero. | Numerically pinned |
| Connect Beyn/SS | Loewner/SS extractors in `run.jl`, rational-coordinate/Loewner tests, `run_delay_count_driven_extractor_agreement`, slow test `agrees across extractors without root oracle` | Reduced extraction is treated as the zero-update contour-realization stage. Loewner succeeds on the exponential global chart where monomial Hankel variants are unstable. On the scalar delay no-oracle control, Loewner-counted and counted SS/Hankel extraction both complete the same three-root contour count and agree on a cross-extractor supported set of size three without exact roots. | Numerically pinned as reduced-extraction rung |
| Connect canonical NLFEAST | Existing `nlfeast!`, `run_canonical_nlfeast_limit_diagnostic`, slow test `canonical NLFEAST limit`, and `ALGORITHM.md` reductions | A three-component one-root-per-component rational control compares existing `nlfeast!`, scalar-expanded residual RII, and compressed residual-Laurent update. All three recover the same three target values to residuals near machine precision, pinning the `K=1` bridge while preserving the distinction from higher-moment state. | Numerically pinned for the one-moment limit |
| Define the moment update | `residual_laurent_update`, `moment_compressed_dual_rii_bases_generic`, `run_residual_laurent_update_ladder_diagnostic`, `run_residual_laurent_compression_diagnostic`, `run_residual_laurent_residual_coordinate_invariance_diagnostic`, `run_residual_laurent_realization_closure_diagnostic`, `ALGORITHM.md`, `DERIVATION.md` | Residual Laurent moments repair left/right physical spaces instead of iterating expanded Hankel columns. The derivation note connects the update to Keldysh local resolvent form, scalar RII, the circular-chart Laurent expansion, and the RII compatibility ladder. The nonnormal chart-cover ladder pins the lower-rung interpretation numerically: reduced extraction alone recovers `34/44`, one residual update recovers `42/44`, and two updates recover `44/44`. The rank-deficient analytic compression diagnostic shows why fully analytic moment-NLFEAST should not be literal scalar RII on expanded moment columns: scalar-expanded RII uses more candidate columns and misses roots, while compressed residual-Laurent repair uses the residual subspace, adds fewer columns, and recovers the full target set. The residual-coordinate invariance diagnostic mixes scalar Ritz residual columns before compression and verifies the same updated physical spaces and retained roots. The exact-realization closure diagnostic verifies zero residual rank and no added physical directions on exact left/right linear eigenspaces. These pin the first and third proof obligations from `DERIVATION.md`. | Core candidate implemented |
| Handle higher moments elegantly | `TrialSpaces`, `MomentPipelineConfig`, `MomentBasisConfig`, `ReducedExtractorConfig`, Loewner/counting extractors, chart policy | The iterative state is physical `X,Y` plus reduced realization/extractor; expanded moments are not exposed as state. The initial contour-moment basis and the chart-local basis/extractor/update bundle are now explicit experiment objects instead of only loose keywords. | Satisfied as design boundary |
| Avoid ad-hoc black-box hacks | `CountDrivenPolicyConfig`, `CountDrivenNumericsConfig`, `MomentPipelineConfig`, `MomentBasisConfig`, `ReducedExtractorConfig`, `ResidualUpdateConfig`, `retention_policy_decision`, `chart_policy_plan`, tests | Retention/refinement is expressed as support/count/residual/agreement diagnostics and explicit chart actions. Selected count-deficit charts shrink, while count-error-only nonnormal charts preserve parent-radius candidates. The newer count-driven stress runners now pass chart spacing, support threshold, refinement depth, chart radii, optional radius-ladder stages, residual tolerance, and count tolerance as one policy object. Numerical basis/extraction/update choices now lower from `CountDrivenNumericsConfig` into a shared `MomentPipelineConfig`, and the central analytic iteration accepts that bundle directly. | Improving, not final |
| Provide theoretical basis | `LITERATURE.md`, `README.md`, `ALGORITHM.md`, `DERIVATION.md` | Systems/Loewner, SS/Hankel realization, invariant-pair/block Newton, algebraic multiplicity by argument principle, rational Krylov/NLEIGS, infinite-GMRES contour solves, SS parameter-estimation references, targeted residual-update and dual-extraction searches, a broader adjacent contour-iteration search, a local NEP-PACK contour implementation check, a May 2026 recheck over RSRR/CISS/Riesz-projection implementations, a targeted adjacent refinement recheck over nonlinear RII, nonlinear FEAST, invariant-pair Newton, refined nonlinear Rayleigh--Ritz, SS projection, and deflation, a focused extraction-versus-update recheck over nonlinear FEAST, Yokota--Sakurai projection, Beyn/tensor variants, RSRR, and refined Rayleigh--Ritz, and a focused residual-realization novelty pass over Loewner/residual/interpolation terminology are summarized. The derivation note records the local residual-Laurent argument, reduction checks, and proof obligations that still need formal constants/rates. | Stronger local basis; publication-level review still remains |
| Provide numerical evidence | Slow tests under `just test --slow 'moment RII'` and focused presets | Linear, polynomial, analytic local-chart, Loewner-layout, extractor-agreement, near-pole rational, rational-coordinate, block-Newton boundary, proof-obligation controls, and matrix-valued mixed diagnostic tests. The polynomial bridge diagnostic now checks the degree-eight nonnormal polynomial against FEAST on the companion pencil and verifies that companion FEAST, polynomial-native extraction, block-Newton cleanup, and the residual-Laurent update recover the same 20 target roots. | Strong for dense experiment controls |
| Care about numerical properties | Rank, support, residual, count, layout/extractor agreement diagnostics | Tests pin cases where residuals, support, count, and agreement disagree. | Satisfied for current controls |
| Care about efficiency | Algorithm avoids expanded Hankel state; residual update uses low-rank residual bases | `run_residual_laurent_compression_diagnostic` now returns an explicit efficiency scorecard. It shows the moment update recovering a rank-deficient analytic chart with fewer candidate columns, candidate-ratio savings, and a larger observable physical realization than scalar expanded RII; scalar-expanded RII fails despite using more candidate columns. `run_residual_laurent_low_rank_equivalence_diagnostic` verifies that residual-basis compression preserves the same updated physical spaces as an uncompressed residual-block update while reducing residual rank and candidate columns. `run_residual_laurent_residual_coordinate_invariance_diagnostic` verifies the compressed residual update is invariant to nonsingular residual-coordinate mixing, so the efficiency transformation is also a realization-gauge invariant transformation. `run_sparse_linear_moment_pipeline_smoke` verifies sparse `Tmatrix`/`Tsolve` can flow through the experiment pipeline. `run_sparse_nonlinear_gallery_moment_pipeline_smoke` verifies a FEAST-native sparse quadratic polynomial gallery operator flows through the nonlinear sparse moment path with known roots. `run_sparse_symbolic_reuse_residual_laurent_smoke` verifies no-store fixed-pattern sparse symbolic factor reuse and side-local dense solve-buffer reuse reproduce the generic sparse residual-Laurent update. `run_sparse_schrodinger_moment_gallery_smoke` verifies a realistic sparse moving-boundary Schrodinger gallery operator with a full-operator contour count and residual-Laurent repair on the established target region. `run_sparse_stored_factor_residual_laurent_smoke` verifies cached contour-node sparse factorizations reproduce the generic sparse update and are reused on a repeated update. `run_residual_laurent_partition_diagnostic` verifies the residual-Laurent update decomposes over contour-node partitions with roundoff-level projection gaps. `run_remote_residual_laurent_worker_diagnostic` verifies actual Julia worker processes can retain operator data and contour-node subsets across two updates while reproducing the serial update. `run_sparse_remote_residual_laurent_worker_smoke` verifies the same persistent worker boundary accepts a sparse linear operator closure. `run_sparse_remote_stored_factor_worker_smoke` verifies persistent workers can own fixed contour-node sparse factors and node-local solve buffers, reuse them across repeated updates, and match the serial update. `run_sparse_nonlinear_remote_stored_factor_worker_smoke` verifies the same stored-factor worker boundary on a nonlinear sparse polynomial gallery control. `run_sparse_schrodinger_remote_stored_factor_worker_smoke` verifies realistic sparse Schrodinger worker-owned factors and solve buffers across repeated updates. `just bench-moment` is a first BenchmarkTools-backed local harness for the sparse Schrodinger serial and remote stored-factor paths, including reuse counters and allocations. Broader sparse workspace reuse, broader realistic sparse gallery coverage, and publication-level performance/scaling evidence remain open. | Partially evidenced |
| Avoid exact-root oracle in policy | `full_operator_count_estimate`, `adaptive_retention_score_summary`, `retention_policy_decision`, `CountDrivenPolicyConfig`, `CountDrivenNumericsConfig`, `count_driven_chart_diagnostic_summary`, `run_count_driven_policy_diagnostic`, `run_count_driven_adaptive_grid_refinement`, `run_delay_count_driven_adaptive_refinement`, `run_delay_count_driven_extractor_agreement`, `run_multi_delay_count_driven_adaptive_refinement`, `run_two_delay_count_driven_adaptive_refinement`, `run_coupled_two_delay_count_driven_adaptive_refinement`, `run_coupled_two_delay_mixed_policy_stress`, `run_dense_multi_delay_weak_support_stress`, `run_near_pole_rational_count_driven_adaptive_refinement`, `run_duplicate_delay_count_driven_adaptive_refinement` | The three-function retention policy now decides support completeness and adaptive stopping against a full-operator argument-principle count. The scalar delay control goes further: it supplies no exact-root list and still stops when retained support matches the reliable contour count. The scalar delay path is pinned with `CountDrivenNumericsConfig(extractor=:ss_counted)` and with a Loewner-counted vs counted SS/Hankel agreement diagnostic, so the no-oracle policy path is not tied only to one reduced extractor. The multi-delay triangular control now pins a no-oracle nonnormal stress where the base cover sees all nine residual-small candidates but retains only three support-2 roots before weak-center refinement completes the count. The two-delay scalar control exercises a single quasipolynomial component with two exponential delay scales. The coupled two-delay controls exercise dense 2x2 NEPs whose determinant roots are not independent scalar component roots; the larger mixed-policy stress has residual-small extras, weak support, and selected local count warnings in the same no-oracle run before completing by weak-center refinement. The dense multi-delay stress repeats the weak-support/exterior-extra pattern on a fully dense 3x3 delay/two-delay NEP, giving a second non-triangular no-oracle problem class. The reusable policy diagnostic wrapper keeps these no-oracle stress cases on the same result/diagnostic path and now receives structured policy and numerics objects. The near-pole rational control exercises that path with exterior rational singularities. The duplicate-delay control supplies no exact roots while exercising the algebraic multiplicity branch. | Satisfied for count-driven analytic policy paths |
| Separate algebraic count from unique-root retention | `local_cluster_multiplicity_estimates`, `run_triangular_count_driven_adaptive_refinement`, `run_squared_sine_count_driven_adaptive_refinement`, `run_duplicate_delay_count_driven_adaptive_refinement`, slow tests `count-driven refinement accounts for multiplicity`, `count-driven refinement handles repeated analytic roots`, and `count-driven refinement handles oracle-free multiplicity` | A nonnormal triangular control validates all 11 unique roots while the full determinant count is 12 because roots coincide. A `sin(z)^2` control validates seven unique roots while the full determinant count is 14. A duplicate-delay triangular control uses no exact-root list: full count is six, retained geometric support has three values, and every retained cluster receives multiplicity two. Local contour counts assign multiplicities to retained clusters, so the multiplicity-weighted retained count satisfies the algebraic count without duplicating scalar values. | Satisfied as boundary evidence |
| Avoid unnecessary multiplicity work | `run_count_driven_adaptive_grid_refinement`, slow test `count-driven refinement stops without exact roots` | The simple-root count-driven path now stops when unique support equals the target count and returns no local multiplicity probes. Local cluster counts are only run on the algebraic/unique mismatch branch. | Satisfied for current count-driven path |

## Current Candidate

The current candidate solution is documented in `ALGORITHM.md`:

```text
generalized moment-NLFEAST =
    local dual contour realization
  + reduced Petrov-Galerkin NEP extraction
  + residual Laurent correction of left/right physical spaces
  + explicit chart policy driven by support, counts, residuals, and agreement
```

This is not yet a public solver API. It is the best current algorithm-family
story and matches the experiments better than scalar RII on expanded moments,
global block Newton, or rational-coordinate-only fixes.

## Negative Results To Preserve

- Analytic invariant-pair block Newton is a local refinement/check, not the
  missing global moment update.
- Naive finite Laurent truncation of only the scalar denominator
  `1/(z-lambda)` is not a valid proof of the update. A linear diagnostic shows
  this can fail even when chart coordinates are small because `T(z)^(-1)` has
  interior poles. Finite-order arguments must use the captured contour
  realization/rank and the lower-rung FEAST identities.
- Inverse and Mobius moment coordinates do not replace Loewner/local charts on
  the exponential many-root global chart.
- Near-pole rational singularities outside the contour do not currently require
  a special update branch; the existing reduced extraction and residual Laurent
  framework is stable on the dense control.
- Scalar expanded RII is not an equivalent replacement for the residual
  Laurent-moment update. On the rank-deficient analytic compression diagnostic
  it loses the observable realization and fails, even though it sees the same
  low residual rank.
- Support thresholds alone are not pruning laws; support `>=3` can drop true
  roots if the cover is not dense enough.
- Scalar residuals alone are not acceptance evidence in high dynamic-range
  analytic NEPs.
- Disabling residual Laurent repair does not guarantee arbitrary local trial
  spaces solve analytic many-root cases; Beyn/SS is the extraction stage, while
  residual repair and chart policy remain central.

## Remaining Gaps

- Sparse and distributed moment-NLFEAST are prototyped only inside the
  experiment, not productized as solver APIs. The experiment now has a sparse
  linear pipeline smoke test showing generic sparse
  `Tmatrix`/`Tsolve` compatibility and a stored-factor sparse smoke showing
  fixed contour-node factorizations can be reused across residual-Laurent
  updates. It also has a partition diagnostic proving that
  residual-Laurent contour-node sums can be distributed algebraically. Serial
  and partitioned residual-Laurent updates now share the same node-local
  accumulation and candidate-compression helpers. The first remote worker
  diagnostic now stores operator closures and contour-node subsets in actual
  Julia worker processes, sends only residual bases for two update calls, and
  reduces the returned Laurent blocks to match the serial update. The remote
  dense and sparse diagnostics now carry lightweight setup/update/worker timing
  metadata as a profiling smoke check. A sparse diagonal linear smoke runs
  through the same persistent worker boundary, and the sparse remote
  stored-factor smoke now verifies fixed contour-node sparse factors live on
  those persistent workers and are reused across repeated updates while
  matching the serial update. The sparse symbolic-reuse smoke verifies the
  no-store fixed-pattern rung by refreshing one UMFPACK factor per side across
  contour nodes while reproducing the generic sparse update. It now also pins
  one side-local dense solve buffer per side across repeated updates. The
  sparse nonlinear gallery smoke adds a first known-root nonlinear sparse
  operator through the FEAST-native gallery materializer and moment pipeline,
  and the sparse nonlinear remote stored-factor smoke verifies the same
  operator family across persistent worker-owned contour factors. The
  Schrodinger sparse gallery smoke adds a realistic sparse NEP with a
  full-operator contour count and residual repair on the established target
  region, and the Schrodinger remote stored-factor smoke verifies the same
  realistic sparse operator family across persistent worker-owned contour
  factors and node-local solve buffers. `just bench-moment` now gives a first
  local BenchmarkTools harness for that Schrodinger serial/remote rung and
  prints timing, allocation, correctness, and reuse counters after warmup. It
  was most recently run on the small default Schrodinger case with two workers
  after adding setup/update timing splits and a sparse stored-factor remote
  plan boundary. The small default Schrodinger run still shows the expected
  pattern: setup and first-use costs dominate the coarse wall clock, while
  repeated updates reuse the same 96 worker-owned factors and solve buffers and
  match the serial spaces to projection gaps near `1e-14`. This is enough
  implementation evidence for now; it is not a benchmark-maxing result.
  The experiment deliberately does not yet provide full reusable sparse
  workspaces, broad realistic sparse gallery coverage, or publication-level
  scaling claims.
  `IMPLEMENTATION.md` records the sparse and distributed rungs needed to turn
  the experiment boundary into a real implementation.
- Split/shrink chart policy is now a reproducible local refinement rung, not a
  fully optimized adaptive chart cover. Naive half-radius child covers fail on
  the selected count-stressed radius-20 charts; residual candidate centers plus
  overlapping parent/child cover anchors recover and support-certify the local
  roots in the current probe. A triangular nonnormal count-error-only chart
  requires preserving the parent radius; the aggressive shrink radii that work
  for count deficits fail there. The policy split is now named explicitly:
  count deficits shrink around residual candidates, while count-error-only
  charts preserve parent-radius candidates. The sparse coupled two-delay
  radius-12 cover is now pinned as an unresolved-defect diagnostic: the policy
  retains 10 of 12
  counted roots and does not over-accept the incomplete cover. A blind
  supplemental half-grid fill-in was tried and rejected: it adds chart centers
  but still plateaus at 10 of 12 retained roots. Adding a larger local chart
  radius `3.0` repairs the same case with one weak-center refinement, so the
  useful next policy rung is adaptive radius/overlap selection, not blind
  densification. `CountDrivenPolicyConfig.chart_radii_stages` now makes that
  rung explicit through the same policy diagnostic path: first stage diagnoses
  `10/12`, second stage completes `12/12` with larger chart overlap.
- Oracle-free target completion and stopping are now covered by the radius-20
  three-function policy path, scalar delay, scalar two-delay, nonnormal
  multi-delay, dense coupled two-delay, near-pole rational, and
  duplicate-delay multiplicity controls, but not yet generalized across every
  experiment harness. Exact roots are still used broadly for validation and for
  supervised controls.
- Near-pole rational controls now diagnose a too-close exterior pole as an
  unreliable contour count rather than pretending the solver should hide an
  unsound contour placement. This is a diagnostic boundary, not a promise to
  solve arbitrary near-pole contours.
- Algebraic multiplicity is now handled for retained unique clusters by local
  contour counts on coincident-component, repeated-root analytic, and
  oracle-free duplicate-delay controls, but there is not yet a full
  multiplicity-aware retained state representation with derivative/Jordan data.
  That remains an escalation path.
- The reduced-extractor interface is still experimental.
- More problem classes are still needed before calling the algorithm
  publication ready, but the count-driven lower rung now covers scalar delay,
  scalar two-delay, nonnormal multi-delay, dense coupled two-delay, near-pole
  rational, and multiplicity controls without exact-root stopping or
  validation oracles. The nonnormal multi-delay test now uses a coarse cover
  that must refine weak support rather than trivially completing on the base
  grid. The fully coupled two-delay mixed-policy stress records residual/support
  disagreement plus local count warnings in a non-triangular no-oracle run. A
  dense 3x3 multi-delay stress shows weak-center refinement is not tuned only to
  the two-delay toy. The triangular multiplicity test covers the algebraic-count
  completion branch.
- Broader literature review now has a first pass over rational Krylov/NLEIGS,
  infinite-GMRES contour solves, SS parameter selection, quasi-Newton/RII
  interpretations, systems/Loewner contour methods, contour invariant-pair
  methods, resolvent-sampling Rayleigh--Ritz, Riesz-projection methods,
  contour algebraic-multiplicity counts, and recent Beyn/RIM region
  partitioning. The latest adjacent implementation recheck suggests RSRR,
  Loewner, CISS Ritz/Hankel extraction, and Riesz-projection methods mostly
  strengthen the extractor/selection side of the architecture rather than
  replacing the residual-Laurent repair loop. A publication-level novelty
  review should still be done before making final claims, with particular care
  around whether an adjacent invariant-pair or model-reduction formulation can
  be specialized to the same update in different language.

## Completion Status

The goal is not complete. We have a strong candidate algorithm family with
test-backed boundaries, and the oracle-free count-driven branch is now covered
across several analytic problem classes, including a nonnormal weak-support
stress. `just test --preset moment-core` is now the focused reduction gate for the
FEAST/SS, Beyn/SS, dual extraction, canonical NLFEAST, and residual-Laurent
update story. The gate was last run after the remote steady-state timing update
and passed 173 assertions in 2m34.9s, covering linear SS-FEAST, dual linear
RII, the polynomial companion bridge, dual extraction/update, canonical
NLFEAST, residual-Laurent compression/equivalence/coordinate invariance/closure, sparse factor reuse,
extractor agreement, and contour partitioning. This is
not yet a decisive final solver or proof. The next
productive steps are:

1. Continue tightening the experiment interface around a small set of stable
   objects: chart, pipeline, basis, extractor, update, policy, and diagnostic.
   `pipeline.jl` now holds the chart/pipeline/basis/extractor/update objects, while
   `policy.jl` holds `CountDrivenPolicyConfig`, `CountDrivenNumericsConfig`,
   and the named count-stressed chart-refinement rule. The local chart sweep
   and central analytic iteration now pass the pipeline bundle through
   directly. Older helper harnesses still expose many loose keywords and should
   be migrated opportunistically.
2. Continue publication-level novelty review before making final claims. The
   current literature, NEP-PACK, RSRR, CISS, Riesz-projection, and targeted
   extraction-versus-update checks did not find the exact residual-Laurent
   finite-realization iteration, but they are not an exhaustive publication
   review.
3. Broaden problem-class evidence around the chart policy and reduced
   extractor layer, not around benchmark maxing. The current sparse/distributed
   implementation rungs are sufficient as rough feasibility evidence; broader
   sparse coverage and scaling can wait until the algorithm story is more
   publication-ready.
4. Only after those pass, consider extracting stable pieces from the experiment
   into a real implementation plan.

## Latest Completion Audit Snapshot

Concrete deliverables from the objective are mostly covered at the experiment
level:

- Unified family story: covered by `ALGORITHM.md` and `DERIVATION.md`.
- Elegant higher-moment update: covered as the residual-Laurent two-sided
  physical-space repair, with scalar expanded RII rejected as the wrong state.
- Numerical evidence: covered by `moment-core`, count-driven, sparse, and
  distributed smoke controls.
- Efficiency awareness: covered by low-rank residual compression and rough
  sparse remote plan evidence; no benchmark-maxing claim is needed.
- Containment: all new algorithm objects remain inside `experiments/moment_rii`.

The remaining blocker is publication-level confidence, not another local
implementation trick. We still need either a more formal proof/derivation that
identifies the residual-Laurent repair in established realization language, or
a broader novelty review showing that adjacent contour projection,
Loewner/realization, invariant-pair Newton, and refined Rayleigh--Ritz methods
do not already contain the same update in different notation. Until then the
status remains "strong candidate", not "solved decisively."
