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
| Define the moment update | `residual_laurent_update`, `moment_compressed_dual_rii_bases_generic`, `run_residual_laurent_update_ladder_diagnostic`, `run_residual_laurent_compression_diagnostic`, `run_residual_laurent_residual_coordinate_invariance_diagnostic`, `run_positive_moment_realization_recurrence_diagnostic`, `run_residual_laurent_realization_closure_diagnostic`, `ALGORITHM.md`, `DERIVATION.md` | Residual Laurent moments repair left/right physical spaces instead of iterating expanded Hankel columns. The derivation note connects the update to Keldysh local resolvent form, scalar RII, the circular-chart Laurent expansion, and the RII compatibility ladder. The nonnormal chart-cover ladder pins the lower-rung interpretation numerically: reduced extraction alone recovers `34/44`, one residual update recovers `42/44`, and two updates recover `44/44`. The rank-deficient analytic compression diagnostic shows why fully analytic moment-NLFEAST should not be literal scalar RII on expanded moment columns: scalar-expanded RII uses more candidate columns and misses roots, while compressed residual-Laurent repair uses the residual subspace, adds fewer columns, and recovers the full target set. The residual-coordinate invariance diagnostic mixes scalar Ritz residual columns before compression and verifies the same updated physical spaces and retained roots. The positive-moment recurrence diagnostic pins the separate extraction layer by verifying `M_k = X*S^(k-1)*C` for a linear transfer realization. The exact-realization closure diagnostic verifies zero residual rank and no added physical directions on exact left/right linear eigenspaces. These pin the first and third proof obligations from `DERIVATION.md` and clarify the realization-rank layer behind the second. | Core candidate implemented |
| Handle higher moments elegantly | `TrialSpaces`, `MomentPipelineConfig`, `MomentBasisConfig`, `ReducedExtractorConfig`, Loewner/counting extractors, chart policy | The iterative state is physical `X,Y` plus reduced realization/extractor; expanded moments are not exposed as state. The initial contour-moment basis and the chart-local basis/extractor/update bundle are now explicit experiment objects instead of only loose keywords. | Satisfied as design boundary |
| Avoid ad-hoc black-box hacks | `CountDrivenPolicyConfig`, `CountDrivenNumericsConfig`, `MomentPipelineConfig`, `MomentBasisConfig`, `ReducedExtractorConfig`, `ResidualUpdateConfig`, `retention_policy_decision`, `chart_policy_plan`, tests | Retention/refinement is expressed as support/count/residual/agreement diagnostics and explicit chart actions. Selected count-deficit charts shrink, while count-error-only nonnormal charts preserve parent-radius candidates. The newer count-driven stress runners now pass chart spacing, support threshold, refinement depth, chart radii, optional radius-ladder stages, residual tolerance, and count tolerance as one policy object. Numerical basis/extraction/update choices now lower from `CountDrivenNumericsConfig` into a shared `MomentPipelineConfig`, and the central analytic iteration accepts that bundle directly. | Improving, not final |
| Provide theoretical basis | `LITERATURE.md`, `README.md`, `ALGORITHM.md`, `DERIVATION.md`, `THEOREM_SKETCH.md` | Systems/Loewner, SS/Hankel realization, invariant-pair/block Newton, algebraic multiplicity by argument principle, rational Krylov/NLEIGS, infinite-GMRES contour solves, SS parameter-estimation references, targeted residual-update and dual-extraction searches, a broader adjacent contour-iteration search, a local NEP-PACK contour implementation check, a May 2026 recheck over RSRR/CISS/Riesz-projection implementations, a targeted adjacent refinement recheck over nonlinear RII, nonlinear FEAST, invariant-pair Newton, refined nonlinear Rayleigh--Ritz, SS projection, and deflation, a focused extraction-versus-update recheck over nonlinear FEAST, Yokota--Sakurai projection, Beyn/tensor variants, RSRR, and refined Rayleigh--Ritz, a focused residual-realization novelty pass over Loewner/residual/interpolation terminology, a Jacobi-Davidson correction-equation boundary pass, and a reduced Ritz perturbation boundary pass are summarized. The derivation note records the local residual-Laurent argument, reduction checks, and proof obligations. `THEOREM_SKETCH.md` now states the local theorem target, proof skeleton, non-claims, and evidence map, and frames the candidate as a FEAST-filtered block correction equation for a finite two-sided contour realization. | Stronger local basis; publication-level review and analytic correction estimate still remain |
| Provide numerical evidence | Slow tests under `just test --slow 'moment RII'` and focused presets | Linear, polynomial, analytic local-chart, Loewner-layout, extractor-agreement, near-pole rational, rational-coordinate, block-Newton boundary, realization/proof-obligation controls, and matrix-valued mixed diagnostic tests. The polynomial bridge diagnostic now checks the degree-eight nonnormal polynomial against FEAST on the companion pencil and verifies that companion FEAST, polynomial-native extraction, block-Newton cleanup, and the residual-Laurent update recover the same 20 target roots. The sparse diagonal proof-control path now records that extracted subspace projection gaps can already be near roundoff while residuals fail, and one residual-Laurent update improves residual scale by more than `1e4`. The rank-deficient analytic compression diagnostic now records nonlinear residual improvement above `1e8` after one compressed residual-Laurent enrichment. | Strong for dense experiment controls; sparse proof-control boundary added |
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
- Subspace angle alone is not a sufficient local theorem. The sparse diagonal
  linear pipeline has extracted right/left subspaces with projection gaps near
  roundoff before the update, but physical residuals are still too large; one
  residual-Laurent update fixes the residuals. The theorem needs reduced
  Ritz/eigenvector residual quality inside the realized subspace, not only an
  angle-to-residue-space statement.
- Reduced block Newton is not the outer moment-update mechanism. Existing
  diagnostics keep it as a local refinement rung after a reliable reduced
  realization is available, not as a substitute for residual-Laurent
  physical-space repair or chart policy.
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
  match the serial spaces to projection gaps near `1e-14`. The fused
  Schrodinger domain-decomposition diagnostic is now the harder Schrodinger
  correctness stress: it eliminates local subdomain interiors from a linear
  finite-difference Schrodinger operator, solves the resulting rational
  interface Schur-complement NEP below the first interior pole, and verifies
  that reduced `Tred` cleanup recovers all 13 target roots on a
  `full_n=207`, `interface_n=15` control after raw fused extraction has large
  physical residuals. Its refinement sweep records the useful calibration:
  64 and 96 contour nodes identify the right rank but remain underresolved
  after cleanup, while 128 nodes plus `Tred` cleanup recovers all targets. This
  path now has a more faithful local-block assembly rung: the larger scale
  smoke uses 64 independent local interior blocks of size 32, giving
  `full_n=2111`, `interface_n=63`, compression ratio above 30, and the same
  13-root recovery after cleanup. A small dense-reference check verifies local
  Schur-complement and derivative assembly to roundoff. This is enough
  implementation evidence for now; it is not a benchmark-maxing result. The
  baseline comparison runner records the expected caveat: on current local
  small and medium instances, full sparse linear FEAST is still faster while
  recovering the same roots. The DD result is therefore a capability/scaling
  formulation result, not yet a performance win on this workstation. The
  packet-defect diagnostic now connects the Julia experiment back to the
  adjacent Lean certificate boundary: using a high-resolution fused/refined
  packet projector as reference, successful `Tred` cleanup removes the
  packet-visible defect on the 128-node DD solve, while the underresolved
  64-node solve keeps a large visible defect. The first policy wrapper now
  turns that split into actions: increase nodes/refine chart for visible packet
  defects, refine extraction/acceptance for packet-invisible gaps, and accept
  when the visible defect is removed.
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
update story. The gate was last run after adding the realization/proof
obligation controls and passed 197 assertions in 2m58.0s, covering linear
SS-FEAST, dual linear RII, the polynomial companion bridge, dual
extraction/update, canonical NLFEAST, residual-Laurent
compression/equivalence/coordinate invariance/closure, the scalar Laurent
truncation boundary, positive-moment realization recurrence, sparse factor
reuse, extractor agreement, and contour partitioning. This is
not yet a decisive final solver or proof. The next
productive steps are:

1. Turn the current theorem sketch into a formal local theorem package. The proof
   should use the now-pinned moment-role split: positive moments provide the
   finite transfer realization and rank/recurrence evidence, while
   inverse-Laurent residual moments enrich the physical trial/test spaces. The
   remaining mathematical gap is a local statement relating residual-subspace
   enrichment, quadrature/rank truncation, and reduced Ritz/eigenvector
   residual quality inside a captured realization. A naive scalar denominator
   truncation proof, a pure subspace-angle proof, and a "just block Newton"
   interpretation have all been explicitly rejected by diagnostics.
   `THEOREM_SKETCH.md` now decomposes the proof into five lemmas; the first
   four are established or diagnostic-pinned, and Lemma 5, the local analytic
   enrichment estimate, is the open mathematical core.
2. Continue tightening the experiment interface around a small set of stable
   objects: chart, pipeline, basis, extractor, update, policy, and diagnostic.
   `pipeline.jl` now holds the chart/pipeline/basis/extractor/update objects, while
   `policy.jl` holds `CountDrivenPolicyConfig`, `CountDrivenNumericsConfig`,
   and the named count-stressed chart-refinement rule. The local chart sweep
   and central analytic iteration now pass the pipeline bundle through
   directly. Older helper harnesses still expose many loose keywords and should
   be migrated opportunistically.
3. Continue publication-level novelty review before making final claims. The
   current literature, NEP-PACK, RSRR, CISS, Riesz-projection, and targeted
   extraction-versus-update checks did not find the exact residual-Laurent
   finite-realization iteration, but they are not an exhaustive publication
   review.
4. Broaden problem-class evidence around the chart policy and reduced
   extractor layer, not around benchmark maxing. The current sparse/distributed
   implementation rungs are sufficient as rough feasibility evidence; broader
   sparse coverage and scaling can wait until the algorithm story is more
   publication-ready.
5. Only after those pass, consider extracting stable pieces from the experiment
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
implementation trick. We now know the formal obstruction precisely:
`THEOREM_SKETCH.md` identifies the missing estimate as perturbation/correction
theory for contour-filtered compressed residual corrections of a two-sided
finite NEP realization. We still need either that proof, or a known theorem
from adjacent Jacobi-Davidson/RII/projection theory that implies it. A broader
novelty review should also verify that adjacent contour projection,
Loewner/realization, invariant-pair Newton, refined Rayleigh--Ritz, and
Jacobi-Davidson methods do not already contain the same update in different
notation. Until then the status remains "strong candidate with a sharply
identified theorem gap", not "solved decisively."

## 2026-05-03 Active Checkpoint

The current implementation/research state should be treated as a principled
stopping boundary, not as a benchmark-tuning checkpoint. The experiment has
enough performance evidence for now: low-rank residual compression, sparse
symbolic/stored-factor reuse smokes, persistent worker ownership smokes, and
the small BenchmarkTools harness all support the claim that the design is not
inherently wasteful. Further benchmark-maxing would not close the research
objective.

The remaining requirement is theoretical and novelty-facing. After sharpening
`THEOREM_SKETCH.md`, the minimum sufficient obligation is not a full global
convergence theorem. We need either:

- a proof of the minimal Lemma 5 range-inclusion estimate in
  `THEOREM_SKETCH.md`, showing that contour-filtered compressed
  residual-Laurent enrichment contains the leading JD/RII-style correction
  directions for a captured two-sided finite NEP realization, up to
  realization, quadrature, and compression errors; or
- a known theorem from Jacobi-Davidson, residual inverse iteration, refined
  Rayleigh--Ritz, invariant-pair, Loewner/realization, or contour projection
  theory that implies the same enrichment step under recognizable hypotheses.

Until one of those exists, the honest claim is:

```text
residual-Laurent two-sided physical-space repair is the best current
experiment-backed candidate for generalized higher-moment NLFEAST iteration,
but it is not yet a decisive theory.
```

No additional local implementation task is currently known to resolve that
gap. Productizing the sparse/distributed rungs or tuning benchmarks should wait
until the local theorem/novelty question is settled.

## 2026-05-03 Fused Realization Completion Audit

### Objective Restated

Explore fused contour-sample realization until either:

- a final unifying algorithm is found; or
- the path is stuck.

The required research direction is to include the interpretation of RII as a
coordinate chart, and to keep the work contained to the experiment.

### Prompt-To-Artifact Checklist

| Requirement | Current Artifact | Evidence | Status |
| --- | --- | --- | --- |
| Fuse outer FEAST and inner Beyn/SS contour data | `ALGORITHM.md` sections `Collapsing Outer And Inner Moments` and `Unified Candidate Algorithm`; `ContourSampleCache` in `pipeline.jl` | The final candidate is now a chart-owned contour sample realization iteration. One contour cache supplies physical moments, small projected transfer moments, and residual-update responses. | Satisfied |
| Avoid solving a projected nonlinear problem as the main path | `ALGORITHM.md`, `DERIVATION.md` | The reduced analytic object `Y^H T(lambda) X` is explicitly demoted to validation, cleanup, or fallback. Extraction is through a small linear realization/SS-Hankel/Loewner/pencil built from the contour transfer data. | Satisfied |
| Pin fused extraction numerically | `run_fused_contour_sample_realization_diagnostic`, `run_fused_polynomial_contour_sample_realization_diagnostic`, tests `fused contour samples replace inner reduced SS` and `fused contour samples recover matrix polynomial realization` | The analytic diagonal control recovers eight roots from one original contour sample cache and matches the redundant inner reduced SS path. The deficient quadratic MatrixMarket control recovers four roots from the fused projected Hankel path and matches both redundant inner reduced SS and companion reference. | Satisfied |
| Pin residual update in the same cache | `augment_contour_sample_cache`, `residual_laurent_moments`, `run_fused_cache_residual_augmentation_diagnostic`, test `fused cache augmentation matches residual Laurent update` | Augmenting the cache with compressed residual probes and reducing those probes with inverse-Laurent weights produces the same repaired physical spaces as the explicit residual-Laurent update, with projection gaps at zero in the diagnostic. | Satisfied |
| Preserve two moment roles | `ALGORITHM.md`, `DERIVATION.md`, `right_moments`, `left_moments`, `residual_laurent_moments` | The implementation and docs now distinguish positive powers for finite realization/extraction from inverse-Laurent powers for residual repair. A failed intermediate attempt clarified that residual probes cannot be treated as ordinary positive moments. | Satisfied |
| Explore RII as coordinate chart | `ALGORITHM.md` section `RII As A Coordinate Chart`; `DERIVATION.md` section `Fused Contour Realization Interpretation` | RII is stated as the scalar coordinate chart of the contour realization: linear FEAST has a global scalar Ritz chart, canonical NLFEAST has a local Keldysh scalar pole chart, and higher-moment NLFEAST needs coordinate-free residual-Laurent cache augmentation. | Satisfied as algorithmic theory |
| Maintain focused evidence gate | `test/options.jl` `moment-core` preset | `moment-core` now includes fused contour sample and fused cache augmentation checks. Latest run: `nix develop --command just test --preset moment-core`, `218/218` assertions passed in `3m14.2s`. | Satisfied |
| Keep work contained to experiment | `experiments/moment_rii/pipeline.jl`, `run.jl`, `ALGORITHM.md`, `DERIVATION.md`; tests under `test/fast/nonlinear.jl` | `ContourSampleCache` and fused diagnostics are experiment-layer objects; no public package API was promoted. | Satisfied |

### Current Algorithmic Conclusion

The fused branch has reached a coherent final algorithmic form:

```text
chart-owned contour sample cache
  -> positive-moment finite realization / small linear extractor
  -> physical residual certification
  -> compressed residual probes appended to the same cache
  -> inverse-Laurent residual reductions
  -> repaired physical spaces
  -> re-extract finite realization
```

This resolves the immediate algorithmic issue that motivated the revisit:
NLFEAST should not first build a projector and then run Beyn/SS as an inner
contour method on a projected nonlinear problem. The inner and outer contour
data are the same transfer data and should be fused.

The coordinate-chart interpretation is also now explicit:

```text
RII is the scalar Ritz-coordinate chart of the contour realization.
Higher-moment NLFEAST replaces scalar expanded-column RII with
coordinate-free residual-Laurent cache augmentation.
```

### Remaining Boundaries

This is an algorithmic completion, not a full convergence proof. The older
Lemma 5 theorem gap remains if the goal becomes publication-level convergence
theory. Productizing this into the main solver API also remains future work.
Those are distinct follow-on projects; they do not block the fused
contour-sample realization objective.
