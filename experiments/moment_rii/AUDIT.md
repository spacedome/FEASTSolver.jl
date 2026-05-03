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
| Keep work contained to the experiment | `experiments/moment_rii/run.jl`, `pipeline.jl`, `experiment_matrix.jl`, `ALGORITHM.md` | No public API promotion; new algorithm objects are experiment-layer only. | Satisfied for current work |
| Explain unified algorithm family | `ALGORITHM.md` | Candidate formula: local dual contour realization + reduced Petrov-Galerkin extraction + residual Laurent repair + chart policy. | Satisfied as candidate |
| Reduce to linear FEAST / SS-FEAST | `README.md`, `run_linear_ss_feast_control`, slow test `linear SS-FEAST` | Same-probe FEAST returns 4/10, wide FEAST returns 10/10, SS-FEAST with `K=3` returns 10/10 using four physical probes. | Numerically pinned |
| Connect Beyn/SS | Loewner/SS extractors in `run.jl`, rational-coordinate/Loewner tests | Reduced extraction is treated as the zero-update contour-realization stage. Loewner succeeds on the exponential global chart where monomial Hankel variants are unstable. | Conceptually supported; not a standalone universal solve |
| Connect canonical NLFEAST | Existing `nlfeast!`/moment tests and `ALGORITHM.md` reductions | `K=1` diagonal-state path is described as the canonical NLFEAST/RFI limit; nonlinear fast tests cover standard `nlfeast!`. | Partially evidenced |
| Define the moment update | `residual_laurent_update`, `moment_compressed_dual_rii_bases_generic`, `ALGORITHM.md` | Residual Laurent moments repair left/right physical spaces instead of iterating expanded Hankel columns. | Core candidate implemented |
| Handle higher moments elegantly | `TrialSpaces`, `ReducedExtractorConfig`, Loewner/counting extractors, chart policy | The iterative state is physical `X,Y` plus reduced realization/extractor; expanded moments are not exposed as state. | Satisfied as design boundary |
| Avoid ad-hoc black-box hacks | `retention_policy_decision`, `chart_policy_plan`, tests | Retention/refinement is expressed as support/count/residual/agreement diagnostics and explicit chart actions. Selected count-deficit charts shrink, while count-error-only nonnormal charts preserve parent-radius candidates. | Improving, not final |
| Provide theoretical basis | `LITERATURE.md`, `README.md`, `ALGORITHM.md` | Systems/Loewner, SS/Hankel realization, invariant-pair/block Newton, algebraic multiplicity by argument principle, rational Krylov/NLEIGS, infinite-GMRES contour solves, and SS parameter-estimation references are summarized. | Stronger local basis; publication-level review still remains |
| Provide numerical evidence | Slow tests under `just test 'moment RII'` | Linear, polynomial, analytic local-chart, Loewner-layout, extractor-agreement, near-pole rational, rational-coordinate, and block-Newton boundary tests. | Strong for dense experiment controls |
| Care about numerical properties | Rank, support, residual, count, layout/extractor agreement diagnostics | Tests pin cases where residuals, support, count, and agreement disagree. | Satisfied for current controls |
| Care about efficiency | Algorithm avoids expanded Hankel state; residual update uses low-rank residual bases | `run_residual_laurent_compression_diagnostic` shows the moment update recovering a rank-deficient analytic chart with fewer candidate columns than scalar expanded RII. Sparse/distributed moment implementation and benchmark-level performance remain open. | Partially evidenced |
| Avoid exact-root oracle in policy | `full_operator_count_estimate`, `adaptive_retention_score_summary`, `retention_policy_decision`, `run_count_driven_adaptive_grid_refinement`, `run_delay_count_driven_adaptive_refinement`, `run_multi_delay_count_driven_adaptive_refinement`, `run_two_delay_count_driven_adaptive_refinement`, `run_coupled_two_delay_count_driven_adaptive_refinement`, `run_near_pole_rational_count_driven_adaptive_refinement`, `run_duplicate_delay_count_driven_adaptive_refinement` | The three-function retention policy now decides support completeness and adaptive stopping against a full-operator argument-principle count. The scalar delay control goes further: it supplies no exact-root list and still stops when retained support matches the reliable contour count. The multi-delay triangular control exercises the same no-oracle simple-root path in a nonnormal dimension-three setting. The two-delay scalar control exercises a single quasipolynomial component with two exponential delay scales. The coupled two-delay control exercises a dense 2x2 NEP whose determinant roots are not independent scalar component roots. The near-pole rational control exercises that path with exterior rational singularities. The duplicate-delay control supplies no exact roots while exercising the algebraic multiplicity branch. | Satisfied for count-driven analytic policy paths |
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

- Sparse and distributed moment-NLFEAST are not implemented.
- Split/shrink chart policy is now a reproducible local refinement rung, not a
  fully optimized adaptive chart cover. Naive half-radius child covers fail on
  the selected count-stressed radius-20 charts; residual candidate centers plus
  overlapping parent/child cover anchors recover and support-certify the local
  roots in the current probe. A triangular nonnormal count-error-only chart
  requires preserving the parent radius; the aggressive shrink radii that work
  for count deficits fail there.
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
  validation oracles.
- Broader literature review now has a first pass over rational Krylov/NLEIGS,
  infinite-GMRES contour solves, SS parameter selection, quasi-Newton/RII
  interpretations, systems/Loewner contour methods, and contour invariant-pair
  methods. A publication-level novelty review should still be done before
  making final claims.

## Completion Status

The goal is not complete. We have a strong candidate algorithm family with
test-backed boundaries, and the oracle-free count-driven branch is now covered
across several analytic problem classes, but this is not yet a decisive final
solver or proof. The next productive steps are:

1. Stress the same policy on harder matrix-valued analytic problems where
   residual-small extras, weak support, and local count warnings interact in
   the same run.
2. Continue the broader literature pass before making novelty claims, focusing
   on whether any published contour method iterates a finite realization by a
   residual-inverse moment correction.
3. Only after those pass, consider extracting stable pieces from the experiment
   into a real implementation plan.
