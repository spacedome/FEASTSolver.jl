module FusedNLFEAST

using LinearAlgebra
using FastGaussQuadrature: gausslegendre
using SparseArrays: sparse

include("cache.jl")
include("counting.jl")
include("tangents.jl")
include("moments.jl")
include("coupling.jl")
include("realization.jl")
include("invariant_pairs.jl")
include("analytic_actions.jl")
include("state_filter.jl")
include("two_sided_state.jl")
include("independent_state.jl")
include("common_state.jl")
include("state_api.jl")
include("state_candidates.jl")
include("state_iteration.jl")
include("state_wrappers.jl")
include("partition.jl")
include("loewner.jl")
include("iteration.jl")
include("cases.jl")

export AbstractContourChart, CircularChart, RectangularChart, ContourSampleCache, FusedConfig
export ContourSolveResult, ContourSolveDiagnostic, ContourSolveFailureError
export chart_boundary_margin, chart_inward_score
export chart_node_cover_radius
export chart_boundary_segments
export ArgumentCountEstimate, argument_principle_count
export determinant_winding_count
export certified_determinant_winding_value
export MeromorphicCountEstimate, meromorphic_eigenvalue_count
export FusedExtraction, FusedResult, IterationRecord, LoewnerDiagnostics
export add_left_probe!, add_right_probe!, corrected_moments, fused_nlfeast, probe_moments
export combined_probe, combined_probe_moments
export drop_left_probe!, drop_right_probe!
export release_left_responses!, release_right_responses!
export mixed_corrected_moments
export modal_loewner_pencil, moment_tangent, probe_moment_data, residual_completed_overlap
export streaming_hankel_realization
export residual_factors, solve_counts
export solve_diagnostic_summary
export CommonStateRealization, balanced_common_state, common_state_realization
export common_state_from_independent
export common_polynomial_realization, common_structured_realization
export IndependentStateRealization, independent_state_realization
export independent_polynomial_realization, independent_structured_realization
export schur_common_state
export StateIterationConfig, StateIterationRecord, StateIterationResult
export InvalidAnalyticDomainError, validate_analytic_chart
export StateCandidateFailure
export fused_state_nlfeast, structured_fused_state_nlfeast, state_components
export holomorphic_fused_state_nlfeast
export state_modal_output
export state_contour_separation
export PartitionedStateResult, bisect_chart, partitioned_state_solve
export PartitionedInvariantState, partitioned_invariant_state
export polynomial_invariant_residual
export polynomial_left_invariant_residual, polynomial_state_divided_overlap
export state_corrected_moments, state_moment_tangent, state_residual_factors
export TwoSidedHankelRealization, common_spectral_restriction
export two_sided_hankel_realization, two_sided_structured_realization
export matrix_divided_action, structured_invariant_residual
export structured_eigenvalue_condition_numbers
export cauchy_invariant_data
export cauchy_left_invariant_residual, cauchy_right_invariant_residual
export cauchy_state_divided_overlap
export lifted_pair_gauges, structured_pair_backward_error
export lifted_residual_error
export lift_normalized_components
export lift_normalized_independent_state
export structured_left_invariant_residual, structured_state_divided_overlap
export block_loewner_pencil, loewner_realization, rational_probe_samples
export canonical_one_root_case, linear_case, many_root_polynomial_case
export nonnormal_analytic_case, polynomial_case, scalar_sine_case, shared_eigenvector_case

end
