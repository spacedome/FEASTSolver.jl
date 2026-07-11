struct IndependentStateRealization
    right_state::Matrix{ComplexF64}
    left_state::Matrix{ComplexF64}
    right::Matrix{ComplexF64}
    left::Matrix{ComplexF64}
    right_residual::Matrix{ComplexF64}
    left_residual::Matrix{ComplexF64}
    right_singular_values::Vector{Float64}
    left_singular_values::Vector{Float64}
    right_detected_rank::Int
    left_detected_rank::Int
    rank::Int
end

function independent_state_realization(
    chart,
    right_moments,
    left_moments,
    right_observer,
    left_observer,
    moment_count;
    ranktol,
    maxrank=typemax(Int),
    fixed_rank=nothing,
    restrict_to_chart=true,
    target_count=nothing,
    right_reference_values=nothing,
    left_reference_values=nothing,
    selection_anchor=:none,
    right_residual,
    left_residual,
)
    selection_anchor in (:none, :right, :left) || throw(ArgumentError(
        "selection_anchor must be :none, :right, or :left",
    ))
    selection_anchor === :none ||
        (right_reference_values === nothing && left_reference_values === nothing) ||
        throw(ArgumentError("explicit reference values cannot be combined with selection_anchor"))
    right_realization = hankel_realization(
        right_moments,
        right_observer,
        moment_count;
        ranktol=ranktol,
        maxrank=maxrank,
        fixed_rank=fixed_rank,
    )
    left_realization = hankel_realization(
        left_moments,
        left_observer,
        moment_count;
        ranktol=ranktol,
        maxrank=maxrank,
        fixed_rank=fixed_rank,
    )
    right_singulars = right_realization.singular_values
    left_singulars = left_realization.singular_values
    right_detected_rank = numerical_rank(right_singulars, ranktol, maxrank)
    left_detected_rank = numerical_rank(left_singulars, ranktol, maxrank)
    right_interior, left_interior = if selection_anchor === :right
        selected_right = interior_state_realization(
            chart,
            right_realization;
            restrict_to_chart=restrict_to_chart,
            target_count=target_count,
        )
        selected_left = interior_state_realization(
            chart,
            left_realization;
            adjoint_coordinate_state=true,
            restrict_to_chart=false,
            reference_values=selected_right.values,
        )
        selected_right, selected_left
    elseif selection_anchor === :left
        selected_left = interior_state_realization(
            chart,
            left_realization;
            adjoint_coordinate_state=true,
            restrict_to_chart=restrict_to_chart,
            target_count=target_count,
        )
        selected_right = interior_state_realization(
            chart,
            right_realization;
            restrict_to_chart=false,
            reference_values=selected_left.values,
        )
        selected_right, selected_left
    else
        selected_right = interior_state_realization(
            chart,
            right_realization;
            restrict_to_chart=restrict_to_chart,
            target_count=target_count,
            reference_values=right_reference_values,
        )
        selected_left = interior_state_realization(
            chart,
            left_realization;
            adjoint_coordinate_state=true,
            restrict_to_chart=restrict_to_chart,
            target_count=target_count,
            reference_values=left_reference_values,
        )
        selected_right, selected_left
    end
    right_interior.count == left_interior.count || throw(DimensionMismatch(
        "right/left interior state counts disagree: " *
        "right=$(right_interior.count), left=$(left_interior.count)",
    ))
    rank = right_interior.count
    right_state = right_interior.state
    left_state = left_interior.state
    right = right_interior.output
    left = left_interior.output
    IndependentStateRealization(
        right_state,
        left_state,
        right,
        left,
        right_residual(right, right_state),
        left_residual(left, left_state),
        Float64.(right_singulars),
        Float64.(left_singulars),
        right_detected_rank,
        left_detected_rank,
        rank,
    )
end

function independent_polynomial_realization(coefficients, args...; kwargs...)
    independent_state_realization(
        args...;
        right_residual=(right, state) -> polynomial_invariant_residual(
            coefficients,
            right,
            state,
        ),
        left_residual=(left, state) -> polynomial_left_invariant_residual(
            coefficients,
            left,
            state,
        ),
        kwargs...,
    )
end

function independent_structured_realization(coefficients, functions, args...; kwargs...)
    independent_state_realization(
        args...;
        right_residual=(right, state) -> structured_invariant_residual(
            coefficients,
            functions,
            right,
            state,
        ),
        left_residual=(left, state) -> structured_left_invariant_residual(
            coefficients,
            functions,
            left,
            state,
        ),
        kwargs...,
    )
end
