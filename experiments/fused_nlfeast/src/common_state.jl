struct CommonStateRealization
    state::Matrix{ComplexF64}
    right::Matrix{ComplexF64}
    left::Matrix{ComplexF64}
    right_residual::Matrix{ComplexF64}
    left_residual::Matrix{ComplexF64}
    gram::Matrix{ComplexF64}
    action::Matrix{ComplexF64}
    right_singular_values::Vector{Float64}
    left_singular_values::Vector{Float64}
    overlap_singular_values::Vector{Float64}
    rank::Int
end

function balanced_common_state(gram, action, right, left; ranktol)
    decomposition = svd(gram)
    isempty(decomposition.S) && error("common state overlap is empty")
    decomposition.S[end] >= ranktol * decomposition.S[1] || error(
        "common state overlap is numerically rank deficient",
    )
    root_inverse = Diagonal(1.0 ./ sqrt.(decomposition.S))
    right_gauge = decomposition.V * root_inverse
    left_gauge = decomposition.U * root_inverse
    (
        state=Matrix{ComplexF64}(adjoint(left_gauge) * action * right_gauge),
        right=Matrix{ComplexF64}(right * right_gauge),
        left=Matrix{ComplexF64}(left * left_gauge),
        right_gauge=Matrix{ComplexF64}(right_gauge),
        left_gauge=Matrix{ComplexF64}(left_gauge),
        singular_values=Float64.(decomposition.S),
    )
end

function schur_common_state(common)
    decomposition = schur(common.state)
    gauge = Matrix{ComplexF64}(decomposition.Z)
    merge(
        common,
        (
            state=Matrix{ComplexF64}(decomposition.T),
            right=Matrix{ComplexF64}(common.right * gauge),
            left=Matrix{ComplexF64}(common.left * gauge),
            schur_gauge=gauge,
        ),
    )
end

function common_state_from_independent(
    independent::IndependentStateRealization;
    overlap_ranktol,
    right_residual,
    left_residual,
    state_overlap,
)
    right_state = independent.right_state
    left_state = independent.left_state
    right = independent.right
    left = independent.left
    right_residual_value = independent.right_residual
    left_residual_value = independent.left_residual
    gram = state_overlap(left_state, left, right_state, right)
    from_right = gram * right_state - adjoint(left) * right_residual_value
    from_left = left_state * gram - adjoint(left_residual_value) * right
    action = (from_right + from_left) / 2
    common = schur_common_state(
        balanced_common_state(gram, action, right, left; ranktol=overlap_ranktol),
    )
    CommonStateRealization(
        common.state,
        common.right,
        common.left,
        right_residual(common.right, common.state),
        left_residual(common.left, common.state),
        gram,
        action,
        independent.right_singular_values,
        independent.left_singular_values,
        common.singular_values,
        independent.rank,
    )
end

function common_state_realization(
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
    overlap_ranktol=ranktol,
    right_residual,
    left_residual,
    state_overlap,
)
    independent = independent_state_realization(
        chart,
        right_moments,
        left_moments,
        right_observer,
        left_observer,
        moment_count;
        ranktol=ranktol,
        maxrank=maxrank,
        fixed_rank=fixed_rank,
        restrict_to_chart=restrict_to_chart,
        target_count=target_count,
        right_reference_values=right_reference_values,
        left_reference_values=left_reference_values,
        selection_anchor=selection_anchor,
        right_residual=right_residual,
        left_residual=left_residual,
    )
    independent = lift_normalized_independent_state(
        independent,
        moment_count;
        center=chart.center,
        radius=chart.radius,
    )
    common_state_from_independent(
        independent;
        overlap_ranktol=overlap_ranktol,
        right_residual=right_residual,
        left_residual=left_residual,
        state_overlap=state_overlap,
    )
end

function common_polynomial_realization(
    coefficients,
    chart,
    right_moments,
    left_moments,
    right_observer,
    left_observer,
    moment_count;
    kwargs...,
)
    common_state_realization(
        chart,
        right_moments,
        left_moments,
        right_observer,
        left_observer,
        moment_count;
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
        state_overlap=(left_state, left, right_state, right) ->
            polynomial_state_divided_overlap(
                coefficients,
                left_state,
                left,
                right_state,
                right,
            ),
        kwargs...,
    )
end

function common_structured_realization(
    coefficients,
    functions,
    chart,
    right_moments,
    left_moments,
    right_observer,
    left_observer,
    moment_count;
    kwargs...,
)
    common_state_realization(
        chart,
        right_moments,
        left_moments,
        right_observer,
        left_observer,
        moment_count;
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
        state_overlap=(left_state, left, right_state, right) ->
            structured_state_divided_overlap(
                coefficients,
                functions,
                left_state,
                left,
                right_state,
                right,
            ),
        kwargs...,
    )
end
