struct TwoSidedHankelRealization
    state::Matrix{ComplexF64}
    right::Matrix{ComplexF64}
    left::Matrix{ComplexF64}
    right_residual::Matrix{ComplexF64}
    left_residual::Matrix{ComplexF64}
    singular_values::Vector{Float64}
    rank::Int
    detected_rank::Int
    restriction_condition::Float64
end

function common_spectral_restriction(
    chart,
    state,
    right,
    left;
    restrict_to_chart=true,
    target_count=nothing,
)
    count = size(state, 1)
    size(state) == (count, count) || throw(DimensionMismatch("state must be square"))
    decomposition = schur(state)
    selected = if !restrict_to_chart
        trues(count)
    elseif target_count === nothing
        BitVector(in_chart(chart, value) for value in decomposition.values)
    else
        requested_count = Int(target_count)
        requested_count > 0 || throw(ArgumentError("target state count must be positive"))
        inside = findall(value -> in_chart(chart, value), decomposition.values)
        length(inside) >= requested_count || throw(DimensionMismatch(
            "common realization has only $(length(inside)) interior states for " *
            "target count $requested_count",
        ))
        inward_order = sort(inside; by=index -> chart_inward_score(chart, decomposition.values[index]))
        selection = falses(count)
        selection[inward_order[1:requested_count]] .= true
        selection
    end
    selected_count = Base.count(selected)
    selected_count > 0 || error("common realization has no states inside the chart")
    if selected_count == count
        gauge = Matrix{ComplexF64}(decomposition.Z)
        return (
            state=Matrix{ComplexF64}(decomposition.T),
            right=Matrix{ComplexF64}(right * gauge),
            left=Matrix{ComplexF64}(left * gauge),
            count=selected_count,
            condition=1.0,
        )
    end

    ordered = ordschur(decomposition, selected)
    target = Matrix{ComplexF64}(ordered.T[1:selected_count, 1:selected_count])
    exterior = Matrix{ComplexF64}(ordered.T[(selected_count + 1):end, (selected_count + 1):end])
    coupling = Matrix{ComplexF64}(ordered.T[1:selected_count, (selected_count + 1):end])
    decoupling = sylvester(target, -exterior, coupling)
    block_gauge = Matrix{ComplexF64}(I, count, count)
    block_gauge[1:selected_count, (selected_count + 1):end] .= decoupling
    gauge = Matrix{ComplexF64}(ordered.Z * block_gauge)
    dual_gauge = Matrix{ComplexF64}(adjoint(inv(gauge)))
    (
        state=target,
        right=Matrix{ComplexF64}(right * gauge[:, 1:selected_count]),
        left=Matrix{ComplexF64}(left * dual_gauge[:, 1:selected_count]),
        count=selected_count,
        condition=Float64(cond(gauge)),
    )
end

function two_sided_hankel_realization(
    chart,
    right_moments,
    left_moments,
    right_observer,
    moment_count;
    ranktol,
    maxrank=typemax(Int),
    fixed_rank=nothing,
    restrict_to_chart=true,
    target_count=nothing,
    right_residual,
    left_residual,
    state_overlap=nothing,
    overlap_ranktol=ranktol,
)
    H0, H1 = block_hankel(right_moments, moment_count; observer=right_observer)
    decomposition = svd(H0)
    detected_rank = numerical_rank(decomposition.S, ranktol, maxrank)
    rank = fixed_rank === nothing ? detected_rank : min(Int(fixed_rank), length(decomposition.S), maxrank)
    rank > 0 || error("two-sided moment Hankel matrix has numerical rank zero")
    U = decomposition.U[:, 1:rank]
    V = decomposition.V[:, 1:rank]
    inverse_singulars = Diagonal(1.0 ./ decomposition.S[1:rank])
    coordinate_state = adjoint(U) * H1 * V * inverse_singulars
    right_row = reduce(hcat, right_moments[1:moment_count])
    left_row = reduce(hcat, left_moments[1:moment_count])
    size(left_row, 2) == size(U, 1) || throw(DimensionMismatch(
        "left moment width must match the cross-Hankel observation width",
    ))
    right = right_row * V * inverse_singulars
    left = left_row * U
    state = chart.center .* Matrix{ComplexF64}(I, rank, rank) .+
        chart.radius .* coordinate_state
    restricted = common_spectral_restriction(
        chart,
        state,
        right,
        left;
        restrict_to_chart=restrict_to_chart,
        target_count=target_count,
    )
    common = if state_overlap === nothing
        (
            state=restricted.state,
            right=restricted.right,
            left=restricted.left,
        )
    else
        right_residual_value = right_residual(restricted.right, restricted.state)
        left_residual_value = left_residual(restricted.left, restricted.state)
        gram = state_overlap(
            restricted.state,
            restricted.left,
            restricted.state,
            restricted.right,
        )
        from_right = gram * restricted.state -
            adjoint(restricted.left) * right_residual_value
        from_left = restricted.state * gram -
            adjoint(left_residual_value) * restricted.right
        action = (from_right + from_left) / 2
        schur_common_state(balanced_common_state(
            gram,
            action,
            restricted.right,
            restricted.left;
            ranktol=overlap_ranktol,
        ))
    end
    TwoSidedHankelRealization(
        common.state,
        common.right,
        common.left,
        right_residual(common.right, common.state),
        left_residual(common.left, common.state),
        Float64.(decomposition.S),
        restricted.count,
        detected_rank,
        restricted.condition,
    )
end

function two_sided_structured_realization(
    coefficients,
    functions,
    chart,
    right_moments,
    left_moments,
    right_observer,
    moment_count;
    kwargs...,
)
    two_sided_hankel_realization(
        chart,
        right_moments,
        left_moments,
        right_observer,
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
