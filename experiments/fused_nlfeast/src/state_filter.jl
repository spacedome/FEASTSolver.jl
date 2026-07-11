function state_residual_factors(right_residual, left_residual; ranktol)
    right_basis, right_coefficients, right_singulars = low_rank_factor(
        right_residual;
        ranktol=ranktol,
    )
    left_basis, left_coefficients, left_singulars = low_rank_factor(
        left_residual;
        ranktol=ranktol,
    )
    (
        right_basis=right_basis,
        right_coefficients=right_coefficients,
        right_singulars=right_singulars,
        left_basis=left_basis,
        left_coefficients=left_coefficients,
        left_singulars=left_singulars,
    )
end

function state_moment_tangent(state, moment_count, requested_width; side, ranktol)
    moment_count > 0 || throw(ArgumentError("moment_count must be positive"))
    requested_width > 0 || throw(ArgumentError("state tangent width must be positive"))
    side in (:right, :left) || throw(ArgumentError("side must be :right or :left"))
    ranktol > 0 || throw(ArgumentError("state tangent rank tolerance must be positive"))
    state_count = size(state, 1)
    size(state) == (state_count, state_count) || throw(DimensionMismatch("state must be square"))
    width = min(requested_width, state_count)
    moment_count * width >= state_count || throw(ArgumentError(
        "state tangent width $width and moment depth $moment_count cannot span $state_count states",
    ))
    direction_sign = side === :right ? 1 : -1
    tangent = ComplexF64[
        cis(direction_sign * 2pi * (row - 1) * (column - 0.5) / state_count) / sqrt(state_count)
        for row in 1:state_count, column in 1:width
    ]
    dynamics = side === :right ? state : adjoint(state)
    controllability = zeros(ComplexF64, state_count, moment_count * width)
    block = copy(tangent)
    for k in 1:moment_count
        columns = ((k - 1) * width + 1):(k * width)
        controllability[:, columns] .= block
        block = dynamics * block
    end
    singular_values = svdvals(controllability)
    controllability_rank = numerical_rank(singular_values, ranktol, state_count)
    controllability_rank == state_count || throw(ArgumentError(
        "state tangent is not controllable at width $width and depth $moment_count: " *
        "rank=$controllability_rank, state_count=$state_count",
    ))
    tangent
end

function state_corrected_moments(
    cache::ContourSampleCache,
    right_state,
    left_state,
    right,
    left,
    right_id::Symbol,
    left_id::Symbol,
    right_coefficients,
    left_coefficients,
    count::Integer;
    right_tangent=nothing,
    left_tangent=nothing,
)
    count > 0 || throw(ArgumentError("moment count must be positive"))
    state_count = size(right_state, 1)
    size(right_state) == (state_count, state_count) || throw(DimensionMismatch(
        "right state must be square",
    ))
    size(left_state) == (state_count, state_count) || throw(DimensionMismatch(
        "left and right states must have the same size",
    ))
    size(right, 2) == state_count || throw(DimensionMismatch("right map has the wrong width"))
    size(left, 2) == state_count || throw(DimensionMismatch("left map has the wrong width"))
    right_tangent === nothing || size(right_tangent, 1) == state_count ||
        throw(DimensionMismatch("right tangent has the wrong height"))
    left_tangent === nothing || size(left_tangent, 1) == state_count ||
        throw(DimensionMismatch("left tangent has the wrong height"))

    right_block = cache.right_blocks[right_id]
    left_block = cache.left_blocks[left_id]
    right_block.role === :residual || throw(ArgumentError("right block must be residual data"))
    left_block.role === :residual || throw(ArgumentError("left block must be residual data"))
    right_direction = right_tangent === nothing ?
        Matrix{ComplexF64}(I, state_count, state_count) : Matrix{ComplexF64}(right_tangent)
    left_direction = left_tangent === nothing ?
        Matrix{ComplexF64}(I, state_count, state_count) : Matrix{ComplexF64}(left_tangent)
    right_moments = [
        zeros(ComplexF64, size(right, 1), size(right_direction, 2)) for _ in 1:count
    ]
    left_moments = [
        zeros(ComplexF64, size(left, 1), size(left_direction, 2)) for _ in 1:count
    ]
    identity_state = Matrix{ComplexF64}(I, state_count, state_count)
    right_triangular = istriu(right_state)
    left_triangular = istriu(left_state)

    for node_index in eachindex(cache.chart.nodes)
        z = cache.chart.nodes[node_index]
        weight = cache.chart.weights[node_index]
        coordinate = chart_coordinate(cache.chart, z)
        right_shift = z .* identity_state .- right_state
        left_shift = conj(z) .* identity_state .- adjoint(left_state)
        right_rational = right_triangular ?
            UpperTriangular(right_shift) \ right_direction : right_shift \ right_direction
        left_rational = left_triangular ?
            LowerTriangular(left_shift) \ left_direction : left_shift \ left_direction
        right_corrected = right * right_rational -
            right_block.responses[node_index] * (right_coefficients * right_rational)
        left_corrected = left * left_rational -
            left_block.responses[node_index] * (left_coefficients * left_rational)
        right_power = one(ComplexF64)
        left_power = one(ComplexF64)
        for k in 1:count
            right_moments[k] .+= (weight * right_power) .* right_corrected
            left_moments[k] .+= (conj(weight) * left_power) .* left_corrected
            right_power *= coordinate
            left_power *= conj(coordinate)
        end
    end
    right_moments, left_moments
end
