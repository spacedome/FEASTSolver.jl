function cauchy_invariant_data(
    chart,
    right_state,
    right,
    left_state,
    left,
    right_action,
    left_action,
)
    state_count = size(right_state, 1)
    size(right_state) == (state_count, state_count) || throw(DimensionMismatch(
        "right state must be square",
    ))
    size(left_state) == (state_count, state_count) || throw(DimensionMismatch(
        "left and right states must have the same size",
    ))
    size(right, 2) == state_count || throw(DimensionMismatch("right map has the wrong width"))
    size(left) == size(right) || throw(DimensionMismatch(
        "right and left maps must have the same shape",
    ))
    all(value -> in_chart(chart, value), eigvals(right_state)) || throw(ArgumentError(
        "the right state must lie inside the Cauchy chart",
    ))
    all(value -> in_chart(chart, value), eigvals(left_state)) || throw(ArgumentError(
        "the left state must lie inside the Cauchy chart",
    ))

    identity_state = Matrix{ComplexF64}(I, state_count, state_count)
    right_residual = zeros(ComplexF64, size(right))
    left_residual_adjoint = zeros(ComplexF64, state_count, size(left, 1))
    overlap = zeros(ComplexF64, state_count, state_count)
    right_triangular = istriu(right_state)
    left_triangular = istriu(left_state)

    for (z, weight) in zip(chart.nodes, chart.weights)
        right_shift = z .* identity_state .- right_state
        left_shift = z .* identity_state .- left_state
        right_inverse = right_triangular ?
            UpperTriangular(right_shift) \ identity_state : right_shift \ identity_state
        left_inverse = left_triangular ?
            UpperTriangular(left_shift) \ identity_state : left_shift \ identity_state
        right_value = Matrix{ComplexF64}(right_action(z, right))
        left_value = Matrix{ComplexF64}(left_action(z, left))
        size(right_value) == size(right) || throw(DimensionMismatch(
            "right_action must preserve the input shape",
        ))
        size(left_value) == size(left) || throw(DimensionMismatch(
            "left_action must preserve the input shape",
        ))
        projected = adjoint(left) * right_value
        right_residual .+= weight .* (right_value * right_inverse)
        left_residual_adjoint .+= weight .* (left_inverse * adjoint(left_value))
        overlap .+= weight .* (left_inverse * projected * right_inverse)
    end
    (
        right_residual=right_residual,
        left_residual=Matrix{ComplexF64}(adjoint(left_residual_adjoint)),
        overlap=overlap,
    )
end

function validate_cauchy_state(chart, state)
    all(value -> in_chart(chart, value), eigvals(state)) || throw(ArgumentError(
        "the state must lie inside the Cauchy chart",
    ))
end

function cauchy_right_invariant_residual(chart, state, right, right_action)
    validate_cauchy_state(chart, state)
    count = size(state, 1)
    identity_state = Matrix{ComplexF64}(I, count, count)
    residual = zeros(ComplexF64, size(right))
    triangular = istriu(state)
    for (z, weight) in zip(chart.nodes, chart.weights)
        shift = z .* identity_state .- state
        inverse_shift = triangular ?
            UpperTriangular(shift) \ identity_state : shift \ identity_state
        residual .+= weight .* (right_action(z, right) * inverse_shift)
    end
    residual
end

function cauchy_left_invariant_residual(chart, state, left, left_action)
    validate_cauchy_state(chart, state)
    count = size(state, 1)
    identity_state = Matrix{ComplexF64}(I, count, count)
    residual_adjoint = zeros(ComplexF64, count, size(left, 1))
    triangular = istriu(state)
    for (z, weight) in zip(chart.nodes, chart.weights)
        shift = z .* identity_state .- state
        inverse_shift = triangular ?
            UpperTriangular(shift) \ identity_state : shift \ identity_state
        residual_adjoint .+= weight .* (inverse_shift * adjoint(left_action(z, left)))
    end
    Matrix{ComplexF64}(adjoint(residual_adjoint))
end

function cauchy_state_divided_overlap(
    chart,
    left_state,
    left,
    right_state,
    right,
    right_action,
)
    validate_cauchy_state(chart, left_state)
    validate_cauchy_state(chart, right_state)
    count = size(right_state, 1)
    identity_state = Matrix{ComplexF64}(I, count, count)
    overlap = zeros(ComplexF64, count, count)
    right_triangular = istriu(right_state)
    left_triangular = istriu(left_state)
    for (z, weight) in zip(chart.nodes, chart.weights)
        right_shift = z .* identity_state .- right_state
        left_shift = z .* identity_state .- left_state
        right_inverse = right_triangular ?
            UpperTriangular(right_shift) \ identity_state : right_shift \ identity_state
        left_inverse = left_triangular ?
            UpperTriangular(left_shift) \ identity_state : left_shift \ identity_state
        projected = adjoint(left) * right_action(z, right)
        overlap .+= weight .* (left_inverse * projected * right_inverse)
    end
    overlap
end
