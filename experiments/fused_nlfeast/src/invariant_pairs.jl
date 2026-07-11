function polynomial_invariant_residual(coefficients, right, state)
    size(right, 2) == size(state, 1) == size(state, 2) || throw(DimensionMismatch(
        "the right map and state must have the same realization dimension",
    ))
    residual = zeros(ComplexF64, size(right))
    power = Matrix{ComplexF64}(I, size(state, 1), size(state, 2))
    for coefficient in coefficients
        residual .+= coefficient * right * power
        power = power * state
    end
    residual
end

function polynomial_left_invariant_residual(coefficients, left, state)
    size(left, 2) == size(state, 1) == size(state, 2) || throw(DimensionMismatch(
        "the left map and state must have the same realization dimension",
    ))
    residual_adjoint = zeros(ComplexF64, size(state, 1), size(left, 1))
    power = Matrix{ComplexF64}(I, size(state, 1), size(state, 2))
    for coefficient in coefficients
        residual_adjoint .+= power * adjoint(left) * coefficient
        power = power * state
    end
    Matrix{ComplexF64}(adjoint(residual_adjoint))
end

function polynomial_state_divided_overlap(
    coefficients,
    left_state,
    left,
    right_state,
    right,
)
    count = size(right_state, 1)
    size(right_state) == (count, count) || throw(DimensionMismatch("right state must be square"))
    size(left_state) == (count, count) || throw(DimensionMismatch(
        "left and right states must have the same size",
    ))
    size(right, 2) == count || throw(DimensionMismatch("right map has the wrong width"))
    size(left, 2) == count || throw(DimensionMismatch("left map has the wrong width"))
    overlap = zeros(ComplexF64, count, count)
    for degree in 1:(length(coefficients) - 1)
        projected = adjoint(left) * coefficients[degree + 1] * right
        for left_degree in 0:(degree - 1)
            overlap .+= left_state^left_degree * projected *
                right_state^(degree - 1 - left_degree)
        end
    end
    overlap
end

function matrix_divided_action(function_value, left_state, direction, right_state)
    left_count = size(left_state, 1)
    right_count = size(right_state, 1)
    size(left_state) == (left_count, left_count) || throw(DimensionMismatch(
        "left state must be square",
    ))
    size(right_state) == (right_count, right_count) || throw(DimensionMismatch(
        "right state must be square",
    ))
    size(direction) == (left_count, right_count) || throw(DimensionMismatch(
        "divided-difference direction has the wrong size",
    ))
    block = zeros(ComplexF64, left_count + right_count, left_count + right_count)
    block[1:left_count, 1:left_count] .= left_state
    block[1:left_count, (left_count + 1):end] .= direction
    block[(left_count + 1):end, (left_count + 1):end] .= right_state
    evaluated = Matrix{ComplexF64}(function_value(block))
    size(evaluated) == size(block) || throw(DimensionMismatch(
        "a structured scalar function must preserve matrix size",
    ))
    evaluated[1:left_count, (left_count + 1):end]
end

function structured_invariant_residual(coefficients, functions, right, state)
    length(coefficients) == length(functions) || throw(DimensionMismatch(
        "one scalar function is required per coefficient",
    ))
    residual = zeros(ComplexF64, size(right))
    for (coefficient, function_value) in zip(coefficients, functions)
        residual .+= coefficient * right * Matrix{ComplexF64}(function_value(state))
    end
    residual
end

function structured_left_invariant_residual(coefficients, functions, left, state)
    length(coefficients) == length(functions) || throw(DimensionMismatch(
        "one scalar function is required per coefficient",
    ))
    residual_adjoint = zeros(ComplexF64, size(state, 1), size(left, 1))
    for (coefficient, function_value) in zip(coefficients, functions)
        residual_adjoint .+= Matrix{ComplexF64}(function_value(state)) *
            adjoint(left) * coefficient
    end
    Matrix{ComplexF64}(adjoint(residual_adjoint))
end

function structured_state_divided_overlap(
    coefficients,
    functions,
    left_state,
    left,
    right_state,
    right,
)
    length(coefficients) == length(functions) || throw(DimensionMismatch(
        "one scalar function is required per coefficient",
    ))
    overlap = zeros(ComplexF64, size(left_state, 1), size(right_state, 1))
    for (coefficient, function_value) in zip(coefficients, functions)
        direction = adjoint(left) * coefficient * right
        overlap .+= matrix_divided_action(
            function_value,
            left_state,
            direction,
            right_state,
        )
    end
    overlap
end

function structured_eigenvalue_condition_numbers(
    coefficients,
    derivative_functions,
    values,
    right,
    left,
)
    length(coefficients) == length(derivative_functions) || throw(DimensionMismatch(
        "one derivative function is required per coefficient",
    ))
    length(values) == size(right, 2) == size(left, 2) || throw(DimensionMismatch(
        "one right and left vector is required per eigenvalue",
    ))
    size(right, 1) == size(left, 1) || throw(DimensionMismatch(
        "right and left vectors must have the same physical dimension",
    ))
    conditions = zeros(Float64, length(values))
    for index in eachindex(values)
        value = values[index]
        right_vector = view(right, :, index)
        left_vector = view(left, :, index)
        derivative_action = zeros(ComplexF64, size(right, 1))
        for (coefficient, derivative) in zip(coefficients, derivative_functions)
            derivative_action .+= derivative(value) .* (coefficient * right_vector)
        end
        denominator = abs(dot(left_vector, derivative_action))
        numerator = norm(right_vector) * norm(left_vector)
        conditions[index] = iszero(denominator) ? Inf : numerator / denominator
    end
    conditions
end

function lifted_pair_gauges(
    right_state,
    right,
    left_state,
    left,
    depth;
    center=0.0,
    radius=1.0,
)
    depth > 0 || throw(ArgumentError("lift depth must be positive"))
    radius > 0 || throw(ArgumentError("lift radius must be positive"))
    state_count = size(right_state, 1)
    identity_state = Matrix{ComplexF64}(I, state_count, state_count)
    right_dynamics = (right_state .- center .* identity_state) ./ radius
    left_dynamics = (left_state .- center .* identity_state) ./ radius
    right_blocks = Matrix{ComplexF64}[]
    left_blocks = Matrix{ComplexF64}[]
    right_power = Matrix{ComplexF64}(I, state_count, state_count)
    left_power = Matrix{ComplexF64}(I, state_count, state_count)
    for _ in 1:depth
        push!(right_blocks, right * right_power)
        push!(left_blocks, left * left_power)
        right_power = right_power * right_dynamics
        left_power = left_power * adjoint(left_dynamics)
    end
    right_lift = reduce(vcat, right_blocks)
    left_lift = reduce(vcat, left_blocks)
    right_qr = qr(right_lift)
    left_qr = qr(left_lift)
    right_R = Matrix{ComplexF64}(right_qr.R)[1:state_count, :]
    left_R = Matrix{ComplexF64}(left_qr.R)[1:state_count, :]
    right_singulars = svdvals(right_R)
    left_singulars = svdvals(left_R)
    isempty(right_singulars) || right_singulars[end] > 0 || error(
        "right invariant pair is not minimal at the requested lift depth",
    )
    isempty(left_singulars) || left_singulars[end] > 0 || error(
        "left invariant pair is not minimal at the requested lift depth",
    )
    (
        right=right_R,
        left=left_R,
        right_condition=Float64(cond(right_R)),
        left_condition=Float64(cond(left_R)),
    )
end

function structured_pair_backward_error(
    coefficients,
    functions,
    right_state,
    right,
    left_state,
    left;
    lift_depth,
    lift_center=0.0,
    lift_radius=1.0,
)
    gauges = lifted_pair_gauges(
        right_state,
        right,
        left_state,
        left,
        lift_depth;
        center=lift_center,
        radius=lift_radius,
    )
    right_inverse = inv(gauges.right)
    left_inverse = inv(gauges.left)
    canonical_right = right * right_inverse
    canonical_right_state = gauges.right * right_state * right_inverse
    canonical_left = left * left_inverse
    canonical_left_state = adjoint(gauges.left) \ (
        left_state * adjoint(gauges.left)
    )
    right_residual = structured_invariant_residual(
        coefficients,
        functions,
        canonical_right,
        canonical_right_state,
    )
    left_residual = structured_left_invariant_residual(
        coefficients,
        functions,
        canonical_left,
        canonical_left_state,
    )
    coefficient_norms = norm.(coefficients)
    right_scale = sum(
        coefficient_norms[k] * norm(canonical_right * functions[k](canonical_right_state))
        for k in eachindex(functions)
    )
    left_scale = sum(
        coefficient_norms[k] * norm(
            canonical_left * adjoint(functions[k](canonical_left_state)),
        )
        for k in eachindex(functions)
    )
    coefficient_scale = sum(coefficient_norms)
    right_reference = coefficient_scale * norm(canonical_right)
    left_reference = coefficient_scale * norm(canonical_left)
    right_scale_degenerate = right_scale <= sqrt(eps(Float64)) * right_reference
    left_scale_degenerate = left_scale <= sqrt(eps(Float64)) * left_reference
    right_error = right_scale_degenerate ? Inf : norm(right_residual) / right_scale
    left_error = left_scale_degenerate ? Inf : norm(left_residual) / left_scale
    (
        right=Float64(right_error),
        left=Float64(left_error),
        maximum=Float64(max(right_error, left_error)),
        right_lift_condition=gauges.right_condition,
        left_lift_condition=gauges.left_condition,
        scale_degenerate=right_scale_degenerate || left_scale_degenerate,
    )
end

function lifted_residual_error(
    right_state,
    right,
    right_residual,
    left_state,
    left,
    left_residual;
    lift_depth,
    operator_scale=1.0,
    lift_center=0.0,
    lift_radius=1.0,
)
    operator_scale > 0 || throw(ArgumentError("operator_scale must be positive"))
    gauges = lifted_pair_gauges(
        right_state,
        right,
        left_state,
        left,
        lift_depth;
        center=lift_center,
        radius=lift_radius,
    )
    state_count = size(right_state, 1)
    normalization = operator_scale * sqrt(state_count)
    right_error = norm(right_residual / gauges.right) / normalization
    left_error = norm(left_residual / gauges.left) / normalization
    (
        right=Float64(right_error),
        left=Float64(left_error),
        maximum=Float64(max(right_error, left_error)),
        right_lift_condition=gauges.right_condition,
        left_lift_condition=gauges.left_condition,
    )
end

function lift_normalized_components(
    right_state,
    right,
    right_residual,
    left_state,
    left,
    left_residual;
    lift_depth,
    lift_center=0.0,
    lift_radius=1.0,
)
    gauges = lifted_pair_gauges(
        right_state,
        right,
        left_state,
        left,
        lift_depth;
        center=lift_center,
        radius=lift_radius,
    )
    right_inverse = inv(gauges.right)
    left_inverse = inv(gauges.left)
    (
        right_state=Matrix{ComplexF64}(gauges.right * right_state * right_inverse),
        left_state=Matrix{ComplexF64}(
            adjoint(gauges.left) \ (left_state * adjoint(gauges.left)),
        ),
        right=Matrix{ComplexF64}(right * right_inverse),
        left=Matrix{ComplexF64}(left * left_inverse),
        right_residual=Matrix{ComplexF64}(right_residual * right_inverse),
        left_residual=Matrix{ComplexF64}(left_residual * left_inverse),
        right_lift_condition=gauges.right_condition,
        left_lift_condition=gauges.left_condition,
    )
end
