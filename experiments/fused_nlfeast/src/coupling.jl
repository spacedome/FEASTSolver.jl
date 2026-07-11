struct LoewnerDiagnostics
    attempted::Bool
    accepted::Bool
    reason::Symbol
    singular_values::Vector{Float64}
    condition_number::Float64
    intertwining_defect::Float64
    pencil_consistency_defect::Float64
    independent_residual::Float64
    candidate_residual::Float64
    gram::Matrix{ComplexF64}
    action::Matrix{ComplexF64}
    rank::Int
end

function disabled_loewner_diagnostics(max_residual; reason=:disabled)
    LoewnerDiagnostics(
        false,
        false,
        reason,
        Float64[],
        NaN,
        NaN,
        NaN,
        max_residual,
        NaN,
        zeros(ComplexF64, 0, 0),
        zeros(ComplexF64, 0, 0),
        0,
    )
end

function modal_loewner_pencil(
    T,
    divided_difference,
    right_values,
    right,
    left_values,
    left,
    ;
    divided_overlap=nothing,
)
    count = length(right_values)
    length(left_values) == count || throw(DimensionMismatch(
        "right and left Loewner data must contain the same number of values",
    ))
    size(right, 2) == count || throw(DimensionMismatch("one right vector is required per right value"))
    size(left, 2) == count || throw(DimensionMismatch("one left vector is required per left value"))
    size(right, 1) == size(left, 1) || throw(DimensionMismatch(
        "right and left vectors must have the same physical dimension",
    ))

    right_residual = zeros(ComplexF64, size(right))
    left_residual = zeros(ComplexF64, size(left))
    for j in eachindex(right_values)
        right_residual[:, j] .= Matrix{ComplexF64}(T(right_values[j])) * view(right, :, j)
    end
    for i in eachindex(left_values)
        left_residual[:, i] .= adjoint(Matrix{ComplexF64}(T(left_values[i]))) * view(left, :, i)
    end

    G = if divided_overlap === nothing
        divided_difference === nothing && throw(ArgumentError(
            "common coupling requires divided_overlap or divided_difference",
        ))
        gram = zeros(ComplexF64, count, count)
        for i in eachindex(left_values), j in eachindex(right_values)
            divided = Matrix{ComplexF64}(divided_difference(left_values[i], right_values[j]))
            size(divided) == (size(right, 1), size(right, 1)) || throw(DimensionMismatch(
                "divided_difference must return one square matrix in the physical dimension",
            ))
            all(isfinite, divided) || throw(ArgumentError(
                "divided_difference returned a nonfinite matrix",
            ))
            gram[i, j] = dot(view(left, :, i), divided * view(right, :, j))
        end
        gram
    else
        gram = Matrix{ComplexF64}(divided_overlap(
            left_values,
            left,
            right_values,
            right,
        ))
        size(gram) == (count, count) || throw(DimensionMismatch(
            "divided_overlap must return one square matrix in the realization dimension",
        ))
        all(isfinite, gram) || throw(ArgumentError("divided_overlap returned a nonfinite matrix"))
        gram
    end

    K = Diagonal(ComplexF64.(left_values))
    S = Diagonal(ComplexF64.(right_values))
    from_right = G * S - adjoint(left) * right_residual
    from_left = K * G - adjoint(left_residual) * right
    C = (from_right + from_left) / 2
    consistency_scale = max(norm(from_right), norm(from_left), eps(Float64))
    pencil_consistency_defect = norm(from_right - from_left) / consistency_scale
    intertwining_scale = max(norm(K * G), norm(G * S), eps(Float64))
    intertwining_defect = norm(K * G - G * S) / intertwining_scale

    (
        gram=G,
        action=C,
        right_residual=right_residual,
        left_residual=left_residual,
        intertwining_defect=Float64(intertwining_defect),
        pencil_consistency_defect=Float64(pencil_consistency_defect),
    )
end

function residual_completed_overlap(
    T,
    divided_difference,
    right_values,
    right,
    left_values,
    left;
    separation_tolerance,
    divided_entry=nothing,
)
    separation_tolerance >= 0 || throw(ArgumentError("separation_tolerance must be nonnegative"))
    count = length(right_values)
    length(left_values) == count || throw(DimensionMismatch(
        "right and left overlap data must contain the same number of values",
    ))
    right_residual = hcat([
        Matrix{ComplexF64}(T(right_values[j])) * view(right, :, j)
        for j in eachindex(right_values)
    ]...)
    left_residual = hcat([
        adjoint(Matrix{ComplexF64}(T(left_values[i]))) * view(left, :, i)
        for i in eachindex(left_values)
    ]...)
    numerator = adjoint(left_residual) * right - adjoint(left) * right_residual
    gram = similar(numerator)
    completed = falses(count, count)
    for i in eachindex(left_values), j in eachindex(right_values)
        difference = left_values[i] - right_values[j]
        scale = max(1.0, abs(left_values[i]), abs(right_values[j]))
        completed[i, j] = abs(difference) <= separation_tolerance * scale
        if completed[i, j]
            gram[i, j] = if divided_entry === nothing
                divided_difference === nothing && throw(ArgumentError(
                    "cluster completion requires divided_entry or divided_difference",
                ))
                divided = Matrix{ComplexF64}(divided_difference(left_values[i], right_values[j]))
                size(divided) == (size(right, 1), size(right, 1)) || throw(DimensionMismatch(
                    "divided_difference must return one square matrix in the physical dimension",
                ))
                dot(view(left, :, i), divided * view(right, :, j))
            else
                ComplexF64(divided_entry(
                    left_values[i],
                    view(left, :, i),
                    right_values[j],
                    view(right, :, j),
                ))
            end
        else
            gram[i, j] = numerator[i, j] / difference
        end
    end
    (gram=gram, completed=completed, right_residual=right_residual, left_residual=left_residual)
end

function balance_dual_columns!(right, left)
    for j in axes(right, 2)
        right_norm = norm(view(right, :, j))
        left_norm = norm(view(left, :, j))
        if iszero(right_norm) || iszero(left_norm)
            continue
        end
        scale = sqrt(left_norm / right_norm)
        right[:, j] .*= scale
        left[:, j] ./= scale
    end
    right, left
end

function balanced_loewner_vectors(data, right, left)
    decomposition = svd(data.gram)
    root_inverse = Diagonal(1.0 ./ sqrt.(decomposition.S))
    right_gauge = decomposition.V * root_inverse
    left_gauge = decomposition.U * root_inverse
    balanced_action = adjoint(left_gauge) * data.action * right_gauge
    state_eigen = eigen(balanced_action)
    right_vectors = right * right_gauge * state_eigen.vectors
    left_vectors = left * left_gauge * inv(state_eigen.vectors)'
    (
        values=ComplexF64.(state_eigen.values),
        right=Matrix{ComplexF64}(right_vectors),
        left=Matrix{ComplexF64}(left_vectors),
        singular_values=Float64.(decomposition.S),
        balanced_action=Matrix{ComplexF64}(balanced_action),
    )
end

function coupled_candidate(
    T,
    chart,
    divided_difference,
    divided_overlap,
    right_values,
    right_vectors,
    left_values,
    left_vectors,
    independent_residual,
    residual_scale;
    residual_metric,
    ranktol,
    acceptance_ratio,
)
    data = modal_loewner_pencil(
        T,
        divided_difference,
        right_values,
        right_vectors,
        left_values,
        left_vectors,
        divided_overlap=divided_overlap,
    )
    singular_values = svdvals(data.gram)
    condition_number = if isempty(singular_values) || iszero(singular_values[end])
        Inf
    else
        singular_values[1] / singular_values[end]
    end
    full_rank = !isempty(singular_values) && !iszero(singular_values[1]) &&
        singular_values[end] >= ranktol * singular_values[1]
    if !full_rank
        diagnostics = LoewnerDiagnostics(
            true,
            false,
            :rank_deficient,
            Float64.(singular_values),
            Float64(condition_number),
            data.intertwining_defect,
            data.pencil_consistency_defect,
            independent_residual,
            Inf,
            data.gram,
            data.action,
            numerical_rank(singular_values, ranktol, length(right_values)),
        )
        return nothing, diagnostics
    end

    candidate = balanced_loewner_vectors(data, right_vectors, left_vectors)
    inside = findall(value -> in_chart(chart, value), candidate.values)
    values = candidate.values[inside]
    right = candidate.right[:, inside]
    left = candidate.left[:, inside]
    balance_dual_columns!(right, left)
    right_residuals = relative_residuals(
        T,
        values,
        right,
        residual_scale;
        residual_metric=residual_metric,
    )
    left_residuals = relative_residuals(
        T,
        values,
        left,
        residual_scale;
        adjoint_operator=true,
        residual_metric=residual_metric,
    )
    residuals = max.(right_residuals, left_residuals)
    candidate_residual = isempty(residuals) ? Inf : maximum(residuals)
    finite_candidate = all(isfinite, values) && all(isfinite, right) && all(isfinite, left) &&
        all(isfinite, residuals)
    same_count = length(values) == length(right_values)
    improves = candidate_residual <= acceptance_ratio * independent_residual
    accepted = same_count && finite_candidate && improves
    reason = accepted ? :accepted : !same_count ? :count_changed :
        !finite_candidate ? :nonfinite : :insufficient_improvement
    diagnostics = LoewnerDiagnostics(
        true,
        accepted,
        reason,
        candidate.singular_values,
        Float64(condition_number),
        data.intertwining_defect,
        data.pencil_consistency_defect,
        independent_residual,
        Float64(candidate_residual),
        data.gram,
        data.action,
        length(singular_values),
    )
    extraction = (
        values=ComplexF64.(values),
        right=Matrix{ComplexF64}(right),
        left=Matrix{ComplexF64}(left),
        right_residuals=Float64.(right_residuals),
        left_residuals=Float64.(left_residuals),
        residuals=Float64.(residuals),
    )
    extraction, diagnostics
end
