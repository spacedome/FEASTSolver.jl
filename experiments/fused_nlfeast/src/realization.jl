struct MomentRealization
    output::Matrix{ComplexF64}
    state::Matrix{ComplexF64}
    singular_values::Vector{Float64}
    rank::Int
end

struct FusedExtraction
    values::Vector{ComplexF64}
    right_interpolation_values::Vector{ComplexF64}
    left_interpolation_values::Vector{ComplexF64}
    right::Matrix{ComplexF64}
    left::Matrix{ComplexF64}
    right_residuals::Vector{Float64}
    left_residuals::Vector{Float64}
    residuals::Vector{Float64}
    match_errors::Vector{Float64}
    rank::Int
    right_singular_values::Vector{Float64}
    left_singular_values::Vector{Float64}
    coupling::LoewnerDiagnostics
    selection_mode::Symbol
end

function block_hankel(moments, moment_count; observer=nothing)
    length(moments) >= 2 * moment_count || throw(ArgumentError("two moment blocks per Hankel order are required"))
    width = size(moments[1], 2)
    projected = observer === nothing ? moments : [adjoint(observer) * moment for moment in moments]
    observed_rows = size(projected[1], 1)
    H0 = zeros(ComplexF64, moment_count * observed_rows, moment_count * width)
    H1 = similar(H0)
    for row in 1:moment_count, column in 1:moment_count
        rows = (row - 1) * observed_rows + 1:row * observed_rows
        columns = (column - 1) * width + 1:column * width
        H0[rows, columns] .= projected[row + column - 1]
        H1[rows, columns] .= projected[row + column]
    end
    H0, H1
end

function numerical_rank(singular_values, ranktol, maxrank)
    isempty(singular_values) && return 0
    count = Base.count(singular_values ./ singular_values[1] .> ranktol)
    min(count, maxrank, length(singular_values))
end

function realization_rank(singular_values, ranktol, maxrank, fixed_rank)
    fixed_rank === nothing && return numerical_rank(singular_values, ranktol, maxrank)
    requested_rank = Int(fixed_rank)
    requested_rank > 0 || throw(ArgumentError("fixed realization rank must be positive"))
    requested_rank <= min(length(singular_values), maxrank) || throw(ArgumentError(
        "fixed realization rank exceeds the available Hankel dimensions",
    ))
    requested_rank
end

function hankel_realization(moments, observer, moment_count; ranktol, maxrank=typemax(Int), fixed_rank=nothing)
    H0, H1 = block_hankel(moments, moment_count; observer=observer)
    F = svd(H0)
    rank = realization_rank(F.S, ranktol, maxrank, fixed_rank)
    rank > 0 || error("moment Hankel matrix has numerical rank zero")
    U = F.U[:, 1:rank]
    V = F.V[:, 1:rank]
    inv_singulars = Diagonal(1.0 ./ F.S[1:rank])
    state = U' * H1 * V * inv_singulars
    moment_row = reduce(hcat, moments[1:moment_count])
    output = moment_row * V * inv_singulars
    MomentRealization(output, state, Float64.(F.S), rank)
end

function streaming_hankel_realization(
    moments,
    moment_count;
    ranktol,
    maxrank=typemax(Int),
    fixed_rank=nothing,
)
    length(moments) >= 2 * moment_count || throw(ArgumentError(
        "two moment blocks per streaming Hankel order are required",
    ))
    width = size(moments[1], 2)
    reduced_R = zeros(ComplexF64, 0, moment_count * width)
    reduced_shift = zeros(ComplexF64, 0, moment_count * width)
    for row in 1:moment_count
        H0_row = reduce(hcat, moments[row:(row + moment_count - 1)])
        H1_row = reduce(hcat, moments[(row + 1):(row + moment_count)])
        stacked_H0 = vcat(reduced_R, H0_row)
        stacked_H1 = vcat(reduced_shift, H1_row)
        decomposition = qr(stacked_H0)
        retained_rows = min(size(stacked_H0)...)
        reduced_R = Matrix{ComplexF64}(decomposition.R)[1:retained_rows, :]
        transformed_shift = adjoint(decomposition.Q) * stacked_H1
        reduced_shift = Matrix{ComplexF64}(transformed_shift[1:retained_rows, :])
    end
    decomposition = svd(reduced_R)
    rank = realization_rank(decomposition.S, ranktol, maxrank, fixed_rank)
    rank > 0 || error("streaming moment Hankel matrix has numerical rank zero")
    U = decomposition.U[:, 1:rank]
    V = decomposition.V[:, 1:rank]
    inverse_singulars = Diagonal(1.0 ./ decomposition.S[1:rank])
    state = adjoint(U) * reduced_shift * V * inverse_singulars
    moment_row = reduce(hcat, moments[1:moment_count])
    output = moment_row * V * inverse_singulars
    MomentRealization(output, state, Float64.(decomposition.S), rank)
end

function interior_state_realization(
    chart,
    realization;
    adjoint_coordinate_state=false,
    restrict_to_chart=true,
    target_count=nothing,
    reference_values=nothing,
)
    count = realization.rank
    identity_state = Matrix{ComplexF64}(I, count, count)
    coordinate_state = adjoint_coordinate_state ? adjoint(realization.state) : realization.state
    physical_state = chart.center .* identity_state .+ chart.radius .* coordinate_state
    decomposition = schur(physical_state)
    selected = if reference_values !== nothing
        references = ComplexF64.(reference_values)
        isempty(references) && throw(ArgumentError("reference state values must not be empty"))
        candidates = restrict_to_chart ?
            findall(value -> in_chart(chart, value), decomposition.values) :
            collect(eachindex(decomposition.values))
        length(candidates) >= length(references) || throw(DimensionMismatch(
            "moment realization has only $(length(candidates)) candidate states for " *
            "$(length(references)) reference values",
        ))
        assignment, _ = bottleneck_subset_match(
            references,
            decomposition.values[candidates],
        )
        selection = falses(count)
        selection[candidates[assignment]] .= true
        selection
    elseif !restrict_to_chart
        trues(count)
    elseif target_count === nothing
        BitVector(in_chart(chart, value) for value in decomposition.values)
    else
        requested_count = Int(target_count)
        requested_count > 0 || throw(ArgumentError("target state count must be positive"))
        inside = findall(value -> in_chart(chart, value), decomposition.values)
        length(inside) >= requested_count || throw(DimensionMismatch(
            "moment realization has only $(length(inside)) interior states for target count $requested_count",
        ))
        inward_order = sort(inside; by=index -> chart_inward_score(chart, decomposition.values[index]))
        selection = falses(count)
        selection[inward_order[1:requested_count]] .= true
        selection
    end
    interior_count = Base.count(selected)
    interior_count > 0 || error("moment realization has no states inside the chart")
    ordered, indices = if interior_count == count
        decomposition, 1:count
    elseif adjoint_coordinate_state
        left_ordered = ordschur(decomposition, .!selected)
        left_ordered, (count - interior_count + 1):count
    else
        right_ordered = ordschur(decomposition, selected)
        right_ordered, 1:interior_count
    end
    gauge = Matrix{ComplexF64}(ordered.Z[:, indices])
    (
        state=Matrix{ComplexF64}(ordered.T[indices, indices]),
        output=Matrix{ComplexF64}(realization.output * gauge),
        count=interior_count,
        total_count=count,
        values=ComplexF64.(ordered.values[indices]),
    )
end

function threshold_subset_matching(costs, threshold)
    reference_count, candidate_count = size(costs)
    reference_count <= candidate_count || return nothing
    matched_reference = zeros(Int, candidate_count)

    function augment(reference_index, visited)
        for candidate_index in 1:candidate_count
            visited[candidate_index] && continue
            costs[reference_index, candidate_index] <= threshold || continue
            visited[candidate_index] = true
            if iszero(matched_reference[candidate_index]) ||
               augment(matched_reference[candidate_index], visited)
                matched_reference[candidate_index] = reference_index
                return true
            end
        end
        false
    end

    for reference_index in 1:reference_count
        augment(reference_index, falses(candidate_count)) || return nothing
    end
    assignment = zeros(Int, reference_count)
    for candidate_index in 1:candidate_count
        reference_index = matched_reference[candidate_index]
        iszero(reference_index) || (assignment[reference_index] = candidate_index)
    end
    assignment
end

function bottleneck_subset_match(reference_values, candidate_values)
    length(reference_values) <= length(candidate_values) || throw(DimensionMismatch(
        "a continuation needs at least as many candidates as reference values",
    ))
    isempty(reference_values) && return Int[], Float64[]
    costs = Float64[
        abs(reference_values[i] - candidate_values[j])
        for i in eachindex(reference_values), j in eachindex(candidate_values)
    ]
    thresholds = sort!(unique!(vec(copy(costs))))
    low = 1
    high = length(thresholds)
    while low < high
        middle = (low + high) >>> 1
        if threshold_subset_matching(costs, thresholds[middle]) === nothing
            low = middle + 1
        else
            high = middle
        end
    end
    assignment = threshold_subset_matching(costs, thresholds[low])
    assignment === nothing && error("internal bottleneck subset matching failure")
    errors = Float64[costs[i, assignment[i]] for i in eachindex(reference_values)]
    assignment, errors
end

function threshold_matching(costs, threshold)
    count = size(costs, 1)
    size(costs, 2) == count || throw(DimensionMismatch("matching costs must be square"))
    matched_right = zeros(Int, count)

    function augment(right_index, visited)
        for left_index in 1:count
            visited[left_index] && continue
            costs[right_index, left_index] <= threshold || continue
            visited[left_index] = true
            if iszero(matched_right[left_index]) || augment(matched_right[left_index], visited)
                matched_right[left_index] = right_index
                return true
            end
        end
        false
    end

    for right_index in 1:count
        augment(right_index, falses(count)) || return nothing
    end
    assignment = zeros(Int, count)
    for left_index in 1:count
        assignment[matched_right[left_index]] = left_index
    end
    assignment
end

function bottleneck_match(right_values, left_values)
    length(right_values) == length(left_values) || throw(DimensionMismatch("right and left realizations must have equal rank"))
    count = length(right_values)
    count == 0 && return Int[], Float64[]
    costs = Float64[abs(right_values[i] - left_values[j]) for i in 1:count, j in 1:count]
    thresholds = sort!(unique!(vec(copy(costs))))
    low = 1
    high = length(thresholds)
    assignment = nothing
    while low < high
        middle = (low + high) >>> 1
        candidate = threshold_matching(costs, thresholds[middle])
        if candidate === nothing
            low = middle + 1
        else
            high = middle
            assignment = candidate
        end
    end
    assignment = threshold_matching(costs, thresholds[low])
    assignment === nothing && error("internal bottleneck matching failure")
    errors = Float64[costs[i, assignment[i]] for i in 1:count]
    assignment, errors
end

function normalize_columns!(vectors; visibilitytol)
    norms = [norm(view(vectors, :, column)) for column in axes(vectors, 2)]
    reference = isempty(norms) ? 0.0 : maximum(norms)
    for column in axes(vectors, 2)
        if norms[column] > visibilitytol * reference
            vectors[:, column] ./= norms[column]
        else
            fill!(view(vectors, :, column), 0)
        end
    end
    vectors
end

function relative_residuals(
    T,
    values,
    vectors,
    residual_scale;
    adjoint_operator=false,
    residual_metric=nothing,
)
    residuals = zeros(Float64, length(values))
    for j in eachindex(values)
        Tvalue = Matrix{ComplexF64}(T(values[j]))
        vector = view(vectors, :, j)
        vector_norm = norm(vector)
        if vector_norm <= 100eps(Float64)
            residuals[j] = Inf
            continue
        end
        side = adjoint_operator ? :left : :right
        if residual_metric === nothing
            residual = adjoint_operator ? adjoint(Tvalue) * vector : Tvalue * vector
            residuals[j] = norm(residual) / (residual_scale * vector_norm)
        else
            residuals[j] = Float64(residual_metric(Tvalue, vector, side, values[j]))
            residuals[j] >= 0 || throw(ArgumentError("residual_metric must return a nonnegative value"))
        end
    end
    residuals
end

function extract_two_sided(
    T,
    chart,
    right_moments,
    left_moments,
    right_observer,
    left_observer,
    moment_count;
    ranktol,
    maxrank=typemax(Int),
    residual_scale,
    residual_metric,
    visibilitytol,
    coupling,
    divided_difference,
    divided_overlap,
    common_ranktol,
    common_acceptance_ratio,
)
    right_H0, _ = block_hankel(right_moments, moment_count; observer=right_observer)
    left_H0, _ = block_hankel(left_moments, moment_count; observer=left_observer)
    right_singulars = svdvals(right_H0)
    left_singulars = svdvals(left_H0)
    right_rank = numerical_rank(right_singulars, ranktol, maxrank)
    left_rank = numerical_rank(left_singulars, ranktol, maxrank)
    right_rank > 0 || error("right moment realization has numerical rank zero")
    left_rank > 0 || error("left moment realization has numerical rank zero")
    rank = max(right_rank, left_rank)
    right_realization = hankel_realization(
        right_moments,
        right_observer,
        moment_count;
        ranktol=ranktol,
        maxrank=maxrank,
        fixed_rank=right_rank,
    )
    left_realization = hankel_realization(
        left_moments,
        left_observer,
        moment_count;
        ranktol=ranktol,
        maxrank=maxrank,
        fixed_rank=left_rank,
    )

    right_eigen = eigen(right_realization.state)
    left_eigen = eigen(left_realization.state)
    all_right_values = chart.center .+ chart.radius .* ComplexF64.(right_eigen.values)
    all_left_values = chart.center .+ chart.radius .* conj.(ComplexF64.(left_eigen.values))
    right_inside = findall(value -> in_chart(chart, value), all_right_values)
    left_inside = findall(value -> in_chart(chart, value), all_left_values)
    length(right_inside) == length(left_inside) || throw(DimensionMismatch(
        "right/left interior realization counts disagree: right=$(length(right_inside)), left=$(length(left_inside))",
    ))
    right_values = all_right_values[right_inside]
    left_values = all_left_values[left_inside]
    left_order, match_errors = bottleneck_match(right_values, left_values)

    right_vectors = right_realization.output * right_eigen.vectors[:, right_inside]
    left_vectors = left_realization.output * left_eigen.vectors[:, left_inside[left_order]]
    normalize_columns!(right_vectors; visibilitytol=visibilitytol)
    normalize_columns!(left_vectors; visibilitytol=visibilitytol)

    values = ComplexF64.(right_values)
    left_values = ComplexF64.(left_values[left_order])
    right_vectors = Matrix{ComplexF64}(right_vectors)
    left_vectors = Matrix{ComplexF64}(left_vectors)
    match_errors = Float64.(match_errors)
    order = sortperm(values; by=value -> (real(value), imag(value)))
    values = values[order]
    left_values = left_values[order]
    right_vectors = right_vectors[:, order]
    left_vectors = left_vectors[:, order]
    match_errors = match_errors[order]
    right_residuals = relative_residuals(
        T,
        values,
        right_vectors,
        residual_scale;
        residual_metric=residual_metric,
    )
    left_residuals = relative_residuals(
        T,
        left_values,
        left_vectors,
        residual_scale;
        adjoint_operator=true,
        residual_metric=residual_metric,
    )
    residuals = max.(right_residuals, left_residuals)
    independent_residual = isempty(residuals) ? Inf : maximum(residuals)
    has_coupling_data = divided_difference !== nothing || divided_overlap !== nothing
    disabled_reason = coupling === :hybrid ? :missing_callback : :disabled
    diagnostics = disabled_loewner_diagnostics(independent_residual; reason=disabled_reason)
    selection_mode = :independent

    if coupling === :hybrid && has_coupling_data
        candidate, diagnostics = coupled_candidate(
            T,
            chart,
            divided_difference,
            divided_overlap,
            values,
            right_vectors,
            left_values,
            left_vectors,
            independent_residual,
            residual_scale;
            residual_metric=residual_metric,
            ranktol=common_ranktol,
            acceptance_ratio=common_acceptance_ratio,
        )
        if diagnostics.accepted
            common_order = sortperm(candidate.values; by=value -> (real(value), imag(value)))
            values = candidate.values[common_order]
            left_values = copy(values)
            right_vectors = candidate.right[:, common_order]
            left_vectors = candidate.left[:, common_order]
            right_residuals = candidate.right_residuals[common_order]
            left_residuals = candidate.left_residuals[common_order]
            residuals = candidate.residuals[common_order]
            selection_mode = :common
        end
    end

    FusedExtraction(
        values,
        copy(values),
        left_values,
        right_vectors,
        left_vectors,
        right_residuals,
        left_residuals,
        residuals,
        match_errors,
        rank,
        Float64.(right_singulars),
        Float64.(left_singulars),
        diagnostics,
        selection_mode,
    )
end
