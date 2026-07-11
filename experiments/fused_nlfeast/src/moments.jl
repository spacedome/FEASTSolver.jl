function probe_moments(cache::ContourSampleCache, id::Symbol, side::Symbol, count::Integer)
    count > 0 || throw(ArgumentError("moment count must be positive"))
    blocks = side === :right ? cache.right_blocks : side === :left ? cache.left_blocks :
        throw(ArgumentError("side must be :right or :left"))
    block = blocks[id]
    block.role === :residual && throw(ArgumentError(
        "residual probe $id must be reduced through corrected_moments",
    ))
    length(block.responses) == length(cache.chart.nodes) || throw(ArgumentError(
        "contour responses for probe $id have been released",
    ))
    n, width = size(block.probe)
    moments = [zeros(ComplexF64, n, width) for _ in 1:count]
    for (z, weight, response) in zip(cache.chart.nodes, cache.chart.weights, block.responses)
        coordinate = chart_coordinate(cache.chart, z)
        side === :left && ((coordinate, weight) = (conj(coordinate), conj(weight)))
        power = one(ComplexF64)
        for moment in moments
            moment .+= (weight * power) .* response
            power *= coordinate
        end
    end
    moments
end

function combined_probe_moments(
    cache::ContourSampleCache,
    ids,
    side::Symbol,
    count::Integer,
)
    isempty(ids) && throw(ArgumentError("at least one probe id is required"))
    sequences = [probe_moments(cache, id, side, count) for id in ids]
    [reduce(hcat, (sequence[k] for sequence in sequences)) for k in 1:count]
end

function combined_probe(cache::ContourSampleCache, ids, side::Symbol)
    isempty(ids) && throw(ArgumentError("at least one probe id is required"))
    blocks = side === :right ? cache.right_blocks : side === :left ? cache.left_blocks :
        throw(ArgumentError("side must be :right or :left"))
    reduce(hcat, (blocks[id].probe for id in ids))
end

function probe_moment_data(
    cache::ContourSampleCache,
    right_id::Symbol,
    left_id::Symbol,
    count::Integer,
)
    right = probe_moments(cache, right_id, :right, count)
    left = probe_moments(cache, left_id, :left, count)
    right_block = cache.right_blocks[right_id]
    left_block = cache.left_blocks[left_id]
    cross = [zeros(ComplexF64, size(left_block.probe, 2), size(right_block.probe, 2)) for _ in 1:count]
    for (z, weight, response) in zip(cache.chart.nodes, cache.chart.weights, right_block.responses)
        coordinate = chart_coordinate(cache.chart, z)
        sample = adjoint(left_block.probe) * response
        power = one(ComplexF64)
        for moment in cross
            moment .+= (weight * power) .* sample
            power *= coordinate
        end
    end
    (right=right, left=left, cross=cross)
end

function low_rank_factor(A; ranktol)
    A = Matrix{ComplexF64}(A)
    F = svd(A)
    if isempty(F.S) || iszero(F.S[1])
        return zeros(ComplexF64, size(A, 1), 0), zeros(ComplexF64, 0, size(A, 2)), Float64[]
    end
    rank = count(F.S ./ F.S[1] .> ranktol)
    U = F.U[:, 1:rank]
    coefficients = Diagonal(F.S[1:rank]) * F.Vt[1:rank, :]
    U, coefficients, Float64.(F.S)
end

function residual_factors(T, right_values, left_values, right, left; ranktol)
    n = size(right, 1)
    length(right_values) == size(right, 2) || throw(DimensionMismatch(
        "one right value is required per right vector",
    ))
    length(left_values) == size(left, 2) || throw(DimensionMismatch(
        "one left value is required per left vector",
    ))
    right_residual = zeros(ComplexF64, n, length(right_values))
    left_residual = zeros(ComplexF64, n, length(left_values))
    for j in eachindex(right_values)
        Tvalue = Matrix{ComplexF64}(T(right_values[j]))
        right_residual[:, j] .= Tvalue * view(right, :, j)
    end
    for j in eachindex(left_values)
        Tvalue = Matrix{ComplexF64}(T(left_values[j]))
        left_residual[:, j] .= adjoint(Tvalue) * view(left, :, j)
    end
    right_basis, right_coefficients, right_singulars = low_rank_factor(right_residual; ranktol=ranktol)
    left_basis, left_coefficients, left_singulars = low_rank_factor(left_residual; ranktol=ranktol)
    (
        right_basis=right_basis,
        right_coefficients=right_coefficients,
        right_singulars=right_singulars,
        left_basis=left_basis,
        left_coefficients=left_coefficients,
        left_singulars=left_singulars,
    )
end

residual_factors(T, values, right, left; ranktol) =
    residual_factors(T, values, values, right, left; ranktol=ranktol)

function corrected_moment_data(
    T,
    cache::ContourSampleCache,
    right_values,
    left_values,
    right,
    left,
    right_id::Symbol,
    left_id::Symbol,
    right_coefficients,
    left_coefficients,
    count::Integer,
    ;
    right_tangent=nothing,
    left_tangent=nothing,
)
    right_block = cache.right_blocks[right_id]
    left_block = cache.left_blocks[left_id]
    right_block.role === :residual || throw(ArgumentError("right corrected block must have role :residual"))
    left_block.role === :residual || throw(ArgumentError("left corrected block must have role :residual"))
    n, state_count = size(right)
    length(right_values) == state_count || throw(DimensionMismatch("one value is required per right Ritz vector"))
    length(left_values) == size(left, 2) || throw(DimensionMismatch("one value is required per left Ritz vector"))
    size(left, 1) == n || throw(DimensionMismatch("right and left Ritz vectors must share a physical dimension"))
    size(left, 2) == state_count || throw(DimensionMismatch("right and left Ritz blocks must have equal width"))
    right_tangent === nothing || size(right_tangent, 1) == state_count || throw(DimensionMismatch(
        "right tangent must have one row per right Ritz vector",
    ))
    left_tangent === nothing || size(left_tangent, 1) == state_count || throw(DimensionMismatch(
        "left tangent must have one row per left Ritz vector",
    ))
    right_width = right_tangent === nothing ? state_count : size(right_tangent, 2)
    left_width = left_tangent === nothing ? state_count : size(left_tangent, 2)
    right_moments = [zeros(ComplexF64, n, right_width) for _ in 1:count]
    left_moments = [zeros(ComplexF64, n, left_width) for _ in 1:count]
    mixed_moments = T === nothing ? nothing : [zeros(ComplexF64, left_width, right_width) for _ in 1:count]

    for node_index in eachindex(cache.chart.nodes)
        z = cache.chart.nodes[node_index]
        weight = cache.chart.weights[node_index]
        coordinate = chart_coordinate(cache.chart, z)

        if right_tangent === nothing
            solved_right_residual = right_block.responses[node_index] * right_coefficients
            right_corrected = right .- solved_right_residual
            right_corrected .*= reshape(1.0 ./ (z .- right_values), 1, :)
        else
            rational_tangent = right_tangent .* reshape(1.0 ./ (z .- right_values), :, 1)
            right_corrected = right * rational_tangent -
                right_block.responses[node_index] * (right_coefficients * rational_tangent)
        end
        if left_tangent === nothing
            solved_left_residual = left_block.responses[node_index] * left_coefficients
            left_corrected = left .- solved_left_residual
            left_corrected .*= reshape(1.0 ./ conj.(z .- left_values), 1, :)
        else
            rational_tangent = left_tangent .* reshape(1.0 ./ conj.(z .- left_values), :, 1)
            left_corrected = left * rational_tangent -
                left_block.responses[node_index] * (left_coefficients * rational_tangent)
        end
        mixed_corrected = T === nothing ? nothing :
            adjoint(left_corrected) * Matrix{ComplexF64}(T(z)) * right_corrected

        right_power = one(ComplexF64)
        left_power = one(ComplexF64)
        for k in 1:count
            right_moments[k] .+= (weight * right_power) .* right_corrected
            left_moments[k] .+= (conj(weight) * left_power) .* left_corrected
            if T !== nothing
                mixed_moments[k] .+= (weight * right_power) .* mixed_corrected
            end
            right_power *= coordinate
            left_power *= conj(coordinate)
        end
    end
    T === nothing ? (right_moments, left_moments) :
        (right=right_moments, left=left_moments, mixed=mixed_moments)
end

function corrected_moment_data(
    T,
    cache::ContourSampleCache,
    values,
    right,
    left,
    args...,
    ;
    kwargs...,
)
    corrected_moment_data(T, cache, values, values, right, left, args...; kwargs...)
end

corrected_moments(cache::ContourSampleCache, args...; kwargs...) =
    corrected_moment_data(nothing, cache, args...; kwargs...)

mixed_corrected_moments(T, cache::ContourSampleCache, args...; kwargs...) =
    corrected_moment_data(T, cache, args...; kwargs...)
