function rational_probe_samples(
    cache::ContourSampleCache,
    id::Symbol,
    side::Symbol,
    interpolation_points,
)
    blocks = side === :right ? cache.right_blocks : side === :left ? cache.left_blocks :
        throw(ArgumentError("side must be :right or :left"))
    block = blocks[id]
    samples = [zeros(ComplexF64, size(block.probe)) for _ in interpolation_points]
    for (z, weight, response) in zip(cache.chart.nodes, cache.chart.weights, block.responses)
        coordinate = chart_coordinate(cache.chart, z)
        if side === :left
            coordinate = conj(coordinate)
            weight = conj(weight)
        end
        for (sample, point) in zip(samples, interpolation_points)
            abs(point - coordinate) > eps(Float64) || throw(ArgumentError(
                "a Loewner interpolation point coincides with a contour node",
            ))
            sample .+= (weight / (point - coordinate)) .* response
        end
    end
    samples
end

function block_loewner_pencil(left_points, left_samples, right_points, right_samples)
    length(left_points) == length(left_samples) || throw(DimensionMismatch(
        "left Loewner point/sample count mismatch",
    ))
    length(right_points) == length(right_samples) || throw(DimensionMismatch(
        "right Loewner point/sample count mismatch",
    ))
    isempty(left_samples) && throw(ArgumentError("at least one left Loewner sample is required"))
    isempty(right_samples) && throw(ArgumentError("at least one right Loewner sample is required"))
    output_width, input_width = size(first(left_samples))
    all(size(sample) == (output_width, input_width) for sample in left_samples) ||
        throw(DimensionMismatch("left Loewner samples must have a common size"))
    all(size(sample) == (output_width, input_width) for sample in right_samples) ||
        throw(DimensionMismatch("right Loewner samples must have a common size"))
    loewner = zeros(
        ComplexF64,
        length(left_points) * output_width,
        length(right_points) * input_width,
    )
    shifted = similar(loewner)
    for (i, left_point) in pairs(left_points), (j, right_point) in pairs(right_points)
        separation = left_point - right_point
        abs(separation) > eps(Float64) || throw(ArgumentError(
            "left and right Loewner interpolation points must be distinct",
        ))
        rows = ((i - 1) * output_width + 1):(i * output_width)
        columns = ((j - 1) * input_width + 1):(j * input_width)
        loewner[rows, columns] .= (left_samples[i] .- right_samples[j]) ./ separation
        shifted[rows, columns] .= (
            left_point .* left_samples[i] .- right_point .* right_samples[j]
        ) ./ separation
    end
    loewner, shifted
end

function loewner_realization(
    left_points,
    left_samples,
    right_points,
    right_samples,
    full_right_samples;
    ranktol,
    maxrank=typemax(Int),
    fixed_rank=nothing,
)
    length(right_points) == length(full_right_samples) || throw(DimensionMismatch(
        "one full output sample is required per right interpolation point",
    ))
    loewner, shifted = block_loewner_pencil(
        left_points,
        left_samples,
        right_points,
        right_samples,
    )
    decomposition = svd(loewner)
    rank = fixed_rank === nothing ?
        numerical_rank(decomposition.S, ranktol, maxrank) :
        min(Int(fixed_rank), length(decomposition.S), maxrank)
    rank > 0 || error("Loewner pencil has numerical rank zero")
    input_width = size(first(right_samples), 2)
    physical_dimension = size(first(full_right_samples), 1)
    all(size(sample) == (physical_dimension, input_width) for sample in full_right_samples) ||
        throw(DimensionMismatch("full right Loewner samples must have a common size"))
    U = decomposition.U[:, 1:rank]
    V = decomposition.V[:, 1:rank]
    inverse_singulars = Diagonal(1.0 ./ decomposition.S[1:rank])
    state = adjoint(U) * shifted * V * inverse_singulars
    sample_row = reduce(hcat, full_right_samples)
    output = sample_row * V * inverse_singulars
    MomentRealization(
        Matrix{ComplexF64}(output),
        Matrix{ComplexF64}(state),
        Float64.(decomposition.S),
        rank,
    )
end
