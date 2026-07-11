function moment_tangent(state_count, width)
    1 <= width <= state_count || throw(ArgumentError("invalid moment tangent width"))
    ComplexF64[
        cis(2pi * (row - 1) * (column - 0.5) / state_count) / sqrt(state_count)
        for row in 1:state_count, column in 1:width
    ]
end

function block_hankel(moments, depth, observer)
    length(moments) >= 2depth || throw(ArgumentError("two moments per depth are required"))
    projected = [adjoint(observer) * moment for moment in moments]
    observed = size(observer, 2)
    width = size(moments[1], 2)
    H0 = zeros(ComplexF64, depth * observed, depth * width)
    H1 = similar(H0)
    for row in 1:depth, column in 1:depth
        rows = ((row - 1) * observed + 1):(row * observed)
        columns = ((column - 1) * width + 1):(column * width)
        H0[rows, columns] .= projected[row + column - 1]
        H1[rows, columns] .= projected[row + column]
    end
    H0, H1
end

function contour_factorizations(H, chart::CircularChart)
    [lu(z * I - H) for z in chart.nodes]
end

function moment_projector_step(
    H,
    chart::CircularChart,
    probe,
    moment_depth,
    target_count;
    ranktol=1e-12,
    factorizations=nothing,
    verify_count=true,
)
    moment_depth > 0 || throw(ArgumentError("moment_depth must be positive"))
    target_count > 0 || throw(ArgumentError("target_count must be positive"))
    size(probe, 2) * moment_depth >= target_count || throw(ArgumentError(
        "moment depth and probe width cannot represent the target count",
    ))
    dimension = size(H, 1)
    moments = [zeros(ComplexF64, dimension, size(probe, 2)) for _ in 1:(2moment_depth)]
    factors = factorizations === nothing ? contour_factorizations(H, chart) : factorizations
    length(factors) == length(chart.nodes) || throw(DimensionMismatch(
        "one factorization is required per contour node",
    ))
    count_diagnostic = determinant_count(factors)
    if verify_count && count_diagnostic.count != target_count
        throw(ContourCountError(
            target_count,
            count_diagnostic.count,
            count_diagnostic.winding,
            count_diagnostic.maximum_phase_step,
        ))
    end
    for (z, weight, factorization) in zip(chart.nodes, chart.weights, factors)
        response = factorization \ probe
        coordinate = (z - chart.center) / chart.radius
        power = one(ComplexF64)
        for moment in moments
            moment .+= (weight * power) .* response
            power *= coordinate
        end
    end

    H0, H1 = block_hankel(moments, moment_depth, probe)
    factorization = svd(H0)
    length(factorization.S) >= target_count || error("moment pencil is too small")
    factorization.S[target_count] >= ranktol * factorization.S[1] || error(
        "moment pencil has numerical rank below the requested target count",
    )
    U = factorization.U[:, 1:target_count]
    V = factorization.V[:, 1:target_count]
    inverse_singulars = Diagonal(1.0 ./ factorization.S[1:target_count])
    coordinate_state = adjoint(U) * H1 * V * inverse_singulars
    physical_state = chart.center .* I + chart.radius .* coordinate_state
    state_values = eigvals(physical_state)
    count_inside = count(value -> abs(value - chart.center) < chart.radius, state_values)
    count_inside == target_count || error(
        "moment pencil found $count_inside states inside the contour, expected $target_count",
    )

    moment_row = reduce(hcat, moments[1:moment_depth])
    output = moment_row * V * inverse_singulars
    basis = orthonormalize(output, target_count)
    reduced = eigen(Hermitian(adjoint(basis) * H * basis))
    orbitals = basis * reduced.vectors
    (
        orbitals=orbitals,
        values=Float64.(reduced.values),
        state_values=ComplexF64.(state_values),
        singular_values=Float64.(factorization.S),
        count_diagnostic=count_diagnostic,
        probe_width=size(probe, 2),
        solve_count=length(chart.nodes),
        rhs_count=length(chart.nodes) * size(probe, 2),
    )
end

function rii_response(H, X, S, z)
    size(S, 1) == size(S, 2) == size(X, 2) || throw(DimensionMismatch(
        "the reduced state must match the orbital width",
    ))
    identity_state = Matrix{ComplexF64}(I, size(S, 1), size(S, 2))
    residual = X * S - H * X
    (X - (z * I - H) \ residual) / (z .* identity_state .- S)
end
