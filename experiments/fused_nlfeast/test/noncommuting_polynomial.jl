@testset "noncommuting quadratic realization" begin
    using FEASTSolver
    using MatrixMarket

    repository = normpath(joinpath(@__DIR__, "..", "..", ".."))
    A0 = Matrix{ComplexF64}(Matrix(mmread(joinpath(repository, "data", "quadraticM0.mtx"))))
    A1 = Matrix{ComplexF64}(Matrix(mmread(joinpath(repository, "data", "quadraticM1.mtx"))))
    coefficients = [A0 - 0.02 .* A1, 0.1 .* A1, A1]
    @test norm(coefficients[1] * coefficients[2] - coefficients[2] * coefficients[1]) > 1e-3

    center = 0.0 + 0.0im
    radius = 0.25
    companion_values, _, companion_residuals = FEASTSolver.companion(coefficients)
    selected = (abs.(companion_values .- center) .< radius) .& (companion_residuals .< 1e-7)
    expected = ComplexF64.(companion_values[selected])
    case = polynomial_case(coefficients)
    rng = MersenneTwister(1251)
    result = fused_nlfeast(
        case.T,
        case.right_solve,
        case.left_solve,
        CircularChart(center, radius, 16),
        rand(rng, ComplexF64, case.dimension, 4),
        rand(rng, ComplexF64, case.dimension, 4);
        config=FusedConfig(moment_count=3, iterations=6, ranktol=1e-10, residual_tol=1e-12),
        divided_overlap=case.divided_overlap,
    )

    @test length(result.extraction.values) == length(expected)
    @test multiset_distance(result.extraction.values, expected) <= 1e-8
    @test maximum(result.extraction.right_residuals) <= 1e-8
    @test maximum(result.extraction.left_residuals) <= 1e-8
    @test result.residual_converged
    @test result.history[1].count > length(expected)
    @test result.history[end].count == length(expected)
    @test result.history[end].max_residual < result.history[1].max_residual
    @test haskey(result.cache.right_blocks, :right_residual_1)
    @test !result.history[1].coupling_accepted
    right_values = result.extraction.right_interpolation_values
    left_values = result.extraction.left_interpolation_values
    structured_overlap = case.divided_overlap(
        left_values,
        result.extraction.left,
        right_values,
        result.extraction.right,
    )
    reference_overlap = ComplexF64[
        dot(
            result.extraction.left[:, i],
            case.divided_difference(left_values[i], right_values[j]) * result.extraction.right[:, j],
        )
        for i in eachindex(left_values), j in eachindex(right_values)
    ]
    @test norm(structured_overlap - reference_overlap) <= 1e-10 * norm(reference_overlap)
end
