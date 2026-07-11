@testset "polynomial companion bridge" begin
    using FEASTSolver

    case = many_root_polynomial_case()
    chart = CircularChart(case.center, case.radius, 64)
    rng = MersenneTwister(1201)
    right_probe = rand(rng, ComplexF64, case.dimension, case.dimension)
    left_probe = rand(rng, ComplexF64, case.dimension, case.dimension)
    result = fused_nlfeast(
        case.T,
        case.right_solve,
        case.left_solve,
        chart,
        right_probe,
        left_probe;
        config=FusedConfig(moment_count=5, iterations=0, ranktol=1e-10, residual_tol=1e-8),
        divided_overlap=case.divided_overlap,
    )

    @test length(result.extraction.values) == length(case.expected)
    @test multiset_distance(result.extraction.values, case.expected) <= 1e-8
    @test maximum(result.extraction.residuals) <= 1e-8
    @test result.extraction.rank == length(case.expected)

    companion_values, _, companion_residuals = FEASTSolver.companion(case.coefficients)
    companion_inside = (abs.(companion_values .- case.center) .< case.radius) .& (companion_residuals .< 1e-7)
    companion_values = ComplexF64.(companion_values[companion_inside])
    @test length(companion_values) == length(case.expected)
    @test multiset_distance(result.extraction.values, companion_values) <= 1e-8
    @test result.residual_converged
end
