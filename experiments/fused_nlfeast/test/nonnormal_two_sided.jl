@testset "nonnormal two-sided extraction" begin
    case = nonnormal_analytic_case(coupling=8.0)
    chart = CircularChart(0.0 + 0.0im, 4.0, 128)
    rng = MersenneTwister(1301)
    right_probe = rand(rng, ComplexF64, case.dimension, case.dimension)
    left_probe = rand(rng, ComplexF64, case.dimension, case.dimension)
    result = fused_nlfeast(
        case.T,
        case.right_solve,
        case.left_solve,
        chart,
        right_probe,
        left_probe;
        config=FusedConfig(moment_count=5, iterations=1, ranktol=1e-10, residual_tol=1e-10),
        divided_overlap=case.divided_overlap,
    )

    expected = sort(
        ComplexF64[-pi, 0, pi, -pi / 2, pi / 2, im * 0];
        by=value -> (real(value), imag(value)),
    )
    @test length(result.extraction.values) == length(expected)
    @test multiset_distance(result.extraction.values, expected) <= 1e-8
    @test maximum(result.extraction.right_residuals) <= 1e-8
    @test maximum(result.extraction.left_residuals) <= 1e-8
    @test maximum(result.extraction.match_errors) <= 1e-8
    @test result.pairing_accepted
    @test all(!record.coupling_accepted for record in result.history)

    right_moments = probe_moments(result.cache, :initial, :right, 10)
    left_moments = probe_moments(result.cache, :initial, :left, 10)
    cross_errors = [norm(left_probe' * right_moments[k] - left_moments[k]' * right_probe) for k in 1:10]
    @test maximum(cross_errors) <= 1e-12
end
