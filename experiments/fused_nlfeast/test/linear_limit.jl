@testset "linear FEAST limit" begin
    eigenvalues = ComplexF64[-1.5, -0.7 + 0.2im, 0.1 - 0.3im, 0.8, 1.4 + 0.1im, 2.1]
    similarity = ComplexF64[
        1 2 0 0 0 0
        0 1 1 0 0 0
        0 0 1 2 0 0
        0 0 0 1 1 0
        0 0 0 0 1 2
        0 0 0 0 0 1
    ]
    A = similarity * Diagonal(eigenvalues) / similarity
    case = linear_case(A)
    chart = CircularChart(0.3 + 0.0im, 2.2, 32)
    rng = MersenneTwister(1101)
    right_probe = rand(rng, ComplexF64, case.dimension, 2)
    left_probe = rand(rng, ComplexF64, case.dimension, 2)
    result = fused_nlfeast(
        case.T,
        case.right_solve,
        case.left_solve,
        chart,
        right_probe,
        left_probe;
        config=FusedConfig(moment_count=3, iterations=0, ranktol=1e-9),
        divided_overlap=case.divided_overlap,
    )

    expected = eigenvalues[abs.(eigenvalues .- chart.center) .< chart.radius]
    @test length(result.extraction.values) == length(expected)
    @test multiset_distance(result.extraction.values, expected) <= 1e-9
    @test maximum(result.extraction.right_residuals) <= 1e-9
    @test maximum(result.extraction.left_residuals) <= 1e-9
    @test solve_counts(result.cache) == (right=32, left=32)
    without_callback = fused_nlfeast(
        case.T,
        case.right_solve,
        case.left_solve,
        chart,
        right_probe,
        left_probe;
        config=FusedConfig(moment_count=3, iterations=0),
    )
    @test without_callback.extraction.selection_mode === :independent
    @test without_callback.extraction.coupling.reason === :missing_callback

    values = ComplexF64[-0.8 + 0.1im, 0.4 - 0.2im]
    left_values = values + ComplexF64[0.17 - 0.08im, -0.13 + 0.11im]
    right = rand(rng, ComplexF64, case.dimension, length(values))
    left = rand(rng, ComplexF64, case.dimension, length(values))
    factors = residual_factors(case.T, values, left_values, right, left; ranktol=1e-14)
    cache = ContourSampleCache(chart, case.right_solve, case.left_solve)
    add_right_probe!(cache, :residual, factors.right_basis; role=:residual)
    add_left_probe!(cache, :residual, factors.left_basis; role=:residual)
    corrected_right, corrected_left = corrected_moments(
        cache,
        values,
        left_values,
        right,
        left,
        :residual,
        :residual,
        factors.right_coefficients,
        factors.left_coefficients,
        4,
    )
    add_right_probe!(cache, :filtered, right; role=:linear_filter)
    add_left_probe!(cache, :filtered, left; role=:linear_filter)
    filtered_right = probe_moments(cache, :filtered, :right, 4)
    filtered_left = probe_moments(cache, :filtered, :left, 4)
    @test maximum(norm.(corrected_right .- filtered_right)) <= 1e-12
    @test maximum(norm.(corrected_left .- filtered_left)) <= 1e-12
    right_tangent = moment_tangent(
        values,
        2,
        1;
        cluster_tolerance=1e-8,
        side=:right,
    )
    left_tangent = moment_tangent(
        left_values,
        2,
        1;
        cluster_tolerance=1e-8,
        side=:left,
    )
    compressed_right, compressed_left = corrected_moments(
        cache,
        values,
        left_values,
        right,
        left,
        :residual,
        :residual,
        factors.right_coefficients,
        factors.left_coefficients,
        4;
        right_tangent=right_tangent,
        left_tangent=left_tangent,
    )
    @test maximum(norm.(compressed_right .- [moment * right_tangent for moment in corrected_right])) <= 1e-12
    @test maximum(norm.(compressed_left .- [moment * left_tangent for moment in corrected_left])) <= 1e-12
    @test_throws ArgumentError moment_tangent(
        ComplexF64[0, 0, 0, 1],
        2,
        2;
        cluster_tolerance=1e-8,
        side=:right,
    )
    partitioned = Matrix(moment_tangent(
        ComplexF64[0, 0, 1, 2],
        2,
        2;
        cluster_tolerance=1e-8,
        side=:right,
    ))
    @test all(Base.count(!iszero, partitioned[row, :]) == 1 for row in axes(partitioned, 1))
    @test findfirst(!iszero, partitioned[1, :]) != findfirst(!iszero, partitioned[2, :])
    @test_throws ArgumentError probe_moments(cache, :residual, :right, 2)
    @test_throws ArgumentError fused_nlfeast(
        case.T,
        case.right_solve,
        case.left_solve,
        CircularChart(0.0, 1.0, 4),
        right_probe,
        left_probe;
        config=FusedConfig(moment_count=3, iterations=0),
        divided_overlap=case.divided_overlap,
    )
    invisible = FusedNLFEAST.relative_residuals(
        z -> Matrix{ComplexF64}(I, 2, 2),
        ComplexF64[0],
        zeros(ComplexF64, 2, 1),
        1.0,
    )
    @test isinf(only(invisible))
end
