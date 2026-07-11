@testset "scalar many-root cache iteration" begin
    case = scalar_sine_case()
    chart = CircularChart(0.0 + 0.0im, 10.0, 32)
    result = fused_nlfeast(
        case.T,
        case.right_solve,
        case.left_solve,
        chart,
        ones(ComplexF64, 1, 1),
        ones(ComplexF64, 1, 1);
        config=FusedConfig(
            moment_count=7,
            iterations=8,
            ranktol=1e-10,
            residual_ranktol=1e-13,
            residual_tol=1e-14,
            target_count=7,
            target_count_certified=true,
        ),
        divided_overlap=case.divided_overlap,
    )

    expected = ComplexF64[k * pi for k in -3:3]
    @test length(result.extraction.values) == length(expected)
    @test multiset_distance(result.extraction.values, expected) <= 1e-8
    @test result.history[end].max_residual < result.history[1].max_residual
    @test result.history[1].rank == length(expected) > case.dimension
    @test result.history[2].max_residual < result.history[1].max_residual
    @test all(
        result.history[index + 1].max_residual < result.history[index].max_residual
        for index in 1:(length(result.history) - 1)
    )
    @test result.history[end].max_residual <= 1e-8
    @test result.extraction.rank > case.dimension
    @test result.pairing_accepted
    @test all(block.role === :residual for (id, block) in result.cache.right_blocks if id !== :initial)
    @test solve_counts(result.cache).right == length(chart.nodes) * length(result.cache.right_blocks)
    @test solve_counts(result.cache).left == length(chart.nodes) * length(result.cache.left_blocks)
    @test result.residual_converged
    @test result.count_matched
    @test result.count_certified
    @test result.converged
    @test result.certified
    @test result.termination_reason === :converged_certified
    @test all(record.coupling_accepted for record in result.history)
    @test length(result.history) == 3

    compressed = fused_nlfeast(
        case.T,
        case.right_solve,
        case.left_solve,
        chart,
        ones(ComplexF64, 1, 1),
        ones(ComplexF64, 1, 1);
        config=FusedConfig(
            moment_count=7,
            iterations=4,
            ranktol=1e-10,
            residual_ranktol=1e-13,
            residual_tol=1e-14,
            right_moment_width=3,
            left_moment_width=3,
            target_count=7,
        ),
        divided_overlap=case.divided_overlap,
    )
    @test multiset_distance(compressed.extraction.values, expected) <= 1e-8
    @test compressed.residual_converged
    @test compressed.count_matched
    @test !compressed.count_certified
    @test compressed.converged
    @test !compressed.certified
    @test compressed.termination_reason === :converged_uncertified
    @test all(record.right_moment_width == 3 for record in compressed.history[2:end])
    @test all(record.left_moment_width == 3 for record in compressed.history[2:end])
    @test length(compressed.history) <= length(result.history)

    scale = 1e-20
    scaled = (
        T=z -> scale .* case.T(z),
        right_solve=(z, B) -> case.right_solve(z, B) ./ scale,
        left_solve=(z, B) -> case.left_solve(z, B) ./ scale,
    )
    scaled_result = fused_nlfeast(
        scaled.T,
        scaled.right_solve,
        scaled.left_solve,
        chart,
        ones(ComplexF64, 1, 1),
        ones(ComplexF64, 1, 1);
        config=result.config,
        divided_overlap=(left_values, left, right_values, right) ->
            scale .* case.divided_overlap(left_values, left, right_values, right),
    )
    @test scaled_result.residual_converged
    @test multiset_distance(scaled_result.extraction.values, expected) <= 1e-8
    @test length(scaled_result.history) == length(result.history)
    @test isapprox(scaled_result.residual_scale, scale * result.residual_scale; rtol=1e-12)

    imbalanced_T = z -> Matrix(Diagonal(ComplexF64[1e12, sin(z)]))
    imbalanced_right_solve = (z, B) -> B ./ ComplexF64[1e12, sin(z)]
    imbalanced_left_solve = (z, B) -> B ./ conj.(ComplexF64[1e12, sin(z)])
    component_metric = function (Tvalue, vector, side, value)
        action = side === :right ? Tvalue * vector : adjoint(Tvalue) * vector
        norm(action) / norm(vector)
    end
    imbalanced = fused_nlfeast(
        imbalanced_T,
        imbalanced_right_solve,
        imbalanced_left_solve,
        chart,
        ComplexF64[0; 1;;],
        ComplexF64[0; 1;;];
        config=FusedConfig(
            moment_count=7,
            iterations=8,
            ranktol=1e-10,
            residual_ranktol=1e-13,
            residual_tol=1e-8,
            residual_metric=component_metric,
            target_count=7,
        ),
        divided_overlap=(left_values, left, right_values, right) -> ComplexF64[
            conj(left[2, i]) * FusedNLFEAST.sine_divided_difference(left_values[i], right_values[j]) *
            right[2, j]
            for i in eachindex(left_values), j in eachindex(right_values)
        ],
    )
    @test imbalanced.residual_scale === nothing
    @test imbalanced.residual_converged
    @test length(imbalanced.history) > 1
    @test multiset_distance(imbalanced.extraction.values, expected) <= 1e-8
end
