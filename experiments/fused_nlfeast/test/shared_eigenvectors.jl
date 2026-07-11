@testset "higher moments separate shared eigenvectors" begin
    case = shared_eigenvector_case()
    for (index, contour) in enumerate(case.contours)
        rng = MersenneTwister(810 + index)
        result = fused_nlfeast(
            case.T,
            case.right_solve,
            case.left_solve,
            CircularChart(contour.center, contour.radius, 16),
            rand(rng, ComplexF64, case.dimension, 1),
            rand(rng, ComplexF64, case.dimension, 1);
            config=FusedConfig(
                moment_count=2,
                iterations=8,
                ranktol=1e-9,
                residual_ranktol=1e-13,
                residual_tol=1e-11,
                maxrank=2,
            ),
            divided_overlap=case.divided_overlap,
        )
        @test multiset_distance(result.extraction.values, contour.expected) <= 2e-9
        @test result.residual_converged
        @test result.extraction.rank == 2
        @test result.extraction.coupling.attempted
        @test result.extraction.coupling.rank == 2
        if contour.shared === :left
            @test rank(result.extraction.left; rtol=1e-8) == 1
        elseif contour.shared === :right
            @test rank(result.extraction.right; rtol=1e-8) == 1
        end
    end
end
