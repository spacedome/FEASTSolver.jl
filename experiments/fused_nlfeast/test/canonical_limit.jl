@testset "K=1 NLFEAST-Beyn limit" begin
    using FEASTSolver

    case = canonical_one_root_case()
    center = 0.0 + 0.0im
    radius = 1.2
    nodes = 8
    rng = MersenneTwister(1151)
    right_probe = rand(rng, ComplexF64, case.dimension, case.dimension)
    left_probe = rand(rng, ComplexF64, case.dimension, case.dimension)
    result = fused_nlfeast(
        case.T,
        case.right_solve,
        case.left_solve,
        CircularChart(center, radius, nodes),
        right_probe,
        left_probe;
        config=FusedConfig(moment_count=1, iterations=4, ranktol=1e-10, residual_tol=1e-12),
        divided_overlap=case.divided_overlap,
    )
    canonical_values, _, canonical_residuals = FEASTSolver.nlfeast!(
        case.T,
        copy(right_probe),
        nodes,
        4;
        c=center,
        r=radius,
        store=true,
    )
    canonical_inside = abs.(canonical_values .- center) .< radius
    canonical_values = ComplexF64.(canonical_values[canonical_inside])

    @test length(result.extraction.values) == length(case.expected)
    @test multiset_distance(result.extraction.values, case.expected) <= 1e-9
    @test multiset_distance(result.extraction.values, canonical_values) <= 1e-9
    @test maximum(result.extraction.residuals) <= 1e-9
    @test maximum(canonical_residuals[canonical_inside]) <= 1e-9
    @test length(result.history) > 1
    @test result.history[1].coupling_accepted
    @test result.extraction.coupling.accepted
    @test result.extraction.selection_mode === :common
    @test size(result.extraction.coupling.gram) == (length(case.expected), length(case.expected))
    @test all(
        isapprox(norm(result.extraction.right[:, j]), norm(result.extraction.left[:, j]); rtol=1e-12)
        for j in eachindex(result.extraction.values)
    )
    @test haskey(result.cache.right_blocks, :right_residual_1)
    @test result.history[end].max_residual < result.history[1].max_residual
    @test result.residual_converged
end
