@testset "residual response blocks have an explicit lifetime" begin
    chart = CircularChart(0.0, 1.0, 8)
    solve = (z, right_hand_side) -> right_hand_side ./ z
    cache = ContourSampleCache(chart, solve, solve)
    probe = ones(ComplexF64, 2, 1)

    add_right_probe!(cache, :temporary, probe; role=:residual)
    add_left_probe!(cache, :temporary, probe; role=:residual)
    counts = solve_counts(cache)
    @test haskey(cache.right_blocks, :temporary)
    @test haskey(cache.left_blocks, :temporary)

    drop_right_probe!(cache, :temporary)
    drop_left_probe!(cache, :temporary)
    @test !haskey(cache.right_blocks, :temporary)
    @test !haskey(cache.left_blocks, :temporary)
    @test solve_counts(cache) == counts
end

@testset "contour solve diagnostics are enforceable" begin
    chart = CircularChart(0.0, 1.0, 8)
    reported_solve = (z, block) -> ContourSolveResult(
        copy(block);
        relative_residual=1e-9,
        iterations=3,
    )
    cache = ContourSampleCache(chart, reported_solve, reported_solve)
    add_right_probe!(
        cache,
        :right,
        ones(ComplexF64, 2, 1);
        require_diagnostics=true,
        maximum_relative_residual=1e-8,
    )
    add_left_probe!(
        cache,
        :left,
        ones(ComplexF64, 2, 1);
        require_diagnostics=true,
        maximum_relative_residual=1e-8,
    )
    summary = solve_diagnostic_summary(cache)
    @test summary.total == 16
    @test summary.reported == 16
    @test summary.maximum_relative_residual == 1e-9
    @test summary.maximum_iterations == 3
    drop_right_probe!(cache, :right)
    drop_left_probe!(cache, :left)
    @test solve_diagnostic_summary(cache) == summary

    raw_cache = ContourSampleCache(chart, (z, block) -> block, (z, block) -> block)
    @test_throws ContourSolveFailureError add_right_probe!(
        raw_cache,
        :right,
        ones(ComplexF64, 2, 1);
        require_diagnostics=true,
    )
    inaccurate_cache = ContourSampleCache(chart, reported_solve, reported_solve)
    @test_throws ContourSolveFailureError add_left_probe!(
        inaccurate_cache,
        :left,
        ones(ComplexF64, 2, 1);
        maximum_relative_residual=1e-10,
    )
end


@testset "probe observers can outlive their contour responses" begin
    case = linear_case(Diagonal(ComplexF64[-0.2, 0.4]))
    cache = ContourSampleCache(
        CircularChart(0.0, 1.0, 8),
        case.right_solve,
        case.left_solve,
    )
    probe = reshape(ComplexF64[1.0, 0.5], :, 1)
    add_right_probe!(cache, :observer, probe)
    responses = release_right_responses!(cache, :observer)
    @test length(responses) == 8
    @test isempty(cache.right_blocks[:observer].responses)
    @test combined_probe(cache, [:observer], :right) == probe
    @test_throws ArgumentError probe_moments(cache, :observer, :right, 2)
end
