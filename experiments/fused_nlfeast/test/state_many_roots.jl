@testset "structured state iteration with more roots than dimension" begin
    case = scalar_sine_case()
    coefficients = [ones(ComplexF64, 1, 1)]
    functions = (sin,)
    chart = CircularChart(0.0 + 0.0im, 10.0, 32)
    cache = ContourSampleCache(chart, case.right_solve, case.left_solve)
    right_initial = add_right_probe!(cache, :initial, ones(ComplexF64, 1, 1))
    left_initial = add_left_probe!(cache, :initial, ones(ComplexF64, 1, 1))
    right_moments = probe_moments(cache, :initial, :right, 14)
    left_moments = probe_moments(cache, :initial, :left, 14)
    common = common_structured_realization(
        coefficients,
        functions,
        chart,
        right_moments,
        left_moments,
        left_initial.probe,
        right_initial.probe,
        7;
        ranktol=1e-10,
        maxrank=7,
    )
    residual_history = Float64[maximum(abs, sin.(eigvals(common.state)))]
    for iteration in 1:3
        factors = state_residual_factors(
            common.right_residual,
            common.left_residual;
            ranktol=1e-13,
        )
        right_id = Symbol("sine_state_right_", iteration)
        left_id = Symbol("sine_state_left_", iteration)
        add_right_probe!(cache, right_id, factors.right_basis; role=:residual)
        add_left_probe!(cache, left_id, factors.left_basis; role=:residual)
        identity_state = Matrix{ComplexF64}(I, common.rank, common.rank)
        chart_state = (common.state .- chart.center .* identity_state) ./ chart.radius
        right_tangent = state_moment_tangent(
            chart_state,
            7,
            1;
            side=:right,
            ranktol=1e-12,
        )
        left_tangent = state_moment_tangent(
            chart_state,
            7,
            1;
            side=:left,
            ranktol=1e-12,
        )
        right_moments, left_moments = state_corrected_moments(
            cache,
            common.state,
            common.state,
            common.right,
            common.left,
            right_id,
            left_id,
            factors.right_coefficients,
            factors.left_coefficients,
            14;
            right_tangent=right_tangent,
            left_tangent=left_tangent,
        )
        common = common_structured_realization(
            coefficients,
            functions,
            chart,
            right_moments,
            left_moments,
            left_initial.probe,
            right_initial.probe,
            7;
            ranktol=1e-10,
            maxrank=7,
        )
        push!(residual_history, maximum(abs, sin.(eigvals(common.state))))
    end

    expected = ComplexF64[k * pi for k in -3:3]
    @test common.rank == 7 > case.dimension
    @test multiset_distance(eigvals(common.state), expected) <= 1e-8
    @test residual_history[end] <= 1e-9
    @test residual_history[end] <= 1e-8 * residual_history[1]
    @test rank(reduce(vcat, [common.right * common.state^k for k in 0:6]); rtol=1e-8) == 7
    @test size(right_moments[1], 2) == 1
    @test size(left_moments[1], 2) == 1
end
