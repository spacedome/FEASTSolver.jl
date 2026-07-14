@testset "frozen cache algebra" begin
    problem = PMF.contact_mean_field_1d(
        points=32,
        half_length=6.0,
        coupling=1.0,
        occupied=3,
    )
    orbitals = PMF.initial_orbitals(problem; seed=33)
    rho = PMF.density(problem, orbitals)
    reference = PMF.reference_scf(problem, rho; mixing=0.5)
    spectrum = eigvals(PMF.hamiltonian(problem, reference.density))
    chart = PMF.CircularChart(
        (spectrum[1] + spectrum[3]) / 2,
        (spectrum[3] - spectrum[1]) / 2 + 0.3 * (spectrum[4] - spectrum[3]),
        64,
    )

    cache = build_density_cache(problem, chart, rho)
    cached = cached_projector_step!(cache, orbitals, 3, 1)
    tangent = PMF.moment_tangent(problem.occupied, 1)
    standalone = PMF.moment_projector_step(
        cache.hamiltonian,
        chart,
        orbitals * tangent,
        3,
        problem.occupied;
        factorizations=cache.factors,
    )
    @test PMF.subspace_gap(cached.orbitals, standalone.orbitals) <= 1e-13

    fused_chart = FusedNLFEAST.CircularChart(
        chart.center,
        chart.radius,
        length(chart.nodes),
    )
    factor_by_node = Dict(zip(chart.nodes, cache.factors))
    solve = (z, block) -> factor_by_node[z] \ block
    sample_cache = FusedNLFEAST.ContourSampleCache(fused_chart, solve, solve)
    FusedNLFEAST.add_right_probe!(sample_cache, :base, cached.probe)
    sample_moments = FusedNLFEAST.probe_moments(sample_cache, :base, :right, 6)
    factor_moments = positive_moments!(
        build_density_cache(problem, chart, rho),
        cached.probe,
        6,
    )
    @test sample_moments == factor_moments

    explicit_cache = build_density_cache(problem, chart, rho)
    state = adjoint(orbitals) * explicit_cache.hamiltonian * orbitals
    explicit = corrected_moments!(
        explicit_cache,
        orbitals,
        state,
        6;
        tangent=tangent,
        realization=:explicit,
    )
    identity_cache = build_density_cache(problem, chart, rho)
    identity = corrected_moments!(
        identity_cache,
        orbitals,
        state,
        6;
        tangent=tangent,
    )
    @test maximum(
        norm(left - right) / norm(right)
        for (left, right) in zip(explicit, identity)
    ) <= 1e-13
    @test cache_stats(explicit_cache).correction_rhs_count ==
        length(chart.nodes) * problem.occupied
    @test cache_stats(identity_cache).base_rhs_count == length(chart.nodes)

    direction = randn(MersenneTwister(9), length(rho))
    cached_response = density_response!(
        cache,
        cached.orbitals,
        cached.values,
        direction,
    )
    standalone_response = PMF.contour_density_response(
        problem,
        cached.orbitals,
        cached.values,
        direction,
        chart;
        factorizations=cache.factors,
    )
    @test cached_response == standalone_response
    @test cache_stats(cache).response_rhs_count ==
        length(chart.nodes) * problem.occupied
    invalidate!(cache)
    @test_throws ArgumentError density_response!(
        cache,
        cached.orbitals,
        cached.values,
        direction,
    )
end
