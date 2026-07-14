@testset "combined corrected cache" begin
    base = PMF.contact_mean_field_1d(
        points=32,
        half_length=6.0,
        coupling=1.0,
        occupied=3,
    )
    problem = PMF.quadratic_density_nep(base; strength=0.02)
    initial = PMF.initial_orbitals(base; seed=33)
    rho = PMF.density(base, initial)
    chart = PMF.CircularChart(1.8, 1.5, 64)
    cache = build_polynomial_cache(problem, chart, rho)
    sampled = polynomial_corrected_moments!(cache, initial, nothing, 6, 1)
    standalone = PMF.corrected_polynomial_raw_block(
        problem,
        rho,
        chart,
        initial,
        nothing,
        3,
        1,
    )
    @test reduce(hcat, sampled.moments[1:3]) == standalone.block
    @test polynomial_cache_stats(cache).base_rhs_count == length(chart.nodes)

    linear_problem = PMF.quadratic_density_nep(base; strength=0.0)
    linear_hamiltonian = PMF.hamiltonian(base, rho)
    exact = eigen(linear_hamiltonian)
    rank_one_orbitals = ComplexF64.(exact.vectors[:, 1:base.occupied])
    perturbed_state = Diagonal(ComplexF64.(exact.values[1:base.occupied]))
    perturbed_state[1, 1] += 0.2
    compressed_cache = build_polynomial_cache(linear_problem, chart, rho)
    compressed = polynomial_corrected_moments!(
        compressed_cache,
        rank_one_orbitals,
        perturbed_state,
        6,
        1,
    )
    uncompressed_cache = build_polynomial_cache(linear_problem, chart, rho)
    uncompressed = polynomial_corrected_moments!(
        uncompressed_cache,
        rank_one_orbitals,
        perturbed_state,
        6,
        1;
        residual_ranktol=0.0,
    )
    @test maximum(
        norm(left - right) / max(norm(right), eps(Float64))
        for (left, right) in zip(compressed.moments, uncompressed.moments)
    ) <= 1e-13
    @test polynomial_cache_stats(compressed_cache).correction_ranks == [1]
    @test polynomial_cache_stats(compressed_cache).correction_rhs_count ==
        length(chart.nodes)
    @test polynomial_cache_stats(uncompressed_cache).correction_rhs_count ==
        length(chart.nodes) * base.occupied

    common = (
        moment_depth=3,
        probe_width=1,
        window_blocks=4,
        refresh_iterations=15,
        reduced_inner_iterations=3,
        reduced_mixing=0.6,
        density_tolerance=1e-7,
        residual_tolerance=1e-7,
    )
    cached = solve_cache_native_combined(
        problem,
        chart,
        initial;
        config=CombinedCacheConfig(; common...),
    )
    established = PMF.solve_combined_windowed_nlfeast(
        problem,
        chart,
        initial;
        config=PMF.CombinedWindowConfig(
            moment_depth=3,
            probe_width=1,
            window_blocks=4,
            refresh_iterations=15,
            inner_iterations=3,
            inner_mixing=0.6,
            density_tolerance=1e-7,
            residual_tolerance=1e-7,
        ),
    )
    @test cached.converged
    @test cached.refresh_count == established.refresh_count
    @test cached.density == established.density
    @test PMF.subspace_gap(cached.orbitals, established.orbitals) == 0
    @test cached.base_rhs_count + cached.correction_rhs_count ==
        established.rhs_count

    forced = solve_cache_native_combined(
        problem,
        chart,
        initial;
        config=CombinedCacheConfig(
            ;
            common...,
            spectral_corrections=3,
            spectral_tolerance=1e-6,
        ),
    )
    @test forced.converged
    @test forced.refresh_count == cached.refresh_count
    @test length(first(forced.history).spectral_residuals) == 2
    @test all(length(record.spectral_residuals) <= 2 for record in forced.history)
end
