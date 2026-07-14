@testset "cache-native policies" begin
    problem = PMF.contact_mean_field_1d(
        points=40,
        half_length=7.0,
        coupling=5.0,
        occupied=3,
    )
    initial = PMF.initial_orbitals(problem; seed=77)
    chart = PMF.OccupiedChartPolicy(
        lower_bound=0.0,
        upper_bound=20.0,
        node_count=48,
    )
    common = (
        moment_depth=3,
        probe_width=1,
        window_blocks=4,
        refresh_iterations=20,
        reduced_inner_iterations=2,
        reduced_mixing=0.5,
        density_tolerance=1e-7,
        residual_tolerance=1e-7,
    )
    cached_window = solve_cache_native_nlfeast(
        problem,
        chart,
        initial;
        config=CacheNativeConfig(; common...),
    )
    standalone_window = PMF.solve_raw_windowed_nlfeast(
        problem,
        chart,
        initial;
        config=PMF.WindowedNLFEASTConfig(
            moment_depth=3,
            probe_width=1,
            window_blocks=4,
            refresh_iterations=20,
            inner_schedule=:fixed,
            fixed_inner_iterations=2,
            inner_method=:anderson,
            inner_mixing=0.5,
            density_tolerance=1e-7,
            residual_tolerance=1e-7,
        ),
    )
    @test cached_window.converged
    @test cached_window.refresh_count == standalone_window.refresh_count
    @test cached_window.base_rhs_count == standalone_window.rhs_count
    @test cached_window.density == standalone_window.density
    @test PMF.subspace_gap(cached_window.orbitals, standalone_window.orbitals) == 0

    combined = solve_cache_native_nlfeast(
        problem,
        chart,
        initial;
        config=CacheNativeConfig(
            ;
            common...,
            pre_update=:response,
            response_warmup_refreshes=2,
            response_warmup_mixing=0.6,
            response_period=3,
            response_krylov_tolerance=0.03,
        ),
    )
    @test combined.converged
    @test combined.refresh_count < cached_window.refresh_count
    @test combined.factorization_count < cached_window.factorization_count
    @test combined.response_rhs_count > 0
    @test norm(combined.density - cached_window.density) /
        norm(cached_window.density) <= 1e-6

    adaptive = solve_cache_native_nlfeast(
        problem,
        chart,
        initial;
        config=CacheNativeConfig(
            ;
            common...,
            pre_update=:adaptive_response,
            response_warmup_refreshes=2,
            response_period=3,
            response_forcing_ratio=1.0,
            response_krylov_tolerance=0.03,
        ),
    )
    @test adaptive.converged
    @test adaptive.factorization_count < cached_window.factorization_count
    @test adaptive.response_rhs_count > 0
end
