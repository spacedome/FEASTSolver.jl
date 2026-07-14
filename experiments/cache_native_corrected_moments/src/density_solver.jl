export CacheNativeConfig, solve_cache_native_nlfeast

Base.@kwdef struct CacheNativeConfig
    moment_depth::Int
    probe_width::Int
    window_blocks::Int = 4
    refresh_iterations::Int = 30
    pre_update::Symbol = :none
    response_warmup_refreshes::Int = 2
    response_warmup_mixing::Float64 = 0.5
    response_period::Int = 1
    response_forcing_ratio::Float64 = 1.0
    response_krylov_tolerance::Float64 = 1e-7
    response_krylov_iterations::Int = 30
    response_damping::Float64 = 1.0
    reduced_inner_iterations::Int = 3
    reduced_inner_method::Symbol = :anderson
    reduced_mixing::Float64 = 0.5
    reduced_history_depth::Int = 6
    reduced_regularization::Float64 = 1e-12
    maximum_step_ratio::Float64 = 4.0
    density_tolerance::Float64 = 1e-9
    residual_tolerance::Float64 = 1e-9
    ranktol::Float64 = 1e-12
    subspace_ranktol::Float64 = 1e-10
end

function validate_config(problem, config::CacheNativeConfig)
    config.moment_depth * config.probe_width >= problem.occupied || throw(
        ArgumentError("moment capacity is below the occupied count"),
    )
    config.window_blocks >= 0 || throw(ArgumentError(
        "window_blocks must be nonnegative; zero selects growing memory",
    ))
    config.refresh_iterations > 0 || throw(ArgumentError(
        "refresh_iterations must be positive",
    ))
    config.pre_update in (:none, :response, :adaptive_response) || throw(ArgumentError(
        "pre_update must be :none, :response, or :adaptive_response",
    ))
    config.response_warmup_refreshes >= 0 || throw(ArgumentError(
        "response_warmup_refreshes must be nonnegative",
    ))
    config.response_period > 0 || throw(ArgumentError(
        "response_period must be positive",
    ))
    config.response_forcing_ratio > 0 || throw(ArgumentError(
        "response_forcing_ratio must be positive",
    ))
    config.reduced_inner_iterations >= 0 || throw(ArgumentError(
        "reduced_inner_iterations must be nonnegative",
    ))
    config.reduced_inner_method in (:anderson, :response) || throw(ArgumentError(
        "reduced_inner_method must be :anderson or :response",
    ))
end

function response_correction!(cache, step, rho, config, refresh)
    output_density = PMF.density(cache.problem, step.orbitals)
    fixed_point_residual = output_density - rho
    density_defect = norm(fixed_point_residual) / max(norm(rho), eps(Float64))
    if refresh <= config.response_warmup_refreshes
        return (
            correction=config.response_warmup_mixing .* fixed_point_residual,
            density_defect=density_defect,
            krylov_steps=0,
            krylov_residual=0.0,
            mode=:mixing,
        )
    end
    action = direction -> direction - density_response!(
        cache,
        step.orbitals,
        step.values,
        direction,
    )
    krylov = PMF.cg_action(
        action,
        fixed_point_residual;
        tolerance=config.response_krylov_tolerance,
        iterations=config.response_krylov_iterations,
    )
    (
        correction=config.response_damping .* krylov.solution,
        density_defect=density_defect,
        krylov_steps=krylov.iterations,
        krylov_residual=krylov.residual,
        mode=:response,
    )
end

function reduced_config(config::CacheNativeConfig)
    PMF.WindowedNLFEASTConfig(
        moment_depth=config.moment_depth,
        probe_width=config.probe_width,
        window_blocks=config.window_blocks,
        refresh_iterations=config.refresh_iterations,
        inner_schedule=:fixed,
        inner_method=config.reduced_inner_method,
        fixed_inner_iterations=max(config.reduced_inner_iterations, 1),
        inner_history_depth=config.reduced_history_depth,
        inner_mixing=config.reduced_mixing,
        inner_regularization=config.reduced_regularization,
        maximum_step_ratio=config.maximum_step_ratio,
        density_tolerance=config.density_tolerance,
        residual_tolerance=config.residual_tolerance,
        ranktol=config.ranktol,
        subspace_ranktol=config.subspace_ranktol,
    )
end

function solve_cache_native_nlfeast(
    problem,
    chart_spec,
    initial;
    config::CacheNativeConfig,
)
    validate_config(problem, config)
    orbitals = PMF.orthonormalize(initial, problem.occupied)
    rho = PMF.density(problem, orbitals)
    blocks = Matrix{ComplexF64}[]
    basis = orbitals
    history = NamedTuple[]
    converged = false
    inner_config = reduced_config(config)
    last_response_refresh = -config.response_period

    for refresh in 1:config.refresh_iterations
        H = PMF.hamiltonian(problem, rho)
        chart_resolution = PMF.resolve_chart(chart_spec, H, problem.occupied)
        cache = build_density_cache(problem, chart_resolution.chart, rho)
        count_diagnostic = PMF.determinant_count(cache.factors)
        count_diagnostic.count == problem.occupied || throw(PMF.ContourCountError(
            problem.occupied,
            count_diagnostic.count,
            count_diagnostic.winding,
            count_diagnostic.maximum_phase_step,
        ))
        step = cached_projector_step!(
            cache,
            orbitals,
            config.moment_depth,
            config.probe_width;
            ranktol=config.ranktol,
        )
        push!(blocks, step.window_block)
        while config.window_blocks > 0 && length(blocks) > config.window_blocks
            popfirst!(blocks)
        end
        window = PMF.window_basis(blocks, config.subspace_ranktol)
        basis = window.basis

        periodic_response = config.pre_update === :response && (
            refresh <= config.response_warmup_refreshes ||
            (refresh - config.response_warmup_refreshes - 1) % config.response_period == 0
        )
        adaptive_response = config.pre_update === :adaptive_response &&
            refresh > config.response_warmup_refreshes && !isempty(history) &&
            refresh - last_response_refresh >= config.response_period &&
            history[end].closure_defect > config.density_tolerance &&
            history[end].closure_defect >
                config.response_forcing_ratio * history[end].leakage
        response_scheduled = periodic_response || adaptive_response
        response = if response_scheduled
            response_correction!(cache, step, rho, config, refresh)
        else
            (
                correction=zeros(Float64, length(rho)),
                density_defect=norm(PMF.density(problem, step.orbitals) - rho) /
                    max(norm(rho), eps(Float64)),
                krylov_steps=0,
                krylov_residual=0.0,
                mode=:none,
            )
        end
        if response.mode !== :none
            correction, _ = PMF.limit_density_correction(
                problem,
                cache.chart,
                step.values,
                rho,
                response.correction;
                preserve_chart=false,
                chart_safety=0.8,
            )
            rho += correction
            response.mode === :response && (last_response_refresh = refresh)
        end
        invalidate!(cache)

        inner = if config.reduced_inner_iterations > 0
            PMF.solve_windowed_inner(problem, basis, rho, inner_config)
        else
            state = PMF.reduced_ritz_state(problem, basis, rho)
            closure_defect = norm(state.output_density - rho) /
                max(norm(rho), eps(Float64))
            (
                orbitals=state.orbitals,
                density=rho,
                closure_defect=closure_defect,
                leakage=state.leakage,
                iterations=0,
                stop_reason=:disabled,
                history=NamedTuple[],
            )
        end
        orbitals = inner.orbitals
        rho = inner.density
        converged = inner.closure_defect <= config.density_tolerance &&
            inner.leakage <= config.residual_tolerance
        stats = cache_stats(cache)
        push!(history, (
            refresh=refresh,
            pre_update=response.mode,
            response_density_defect=response.density_defect,
            response_krylov_steps=response.krylov_steps,
            response_krylov_residual=response.krylov_residual,
            reduced_inner_iterations=inner.iterations,
            closure_defect=inner.closure_defect,
            leakage=inner.leakage,
            basis_dimension=size(basis, 2),
            window_blocks=length(blocks),
            factorization_count=stats.factorization_count,
            chart_factorization_count=chart_resolution.factorization_count,
            base_rhs_count=stats.base_rhs_count,
            correction_rhs_count=stats.correction_rhs_count,
            response_rhs_count=stats.response_rhs_count,
        ))
        converged && break
    end

    final_state = PMF.reduced_ritz_state(problem, basis, rho)
    (
        variant=:cache_native_corrected_moments,
        pre_update=config.pre_update,
        reduced_inner_method=config.reduced_inner_method,
        orbitals=final_state.orbitals,
        density=rho,
        basis=basis,
        history=history,
        converged=converged,
        closure_defect=norm(final_state.output_density - rho) /
            max(norm(rho), eps(Float64)),
        residual=final_state.leakage,
        refresh_count=length(history),
        factorization_count=sum(record.factorization_count for record in history),
        chart_factorization_count=sum(
            record.chart_factorization_count for record in history
        ),
        base_rhs_count=sum(record.base_rhs_count for record in history),
        correction_rhs_count=sum(record.correction_rhs_count for record in history),
        response_rhs_count=sum(record.response_rhs_count for record in history),
    )
end
