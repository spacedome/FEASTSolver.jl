export DualCacheConfig, solve_cache_native_dual

Base.@kwdef struct DualCacheConfig
    moment_depth::Int
    probe_width::Int
    window_blocks::Int = 4
    refresh_iterations::Int = 30
    pre_update::Symbol = :none
    response_warmup_refreshes::Int = 2
    response_warmup_mixing::Float64 = 0.5
    response_period::Int = 1
    response_forcing_ratio::Float64 = 1.0
    response_krylov_tolerance::Float64 = 0.03
    response_krylov_iterations::Int = 30
    response_damping::Float64 = 1.0
    reduced_inner_iterations::Int = 3
    reduced_mixing::Float64 = 0.4
    reduced_history_depth::Int = 8
    reduced_regularization::Float64 = 1e-12
    maximum_step_ratio::Float64 = 4.0
    density_tolerance::Float64 = 1e-8
    residual_tolerance::Float64 = 1e-8
    subspace_ranktol::Float64 = 1e-10
    overlap_ranktol::Float64 = 1e-10
end

function validate_dual_config(problem, config::DualCacheConfig)
    config.moment_depth * config.probe_width >= problem.base.occupied || throw(
        ArgumentError("dual moment capacity is below the occupied count"),
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
    config.response_period > 0 || throw(ArgumentError(
        "response_period must be positive",
    ))
    config.response_forcing_ratio > 0 || throw(ArgumentError(
        "response_forcing_ratio must be positive",
    ))
    config.reduced_inner_iterations > 0 || throw(ArgumentError(
        "reduced_inner_iterations must be positive",
    ))
end

function dual_reduced_inner(problem, chart, right_basis, left_basis, initial_density, config)
    rho = copy(initial_density)
    densities = Vector{Float64}[]
    residuals = Vector{Float64}[]
    performed = 0
    state = PMF.dual_reduced_state(problem, chart, right_basis, left_basis, rho)
    for inner in 1:config.reduced_inner_iterations
        state = PMF.dual_reduced_state(problem, chart, right_basis, left_basis, rho)
        fixed_point_residual = state.density - rho
        density_defect = norm(fixed_point_residual) / max(norm(rho), eps(Float64))
        performed = inner
        if density_defect <= config.density_tolerance &&
                state.residual <= config.residual_tolerance
            break
        end
        push!(densities, copy(rho))
        push!(residuals, fixed_point_residual)
        while length(densities) > config.reduced_history_depth + 1
            popfirst!(densities)
            popfirst!(residuals)
        end
        correction = PMF.anderson_correction(
            densities,
            residuals,
            config.reduced_mixing,
            config.reduced_regularization,
        )
        maximum_step = config.maximum_step_ratio * max(
            norm(fixed_point_residual),
            eps(Float64),
        )
        norm(correction) > maximum_step && (
            correction .*= maximum_step / norm(correction)
        )
        rho += correction
        PMF.normalize_density!(problem.base, rho)
    end
    state = PMF.dual_reduced_state(problem, chart, right_basis, left_basis, rho)
    density_defect = norm(state.density - rho) / max(norm(rho), eps(Float64))
    (
        right=state.right,
        left=state.left,
        density=rho,
        density_defect=density_defect,
        residual=state.residual,
        iterations=performed,
    )
end

function solve_cache_native_dual(
    problem,
    chart::PMF.CircularChart,
    initial_right,
    initial_left;
    config::DualCacheConfig,
)
    validate_dual_config(problem, config)
    occupied = problem.base.occupied
    right, left = PMF.biorthogonalize(initial_right, initial_left, occupied)
    rho = PMF.oblique_density(problem, right, left)
    tangent = PMF.moment_tangent(occupied, config.probe_width)
    right_blocks = Matrix{ComplexF64}[]
    left_blocks = Matrix{ComplexF64}[]
    right_basis = right
    left_basis = left
    history = NamedTuple[]
    converged = false
    last_response_refresh = -config.response_period

    for refresh in 1:config.refresh_iterations
        cache = build_dual_cache(problem, chart, rho)
        count_diagnostic = PMF.determinant_count(cache.factors)
        count_diagnostic.count == occupied || throw(PMF.ContourCountError(
            occupied,
            count_diagnostic.count,
            count_diagnostic.winding,
            count_diagnostic.maximum_phase_step,
        ))
        filtered = dual_positive_moments!(
            cache,
            right * tangent,
            left * tangent,
            config.moment_depth,
        )
        current_window = PMF.dual_window_basis(
            [filtered.right],
            [filtered.left],
            config.subspace_ranktol,
            config.overlap_ranktol,
        )
        modal = dual_modal_state(cache, current_window.right, current_window.left)

        push!(right_blocks, filtered.right)
        push!(left_blocks, filtered.left)
        while config.window_blocks > 0 && length(right_blocks) > config.window_blocks
            popfirst!(right_blocks)
            popfirst!(left_blocks)
        end
        window = PMF.dual_window_basis(
            right_blocks,
            left_blocks,
            config.subspace_ranktol,
            config.overlap_ranktol,
        )
        right_basis = window.right
        left_basis = window.left

        periodic_response = config.pre_update === :response && (
            refresh <= config.response_warmup_refreshes ||
            (refresh - config.response_warmup_refreshes - 1) % config.response_period == 0
        )
        adaptive_response = config.pre_update === :adaptive_response &&
            refresh > config.response_warmup_refreshes && !isempty(history) &&
            refresh - last_response_refresh >= config.response_period &&
            history[end].density_defect > config.density_tolerance &&
            history[end].density_defect >
                config.response_forcing_ratio * history[end].residual
        response_scheduled = periodic_response || adaptive_response
        response_mode = :none
        krylov_steps = 0
        if response_scheduled
            fixed_point_residual = modal.density - rho
            correction = if refresh <= config.response_warmup_refreshes
                response_mode = :mixing
                config.response_warmup_mixing .* fixed_point_residual
            else
                response_mode = :response
                action = direction -> direction - dual_density_response!(
                    cache,
                    modal.right,
                    modal.left,
                    modal.values,
                    direction,
                )
                krylov = PMF.cg_action(
                    action,
                    fixed_point_residual;
                    tolerance=config.response_krylov_tolerance,
                    iterations=config.response_krylov_iterations,
                )
                krylov_steps = krylov.iterations
                last_response_refresh = refresh
                config.response_damping .* krylov.solution
            end
            maximum_step = config.maximum_step_ratio * max(
                norm(fixed_point_residual),
                eps(Float64),
            )
            norm(correction) > maximum_step && (
                correction .*= maximum_step / norm(correction)
            )
            rho += correction
            PMF.normalize_density!(problem.base, rho)
        end
        invalidate!(cache)
        inner = dual_reduced_inner(
            problem,
            chart,
            right_basis,
            left_basis,
            rho,
            config,
        )
        right = inner.right
        left = inner.left
        rho = inner.density
        converged = inner.density_defect <= config.density_tolerance &&
            inner.residual <= config.residual_tolerance
        stats = dual_cache_stats(cache)
        push!(history, (
            refresh=refresh,
            pre_update=response_mode,
            response_krylov_steps=krylov_steps,
            reduced_inner_iterations=inner.iterations,
            density_defect=inner.density_defect,
            residual=inner.residual,
            basis_dimension=window.rank,
            factorization_count=stats.factorization_count,
            right_rhs_count=stats.right_rhs_count,
            left_rhs_count=stats.left_rhs_count,
            response_rhs_count=stats.response_rhs_count,
        ))
        converged && break
    end

    final = PMF.dual_reduced_state(
        problem,
        chart,
        right_basis,
        left_basis,
        rho,
    )
    (
        variant=:cache_native_dual,
        right=final.right,
        left=final.left,
        density=rho,
        right_basis=right_basis,
        left_basis=left_basis,
        history=history,
        converged=converged,
        density_defect=norm(final.density - rho) / max(norm(rho), eps(Float64)),
        residual=final.residual,
        refresh_count=length(history),
        factorization_count=sum(record.factorization_count for record in history),
        right_rhs_count=sum(record.right_rhs_count for record in history),
        left_rhs_count=sum(record.left_rhs_count for record in history),
        response_rhs_count=sum(record.response_rhs_count for record in history),
    )
end
