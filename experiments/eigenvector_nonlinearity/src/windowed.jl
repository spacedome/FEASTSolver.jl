Base.@kwdef struct WindowedNLFEASTConfig
    moment_depth::Int
    probe_width::Int
    window_blocks::Int = 4
    refresh_iterations::Int = 20
    inner_schedule::Symbol = :fixed
    inner_method::Symbol = :anderson
    fixed_inner_iterations::Int = 3
    maximum_inner_iterations::Int = 40
    minimum_inner_iterations::Int = 1
    forcing_ratio::Float64 = 1.0
    inner_history_depth::Int = 5
    inner_mixing::Float64 = 0.5
    inner_regularization::Float64 = 1e-12
    inner_krylov_tolerance::Float64 = 1e-8
    inner_krylov_iterations::Int = 30
    inner_damping::Float64 = 1.0
    maximum_step_ratio::Float64 = 4.0
    density_tolerance::Float64 = 1e-10
    residual_tolerance::Float64 = 1e-10
    ranktol::Float64 = 1e-12
    subspace_ranktol::Float64 = 1e-10
    seed::Int = 4817
end

function validate_windowed_config(problem, config::WindowedNLFEASTConfig)
    config.moment_depth * config.probe_width >= problem.occupied || throw(ArgumentError(
        "moment capacity is below the occupied count",
    ))
    config.window_blocks >= 0 || throw(ArgumentError(
        "window_blocks must be nonnegative; zero selects growing memory",
    ))
    config.refresh_iterations > 0 || throw(ArgumentError(
        "refresh_iterations must be positive",
    ))
    config.inner_schedule in (:fixed, :leakage) || throw(ArgumentError(
        "inner_schedule must be :fixed or :leakage",
    ))
    config.inner_method in (:anderson, :response) || throw(ArgumentError(
        "inner_method must be :anderson or :response",
    ))
    config.fixed_inner_iterations > 0 || throw(ArgumentError(
        "fixed_inner_iterations must be positive",
    ))
    1 <= config.minimum_inner_iterations <= config.maximum_inner_iterations || throw(
        ArgumentError("inner iteration limits are inconsistent"),
    )
    config.forcing_ratio > 0 || throw(ArgumentError("forcing_ratio must be positive"))
    config.inner_history_depth > 0 || throw(ArgumentError(
        "inner_history_depth must be positive",
    ))
    0 < config.inner_mixing <= 1 || throw(ArgumentError(
        "inner_mixing must lie in (0,1]",
    ))
    config.inner_krylov_tolerance > 0 || throw(ArgumentError(
        "inner_krylov_tolerance must be positive",
    ))
    config.inner_krylov_iterations > 0 || throw(ArgumentError(
        "inner_krylov_iterations must be positive",
    ))
    0 < config.inner_damping <= 1 || throw(ArgumentError(
        "inner_damping must lie in (0,1]",
    ))
end

function reduced_density_response(problem::ContactMeanField1D, state, direction)
    occupied = problem.occupied
    size(state.ritz_vectors, 2) > occupied || return zeros(Float64, length(direction))
    X = state.ritz_vectors[:, 1:occupied]
    virtual = state.ritz_vectors[:, (occupied + 1):end]
    coupling = adjoint(virtual) * (problem.coupling .* direction .* X)
    denominators = transpose(state.reduced_values[1:occupied]) .-
        state.reduced_values[(occupied + 1):end]
    orbital_response = virtual * (coupling ./ denominators)
    (2 / problem.spacing) .* vec(real.(sum(conj.(X) .* orbital_response; dims=2)))
end

function window_basis(blocks, ranktol)
    decomposition = svd(reduce(hcat, blocks))
    threshold = ranktol * decomposition.S[1]
    retained = count(singular -> singular >= threshold, decomposition.S)
    retained > 0 || error("windowed subspace has zero numerical rank")
    (
        basis=decomposition.U[:, 1:retained],
        singular_values=Float64.(decomposition.S),
        rank=retained,
    )
end

function solve_windowed_inner(
    problem,
    basis,
    initial_density,
    config::WindowedNLFEASTConfig,
)
    rho = copy(initial_density)
    densities = Vector{Float64}[]
    residuals = Vector{Float64}[]
    history = NamedTuple[]
    stop_reason = :inner_limit
    limit = config.inner_schedule === :fixed ?
        config.fixed_inner_iterations : config.maximum_inner_iterations

    for iteration in 1:limit
        state = reduced_ritz_state(problem, basis, rho)
        fixed_point_residual = state.output_density - rho
        closure_defect = norm(fixed_point_residual) / max(norm(rho), eps(Float64))
        push!(history, (
            iteration=iteration,
            closure_defect=closure_defect,
            leakage=state.leakage,
            forcing_quotient=closure_defect / max(state.leakage, eps(Float64)),
            krylov_steps=0,
        ))
        if closure_defect <= config.density_tolerance &&
                state.leakage <= config.residual_tolerance
            stop_reason = :converged
            break
        end
        if config.inner_schedule === :leakage &&
                iteration >= config.minimum_inner_iterations &&
                closure_defect <= config.forcing_ratio * state.leakage
            stop_reason = :leakage_forcing
            break
        end

        push!(densities, copy(rho))
        push!(residuals, fixed_point_residual)
        while length(densities) > config.inner_history_depth + 1
            popfirst!(densities)
            popfirst!(residuals)
        end
        correction, krylov_steps = if config.inner_method === :response
            action = direction -> direction - reduced_density_response(
                problem,
                state,
                direction,
            )
            krylov = cg_action(
                action,
                fixed_point_residual;
                tolerance=config.inner_krylov_tolerance,
                iterations=config.inner_krylov_iterations,
            )
            (config.inner_damping .* krylov.solution, krylov.iterations)
        else
            (
                anderson_correction(
                    densities,
                    residuals,
                    config.inner_mixing,
                    config.inner_regularization,
                ),
                0,
            )
        end
        history[end] = merge(history[end], (krylov_steps=krylov_steps,))
        maximum_step = config.maximum_step_ratio * max(
            norm(fixed_point_residual),
            eps(Float64),
        )
        norm(correction) > maximum_step && (
            correction .*= maximum_step / norm(correction)
        )
        rho += correction
        normalize_density!(problem, rho)
        iteration == limit && (stop_reason = config.inner_schedule === :fixed ?
            :fixed_count : :inner_limit)
    end

    state = reduced_ritz_state(problem, basis, rho)
    closure_defect = norm(state.output_density - rho) / max(norm(rho), eps(Float64))
    (
        orbitals=state.orbitals,
        values=state.values,
        density=rho,
        closure_defect=closure_defect,
        leakage=state.leakage,
        iterations=length(history),
        stop_reason=stop_reason,
        history=history,
    )
end

function solve_windowed_nlfeast(
    problem::ContactMeanField1D,
    chart_spec,
    initial;
    config::WindowedNLFEASTConfig,
)
    validate_windowed_config(problem, config)
    orbitals = orthonormalize(initial, problem.occupied)
    rho = density(problem, orbitals)
    tangent = moment_tangent(problem.occupied, config.probe_width)
    blocks = Matrix{ComplexF64}[]
    history = NamedTuple[]
    inner_history = NamedTuple[]
    basis = orbitals
    converged = false

    for refresh in 1:config.refresh_iterations
        H = hamiltonian(problem, rho)
        chart_resolution = resolve_chart(chart_spec, H, problem.occupied)
        step = moment_projector_step(
            H,
            chart_resolution.chart,
            orbitals * tangent,
            config.moment_depth,
            problem.occupied;
            ranktol=config.ranktol,
        )
        push!(blocks, step.orbitals)
        while config.window_blocks > 0 && length(blocks) > config.window_blocks
            popfirst!(blocks)
        end
        window = window_basis(blocks, config.subspace_ranktol)
        basis = window.basis
        inner = solve_windowed_inner(problem, basis, rho, config)
        orbitals = inner.orbitals
        rho = inner.density
        append!(inner_history, [merge(record, (refresh=refresh,)) for record in inner.history])
        converged = inner.closure_defect <= config.density_tolerance &&
            inner.leakage <= config.residual_tolerance
        push!(history, (
            refresh=refresh,
            stop_reason=inner.stop_reason,
            inner_iterations=inner.iterations,
            closure_defect=inner.closure_defect,
            leakage=inner.leakage,
            basis_dimension=size(basis, 2),
            window_blocks=length(blocks),
            window_singular_ratio=window.singular_values[window.rank] /
                window.singular_values[1],
            moment_singular_ratio=step.singular_values[problem.occupied] /
                step.singular_values[1],
            reduced_response_actions=sum(record.krylov_steps for record in inner.history),
            solve_count=step.solve_count,
            rhs_count=step.rhs_count,
            chart_factorization_count=chart_resolution.factorization_count,
        ))
        converged && break
    end

    final_state = reduced_ritz_state(problem, basis, rho)
    (
        variant=config.window_blocks == 0 ?
            :post_extraction_growing : :post_extraction_window,
        inner_method=config.inner_method,
        memory_blocks=config.window_blocks,
        orbitals=final_state.orbitals,
        density=rho,
        basis=basis,
        history=history,
        inner_history=inner_history,
        converged=converged,
        closure_defect=norm(final_state.output_density - rho) /
            max(norm(rho), eps(Float64)),
        residual=final_state.leakage,
        energy=contact_energy(problem, final_state.orbitals),
        refresh_count=length(history),
        inner_iterations=sum(record.inner_iterations for record in history),
        reduced_response_actions=sum(
            record.reduced_response_actions for record in history
        ),
        solve_count=sum(record.solve_count for record in history),
        solve_application_count=sum(record.solve_count for record in history),
        rhs_count=sum(record.rhs_count for record in history),
        chart_factorization_count=sum(
            record.chart_factorization_count for record in history
        ),
    )
end
