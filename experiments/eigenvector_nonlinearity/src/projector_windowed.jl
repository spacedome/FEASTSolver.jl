Base.@kwdef struct ProjectorWindowConfig
    moment_depth::Int
    probe_width::Int
    window_blocks::Int = 4
    refresh_iterations::Int = 30
    inner_iterations::Int = 3
    inner_method::Symbol = :anderson
    inner_mixing::Float64 = 0.2
    inner_history_depth::Int = 6
    inner_regularization::Float64 = 1e-12
    maximum_step_ratio::Float64 = 4.0
    projector_tolerance::Float64 = 1e-9
    residual_tolerance::Float64 = 1e-9
    subspace_ranktol::Float64 = 1e-10
end

function solve_projector_windowed_nlfeast(
    problem::NonlocalProjector1D,
    chart_spec,
    initial;
    config::ProjectorWindowConfig,
)
    config.moment_depth * config.probe_width >= problem.occupied || throw(
        ArgumentError("moment capacity is below the occupied count"),
    )
    config.window_blocks >= 0 || throw(ArgumentError(
        "window_blocks must be nonnegative; zero selects growing memory",
    ))
    config.refresh_iterations > 0 || throw(ArgumentError(
        "refresh_iterations must be positive",
    ))
    config.inner_iterations > 0 || throw(ArgumentError(
        "inner_iterations must be positive",
    ))
    config.inner_history_depth > 0 || throw(ArgumentError(
        "inner_history_depth must be positive",
    ))
    0 < config.inner_mixing <= 1 || throw(ArgumentError(
        "inner_mixing must lie in (0,1]",
    ))
    config.inner_method in (:mixing, :anderson) || throw(ArgumentError(
        "inner_method must be :mixing or :anderson",
    ))
    orbitals = orthonormalize(initial, problem.occupied)
    projector = projector_state(orbitals)
    tangent = moment_tangent(problem.occupied, config.probe_width)
    blocks = Matrix{ComplexF64}[]
    basis = orbitals
    history = NamedTuple[]
    converged = false

    for refresh in 1:config.refresh_iterations
        H = hamiltonian(problem, projector)
        chart_resolution = resolve_chart(chart_spec, H, problem.occupied)
        filtered = raw_contour_moment_block(
            H,
            chart_resolution.chart,
            orbitals * tangent,
            config.moment_depth,
            problem.occupied,
        )
        push!(blocks, filtered.block)
        while config.window_blocks > 0 && length(blocks) > config.window_blocks
            popfirst!(blocks)
        end
        window = window_basis(blocks, config.subspace_ranktol)
        basis = window.basis
        closure_defect = Inf
        state = projector_ritz_state(problem, basis, projector)
        performed = 0
        projectors = Matrix{ComplexF64}[]
        residuals = Matrix{ComplexF64}[]
        for inner in 1:config.inner_iterations
            state = projector_ritz_state(problem, basis, projector)
            fixed_point_residual = state.output_projector - projector
            closure_defect = norm(fixed_point_residual) /
                max(norm(projector), eps(Float64))
            performed = inner
            if closure_defect <= config.projector_tolerance &&
                    state.leakage <= config.residual_tolerance
                converged = true
                break
            end
            correction = if config.inner_method === :anderson
                push!(projectors, copy(projector))
                push!(residuals, fixed_point_residual)
                while length(projectors) > config.inner_history_depth + 1
                    popfirst!(projectors)
                    popfirst!(residuals)
                end
                projector_anderson_correction(
                    projectors,
                    residuals,
                    config.inner_mixing,
                    config.inner_regularization,
                )
            else
                config.inner_mixing .* fixed_point_residual
            end
            maximum_step = config.maximum_step_ratio * max(
                norm(fixed_point_residual),
                eps(Float64),
            )
            norm(correction) > maximum_step && (
                correction .*= maximum_step / norm(correction)
            )
            projector = normalize_projector_trace(problem, projector + correction)
        end
        state = projector_ritz_state(problem, basis, projector)
        orbitals = state.orbitals
        closure_defect = norm(state.output_projector - projector) /
            max(norm(projector), eps(Float64))
        converged = closure_defect <= config.projector_tolerance &&
            state.leakage <= config.residual_tolerance
        push!(history, (
            refresh=refresh,
            inner_iterations=performed,
            closure_defect=closure_defect,
            leakage=state.leakage,
            basis_dimension=size(basis, 2),
            solve_count=filtered.solve_count,
            rhs_count=filtered.rhs_count,
            chart_factorization_count=chart_resolution.factorization_count,
        ))
        converged && break
    end

    final_state = projector_ritz_state(problem, basis, projector)
    (
        variant=config.window_blocks == 0 ?
            :nonlocal_pre_extraction_growing : :nonlocal_pre_extraction_window,
        inner_method=config.inner_method,
        orbitals=final_state.orbitals,
        projector=projector,
        basis=basis,
        history=history,
        converged=converged,
        closure_defect=norm(final_state.output_projector - projector) /
            max(norm(projector), eps(Float64)),
        residual=final_state.leakage,
        refresh_count=length(history),
        inner_iterations=sum(record.inner_iterations for record in history),
        solve_count=sum(record.solve_count for record in history),
        rhs_count=sum(record.rhs_count for record in history),
        chart_factorization_count=sum(record.chart_factorization_count for record in history),
    )
end
