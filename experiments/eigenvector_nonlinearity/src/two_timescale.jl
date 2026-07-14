Base.@kwdef struct TwoTimescaleConfig
    moment_depth::Int
    probe_width::Int
    subspace_dimension::Int
    refresh_iterations::Int = 20
    maximum_inner_iterations::Int = 40
    minimum_inner_iterations::Int = 1
    forcing_ratio::Float64 = 1.0
    inner_history_depth::Int = 8
    inner_mixing::Float64 = 0.5
    inner_regularization::Float64 = 1e-12
    maximum_step_ratio::Float64 = 4.0
    density_tolerance::Float64 = 1e-10
    residual_tolerance::Float64 = 1e-10
    ranktol::Float64 = 1e-12
    enrichment_ranktol::Float64 = 1e-10
    history_policy::Symbol = :restart
    maximum_subspace_dimension::Int = 0
    growth_increment::Int = 0
    stagnation_ratio::Float64 = 0.4
    stagnation_iterations::Int = 2
    seed::Int = 4817
end

function reduced_ritz_state(problem::ContactMeanField1D, basis, rho)
    H = hamiltonian(problem, rho)
    reduced_hamiltonian = Hermitian(adjoint(basis) * H * basis)
    decomposition = eigen(reduced_hamiltonian)
    ritz_vectors = basis * decomposition.vectors
    occupied = 1:problem.occupied
    orbitals = ritz_vectors[:, occupied]
    values = Float64.(decomposition.values[occupied])
    output_density = density(problem, orbitals)
    residual = H * orbitals - orbitals * Diagonal(values)
    residual_norm = norm(residual)
    hamiltonian_action_norm = norm(H * orbitals)
    leakage = residual_norm / max(hamiltonian_action_norm, eps(Float64))
    (
        orbitals=orbitals,
        values=values,
        output_density=output_density,
        leakage=leakage,
        residual_norm=residual_norm,
        hamiltonian_action_norm=hamiltonian_action_norm,
        reduced_values=Float64.(decomposition.values),
        ritz_vectors=ritz_vectors,
        hamiltonian=H,
    )
end

function normalize_density!(problem::ContactMeanField1D, rho)
    rho .= max.(real.(rho), 0.0)
    mass = problem.spacing * sum(rho)
    mass > eps(Float64) || error("density update lost all mass")
    rho .*= problem.occupied / mass
    rho
end

function occupied_moment_enrichment(
    problem::ContactMeanField1D,
    chart::CircularChart,
    basis,
    rho;
    moment_depth,
    probe_width,
    ranktol=1e-12,
    enrichment_ranktol=1e-10,
    output_dimension=size(basis, 2),
)
    state = reduced_ritz_state(problem, basis, rho)
    tangent = moment_tangent(problem.occupied, probe_width)
    step = moment_projector_step(
        state.hamiltonian,
        chart,
        state.orbitals * tangent,
        moment_depth,
        problem.occupied;
        ranktol=ranktol,
    )

    q = size(basis, 2)
    q <= output_dimension <= min(size(state.hamiltonian, 1), q + problem.occupied) || throw(
        ArgumentError("output_dimension is outside the available enriched space"),
    )
    if q == problem.occupied && output_dimension == q
        return (
            basis=step.orbitals,
            filtered_orbitals=step.orbitals,
            correction_rank=problem.occupied,
            correction_singular_values=ones(Float64, problem.occupied),
            step=step,
        )
    end

    complement = step.orbitals - basis * (adjoint(basis) * step.orbitals)
    decomposition = svd(complement)
    scale = isempty(decomposition.S) ? 0.0 : decomposition.S[1]
    correction_rank = scale == 0 ? 0 : count(
        singular -> singular >= enrichment_ranktol * scale,
        decomposition.S,
    )
    correction_rank > 0 || error(
        "occupied FEAST correction is numerically contained in the reduced space",
    )
    raw_correction = decomposition.U[:, 1:correction_rank]
    projected_correction = raw_correction - basis * (adjoint(basis) * raw_correction)
    correction = orthonormalize(projected_correction, correction_rank)
    expanded = orthonormalize(hcat(basis, correction), q + correction_rank)
    reduced_hamiltonian = Hermitian(adjoint(expanded) * state.hamiltonian * expanded)
    reduced = eigen(reduced_hamiltonian)
    available_dimension = size(expanded, 2)
    realized_dimension = min(output_dimension, available_dimension)
    updated_basis = orthonormalize(
        expanded * reduced.vectors[:, 1:realized_dimension],
        realized_dimension,
    )
    (
        basis=updated_basis,
        filtered_orbitals=step.orbitals,
        correction_rank=correction_rank,
        correction_singular_values=Float64.(decomposition.S),
        step=step,
    )
end

function validate_two_timescale_config(problem, config::TwoTimescaleConfig)
    problem.occupied <= config.subspace_dimension <= length(problem.grid) || throw(
        ArgumentError("subspace_dimension must lie between occupied count and problem size"),
    )
    config.moment_depth * config.probe_width >= problem.occupied || throw(ArgumentError(
        "moment capacity is below the occupied count",
    ))
    1 <= config.minimum_inner_iterations <= config.maximum_inner_iterations || throw(
        ArgumentError("inner iteration limits are inconsistent"),
    )
    config.refresh_iterations >= 0 || throw(ArgumentError(
        "refresh_iterations must be nonnegative",
    ))
    config.forcing_ratio > 0 || throw(ArgumentError("forcing_ratio must be positive"))
    config.inner_history_depth > 0 || throw(ArgumentError(
        "inner_history_depth must be positive",
    ))
    0 < config.inner_mixing <= 1 || throw(ArgumentError(
        "inner_mixing must lie in (0,1]",
    ))
    config.history_policy in (:restart, :physical) || throw(ArgumentError(
        "history_policy must be :restart or :physical",
    ))
    if config.maximum_subspace_dimension > 0
        config.subspace_dimension <= config.maximum_subspace_dimension <=
            length(problem.grid) || throw(ArgumentError(
                "maximum_subspace_dimension must lie between the initial dimension and problem size",
            ))
        config.growth_increment >= 0 || throw(ArgumentError(
            "growth_increment must be nonnegative",
        ))
        0 < config.stagnation_ratio < 1 || throw(ArgumentError(
            "stagnation_ratio must lie in (0,1)",
        ))
        config.stagnation_iterations > 0 || throw(ArgumentError(
            "stagnation_iterations must be positive",
        ))
    end
end

function solve_two_timescale_nlfeast(
    problem::ContactMeanField1D,
    chart_spec,
    initial;
    config::TwoTimescaleConfig,
)
    validate_two_timescale_config(problem, config)
    basis = augmented_basis(initial, config.subspace_dimension, config.seed)
    orbitals = orthonormalize(initial, problem.occupied)
    rho = density(problem, orbitals)
    densities = Vector{Float64}[]
    residuals = Vector{Float64}[]
    inner_history = NamedTuple[]
    refresh_history = NamedTuple[]
    converged = false
    total_inner_iterations = 0
    chart_factorization_count = 0
    dimension_leakages = Float64[]

    for phase in 1:(config.refresh_iterations + 1)
        phase_start = length(inner_history) + 1
        stop_reason = :inner_limit
        state = reduced_ritz_state(problem, basis, rho)
        closure_defect = Inf

        for inner_iteration in 1:config.maximum_inner_iterations
            state = reduced_ritz_state(problem, basis, rho)
            fixed_point_residual = state.output_density - rho
            closure_defect = norm(fixed_point_residual) / max(norm(rho), eps(Float64))
            total_inner_iterations += 1
            forcing_quotient = closure_defect / max(state.leakage, eps(Float64))
            push!(inner_history, (
                iteration=total_inner_iterations,
                phase=phase,
                inner_iteration=inner_iteration,
                closure_defect=closure_defect,
                leakage=state.leakage,
                forcing_quotient=forcing_quotient,
                energy=contact_energy(problem, state.orbitals),
            ))

            push!(densities, copy(rho))
            push!(residuals, fixed_point_residual)
            while length(densities) > config.inner_history_depth + 1
                popfirst!(densities)
                popfirst!(residuals)
            end

            if closure_defect <= config.density_tolerance &&
                    state.leakage <= config.residual_tolerance
                stop_reason = :converged
                converged = true
                orbitals = state.orbitals
                break
            end
            if inner_iteration >= config.minimum_inner_iterations &&
                    closure_defect <= config.forcing_ratio * state.leakage
                stop_reason = :leakage_forcing
                orbitals = state.orbitals
                break
            end

            correction = anderson_correction(
                densities,
                residuals,
                config.inner_mixing,
                config.inner_regularization,
            )
            maximum_step = config.maximum_step_ratio * max(
                norm(fixed_point_residual),
                eps(Float64),
            )
            norm(correction) > maximum_step && (
                correction .*= maximum_step / norm(correction)
            )
            rho += correction
            normalize_density!(problem, rho)
            orbitals = state.orbitals
        end

        if stop_reason === :inner_limit
            state = reduced_ritz_state(problem, basis, rho)
            closure_defect = norm(state.output_density - rho) /
                max(norm(rho), eps(Float64))
            orbitals = state.orbitals
        end
        phase_end = length(inner_history)
        push!(dimension_leakages, state.leakage)
        if converged || phase > config.refresh_iterations
            push!(refresh_history, (
                phase=phase,
                stop_reason=stop_reason,
                inner_iterations=phase_end - phase_start + 1,
                closure_defect=closure_defect,
                leakage=state.leakage,
                correction_rank=0,
                singular_ratio=NaN,
                solve_count=0,
                rhs_count=0,
                chart_factorization_count=0,
                basis_dimension=size(basis, 2),
                next_basis_dimension=size(basis, 2),
            ))
            break
        end

        chart_resolution = resolve_chart(chart_spec, state.hamiltonian, problem.occupied)
        chart = chart_resolution.chart
        current_dimension = size(basis, 2)
        adaptive = config.maximum_subspace_dimension > current_dimension
        slow_contraction = adaptive &&
            length(dimension_leakages) >= config.stagnation_iterations + 1 &&
            all(
                dimension_leakages[index] >=
                    config.stagnation_ratio * dimension_leakages[index - 1]
                for index in (length(dimension_leakages) - config.stagnation_iterations + 1):
                    length(dimension_leakages)
            )
        growth = config.growth_increment == 0 ? problem.occupied : config.growth_increment
        requested_dimension = slow_contraction ?
            min(config.maximum_subspace_dimension, current_dimension + growth) :
            current_dimension
        repair = occupied_moment_enrichment(
            problem,
            chart,
            basis,
            rho;
            moment_depth=config.moment_depth,
            probe_width=config.probe_width,
            ranktol=config.ranktol,
            enrichment_ranktol=config.enrichment_ranktol,
            output_dimension=requested_dimension,
        )
        basis = repair.basis
        size(basis, 2) > current_dimension && empty!(dimension_leakages)
        chart_factorization_count += chart_resolution.factorization_count
        push!(refresh_history, (
            phase=phase,
            stop_reason=stop_reason,
            inner_iterations=phase_end - phase_start + 1,
            closure_defect=closure_defect,
            leakage=state.leakage,
            correction_rank=repair.correction_rank,
            singular_ratio=repair.step.singular_values[problem.occupied] /
                repair.step.singular_values[1],
            solve_count=repair.step.solve_count,
            rhs_count=repair.step.rhs_count,
            chart_factorization_count=chart_resolution.factorization_count,
            basis_dimension=current_dimension,
            next_basis_dimension=size(basis, 2),
        ))
        if config.history_policy === :restart
            empty!(densities)
            empty!(residuals)
        end
    end

    final_state = reduced_ritz_state(problem, basis, rho)
    orbitals = final_state.orbitals
    (
        variant=config.maximum_subspace_dimension > config.subspace_dimension ?
            :adaptive_dimension_enrichment : :fixed_dimension_enrichment,
        inner_method=:anderson,
        orbitals=orbitals,
        density=rho,
        basis=basis,
        inner_history=inner_history,
        refresh_history=refresh_history,
        converged=converged,
        closure_defect=norm(final_state.output_density - rho) /
            max(norm(rho), eps(Float64)),
        residual=final_state.leakage,
        energy=contact_energy(problem, orbitals),
        inner_iterations=total_inner_iterations,
        refresh_count=sum(record.solve_count > 0 for record in refresh_history),
        solve_count=sum(record.solve_count for record in refresh_history),
        solve_application_count=sum(record.solve_count for record in refresh_history),
        rhs_count=sum(record.rhs_count for record in refresh_history),
        chart_factorization_count=chart_factorization_count,
    )
end

function solve_adaptive_dimension_nlfeast(
    problem::ContactMeanField1D,
    chart_spec,
    initial;
    config::TwoTimescaleConfig,
)
    config.maximum_subspace_dimension > config.subspace_dimension || throw(ArgumentError(
        "adaptive dimension requires maximum_subspace_dimension above subspace_dimension",
    ))
    solve_two_timescale_nlfeast(problem, chart_spec, initial; config=config)
end
