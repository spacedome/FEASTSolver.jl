function projector_occupied_enrichment(
    problem::NonlocalProjector1D,
    chart::CircularChart,
    basis,
    projector;
    moment_depth,
    probe_width,
    ranktol,
    enrichment_ranktol,
    output_dimension=size(basis, 2),
)
    state = projector_ritz_state(problem, basis, projector)
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
    q <= output_dimension <= min(length(problem.grid), q + problem.occupied) || throw(
        ArgumentError("output_dimension is outside the enriched projector space"),
    )
    if q == problem.occupied && output_dimension == q
        return (basis=step.orbitals, correction_rank=problem.occupied, step=step)
    end

    complement = step.orbitals - basis * (adjoint(basis) * step.orbitals)
    decomposition = svd(complement)
    scale = decomposition.S[1]
    correction_rank = scale == 0 ? 0 : count(
        singular -> singular >= enrichment_ranktol * scale,
        decomposition.S,
    )
    correction_rank > 0 || error(
        "nonlocal occupied correction is numerically contained in the reduced space",
    )
    raw_correction = decomposition.U[:, 1:correction_rank]
    projected = raw_correction - basis * (adjoint(basis) * raw_correction)
    correction = orthonormalize(projected, correction_rank)
    expanded = orthonormalize(hcat(basis, correction), q + correction_rank)
    reduced = eigen(Hermitian(adjoint(expanded) * state.hamiltonian * expanded))
    realized_dimension = min(output_dimension, size(expanded, 2))
    (
        basis=orthonormalize(
            expanded * reduced.vectors[:, 1:realized_dimension],
            realized_dimension,
        ),
        correction_rank=correction_rank,
        step=step,
    )
end

function solve_projector_two_timescale(
    problem::NonlocalProjector1D,
    chart_spec,
    initial;
    config::TwoTimescaleConfig,
)
    validate_two_timescale_config(problem, config)
    basis = augmented_basis(initial, config.subspace_dimension, config.seed)
    orbitals = orthonormalize(initial, problem.occupied)
    projector = projector_state(orbitals)
    history = NamedTuple[]
    converged = false
    dimension_leakages = Float64[]
    total_inner_iterations = 0

    for phase in 1:(config.refresh_iterations + 1)
        state = projector_ritz_state(problem, basis, projector)
        closure_defect = Inf
        stop_reason = :inner_limit
        inner_iterations = 0
        projectors = Matrix{ComplexF64}[]
        residuals = Matrix{ComplexF64}[]
        for inner in 1:config.maximum_inner_iterations
            state = projector_ritz_state(problem, basis, projector)
            fixed_point_residual = state.output_projector - projector
            closure_defect = norm(fixed_point_residual) /
                max(norm(projector), eps(Float64))
            inner_iterations = inner
            total_inner_iterations += 1
            if closure_defect <= config.density_tolerance &&
                    state.leakage <= config.residual_tolerance
                converged = true
                stop_reason = :converged
                break
            end
            if inner >= config.minimum_inner_iterations &&
                    closure_defect <= config.forcing_ratio * state.leakage
                stop_reason = :leakage_forcing
                break
            end
            push!(projectors, copy(projector))
            push!(residuals, fixed_point_residual)
            while length(projectors) > config.inner_history_depth + 1
                popfirst!(projectors)
                popfirst!(residuals)
            end
            correction = projector_anderson_correction(
                projectors,
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
            projector = normalize_projector_trace(problem, projector + correction)
        end
        if stop_reason === :inner_limit
            state = projector_ritz_state(problem, basis, projector)
            closure_defect = norm(state.output_projector - projector) /
                max(norm(projector), eps(Float64))
        end
        push!(dimension_leakages, state.leakage)

        if converged || phase > config.refresh_iterations
            push!(history, (
                phase=phase,
                stop_reason=stop_reason,
                inner_iterations=inner_iterations,
                closure_defect=closure_defect,
                leakage=state.leakage,
                basis_dimension=size(basis, 2),
                next_basis_dimension=size(basis, 2),
                solve_count=0,
                rhs_count=0,
                chart_factorization_count=0,
            ))
            break
        end

        chart_resolution = resolve_chart(chart_spec, state.hamiltonian, problem.occupied)
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
        repair = projector_occupied_enrichment(
            problem,
            chart_resolution.chart,
            basis,
            projector;
            moment_depth=config.moment_depth,
            probe_width=config.probe_width,
            ranktol=config.ranktol,
            enrichment_ranktol=config.enrichment_ranktol,
            output_dimension=requested_dimension,
        )
        basis = repair.basis
        size(basis, 2) > current_dimension && empty!(dimension_leakages)
        push!(history, (
            phase=phase,
            stop_reason=stop_reason,
            inner_iterations=inner_iterations,
            closure_defect=closure_defect,
            leakage=state.leakage,
            basis_dimension=current_dimension,
            next_basis_dimension=size(basis, 2),
            solve_count=repair.step.solve_count,
            rhs_count=repair.step.rhs_count,
            chart_factorization_count=chart_resolution.factorization_count,
        ))
    end

    final_state = projector_ritz_state(problem, basis, projector)
    (
        variant=config.maximum_subspace_dimension > config.subspace_dimension ?
            :nonlocal_adaptive_dimension : :nonlocal_fixed_dimension,
        inner_method=:projector_anderson,
        orbitals=final_state.orbitals,
        projector=projector,
        basis=basis,
        history=history,
        converged=converged,
        closure_defect=norm(final_state.output_projector - projector) /
            max(norm(projector), eps(Float64)),
        residual=final_state.leakage,
        inner_iterations=total_inner_iterations,
        refresh_count=sum(record.solve_count > 0 for record in history),
        solve_count=sum(record.solve_count for record in history),
        rhs_count=sum(record.rhs_count for record in history),
        chart_factorization_count=sum(record.chart_factorization_count for record in history),
    )
end
