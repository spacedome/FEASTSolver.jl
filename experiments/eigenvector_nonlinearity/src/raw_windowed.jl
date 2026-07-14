function raw_contour_moment_block(
    H,
    chart::CircularChart,
    probe,
    moment_depth,
    target_count,
)
    moment_depth > 0 || throw(ArgumentError("moment_depth must be positive"))
    factors = contour_factorizations(H, chart)
    count_diagnostic = determinant_count(factors)
    count_diagnostic.count == target_count || throw(ContourCountError(
        target_count,
        count_diagnostic.count,
        count_diagnostic.winding,
        count_diagnostic.maximum_phase_step,
    ))
    moments = [zeros(ComplexF64, size(H, 1), size(probe, 2)) for _ in 1:moment_depth]
    for (z, weight, factorization) in zip(chart.nodes, chart.weights, factors)
        response = factorization \ probe
        coordinate = (z - chart.center) / chart.radius
        power = one(ComplexF64)
        for moment in moments
            moment .+= (weight * power) .* response
            power *= coordinate
        end
    end
    (
        block=reduce(hcat, moments),
        count_diagnostic=count_diagnostic,
        solve_count=length(chart.nodes),
        rhs_count=length(chart.nodes) * size(probe, 2),
    )
end

function raw_window_probe(problem, basis, rho, occupied, width, fallback, seed)
    if width <= problem.occupied
        tangent = moment_tangent(problem.occupied, width)
        return occupied * tangent
    end

    H = hamiltonian(problem, rho)
    decomposition = eigen(Hermitian(adjoint(basis) * H * basis))
    retained = min(width, size(basis, 2))
    primary = basis * decomposition.vectors[:, 1:retained]
    candidates = hcat(primary, fallback)
    candidate_svd = svd(candidates)
    threshold = 1e-12 * candidate_svd.S[1]
    candidate_rank = count(singular -> singular >= threshold, candidate_svd.S)
    independent = candidate_svd.U[:, 1:min(width, candidate_rank)]
    augmented_basis(independent, width, seed)
end

function solve_raw_windowed_nlfeast(
    problem::ContactMeanField1D,
    chart_spec,
    initial;
    config::WindowedNLFEASTConfig,
)
    validate_windowed_config(problem, config)
    orbitals = orthonormalize(initial, problem.occupied)
    rho = density(problem, orbitals)
    trial_width = max(problem.occupied, config.probe_width)
    trial = augmented_basis(orbitals, trial_width, config.seed)
    probe = raw_window_probe(
        problem,
        trial,
        rho,
        orbitals,
        config.probe_width,
        trial,
        config.seed,
    )
    blocks = Matrix{ComplexF64}[]
    history = NamedTuple[]
    inner_history = NamedTuple[]
    basis = orbitals
    converged = false

    for refresh in 1:config.refresh_iterations
        H = hamiltonian(problem, rho)
        chart_resolution = resolve_chart(chart_spec, H, problem.occupied)
        filtered = raw_contour_moment_block(
            H,
            chart_resolution.chart,
            probe,
            config.moment_depth,
            problem.occupied,
        )
        push!(blocks, filtered.block)
        while config.window_blocks > 0 && length(blocks) > config.window_blocks
            popfirst!(blocks)
        end
        window = window_basis(blocks, config.subspace_ranktol)
        size(window.basis, 2) >= problem.occupied || error(
            "raw contour window has rank below the occupied count",
        )
        basis = window.basis
        inner = solve_windowed_inner(problem, basis, rho, config)
        orbitals = inner.orbitals
        rho = inner.density
        append!(inner_history, [merge(record, (refresh=refresh,)) for record in inner.history])
        probe = raw_window_probe(
            problem,
            basis,
            rho,
            orbitals,
            config.probe_width,
            probe,
            config.seed + refresh,
        )
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
            reduced_response_actions=sum(record.krylov_steps for record in inner.history),
            solve_count=filtered.solve_count,
            rhs_count=filtered.rhs_count,
            chart_factorization_count=chart_resolution.factorization_count,
        ))
        converged && break
    end

    final_state = reduced_ritz_state(problem, basis, rho)
    (
        variant=config.window_blocks == 0 ?
            :pre_extraction_growing : :pre_extraction_window,
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
