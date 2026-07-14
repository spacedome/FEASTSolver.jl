Base.@kwdef struct CombinedWindowConfig
    moment_depth::Int
    probe_width::Int
    window_blocks::Int = 4
    refresh_iterations::Int = 30
    inner_iterations::Int = 3
    inner_mixing::Float64 = 0.5
    inner_history_depth::Int = 8
    inner_regularization::Float64 = 1e-12
    maximum_step_ratio::Float64 = 4.0
    density_tolerance::Float64 = 1e-9
    residual_tolerance::Float64 = 1e-9
    subspace_ranktol::Float64 = 1e-10
end

function corrected_polynomial_raw_block(
    problem::QuadraticDensityNEP,
    rho,
    chart::CircularChart,
    orbitals,
    state,
    moment_depth,
    probe_width,
)
    state_count = size(orbitals, 2)
    tangent = moment_tangent(state_count, probe_width)
    probe = orbitals * tangent
    moments = [
        zeros(ComplexF64, size(orbitals, 1), probe_width)
        for _ in 1:moment_depth
    ]
    residual = state === nothing ? nothing :
        -hamiltonian(problem.base, rho) * orbitals +
        orbitals * state + problem.quadratic * orbitals * state^2
    state_identity = Matrix{ComplexF64}(I, state_count, state_count)
    factors = [lu(polynomial_matrix(problem, rho, z)) for z in chart.nodes]
    count_diagnostic = determinant_count(factors)
    count_diagnostic.count == state_count || throw(ContourCountError(
        state_count,
        count_diagnostic.count,
        count_diagnostic.winding,
        count_diagnostic.maximum_phase_step,
    ))
    for (z, weight, factorization) in zip(chart.nodes, chart.weights, factors)
        response = if state === nothing
            factorization \ probe
        else
            corrected = (orbitals - factorization \ residual) /
                (z .* state_identity .- state)
            corrected * tangent
        end
        coordinate = (z - chart.center) / chart.radius
        power = one(ComplexF64)
        for moment in moments
            moment .+= (weight * power) .* response
            power *= coordinate
        end
    end
    (
        block=reduce(hcat, moments),
        solve_count=length(chart.nodes),
        rhs_count=length(chart.nodes) * (state === nothing ? probe_width : state_count),
    )
end

function quadratic_reduced_state(problem::QuadraticDensityNEP, chart, basis, rho)
    q = size(basis, 2)
    H = adjoint(basis) * hamiltonian(problem.base, rho) * basis
    C = adjoint(basis) * problem.quadratic * basis
    A = zeros(ComplexF64, 2q, 2q)
    B = zeros(ComplexF64, 2q, 2q)
    identity_q = Matrix{ComplexF64}(I, q, q)
    A[1:q, (q + 1):(2q)] .= identity_q
    A[(q + 1):(2q), 1:q] .= H
    A[(q + 1):(2q), (q + 1):(2q)] .= -identity_q
    B[1:q, 1:q] .= identity_q
    B[(q + 1):(2q), (q + 1):(2q)] .= C
    decomposition = eigen(A, B)
    indices = findall(
        value -> isfinite(value) && abs(value - chart.center) < chart.radius,
        decomposition.values,
    )
    length(indices) == problem.base.occupied || error(
        "reduced quadratic extraction found $(length(indices)) target states, expected $(problem.base.occupied)",
    )
    values = ComplexF64.(decomposition.values[indices])
    raw_orbitals = decomposition.vectors[1:q, indices]
    factorization = qr(raw_orbitals)
    reduced_orbitals = Matrix(factorization.Q)[:, 1:problem.base.occupied]
    gauge = Matrix(factorization.R)[
        1:problem.base.occupied,
        1:problem.base.occupied,
    ]
    state = gauge * Diagonal(values) / gauge
    orbitals = basis * reduced_orbitals
    output_density = density(problem.base, orbitals)
    (
        orbitals=orbitals,
        state=state,
        density=output_density,
        residual=polynomial_invariant_residual(problem, rho, orbitals, state),
        values=values,
    )
end

function solve_combined_windowed_nlfeast(
    problem::QuadraticDensityNEP,
    chart::CircularChart,
    initial;
    config::CombinedWindowConfig,
)
    occupied = problem.base.occupied
    config.moment_depth * config.probe_width >= occupied || throw(ArgumentError(
        "combined moment capacity is below the occupied count",
    ))
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
    orbitals = orthonormalize(initial, occupied)
    state = nothing
    rho = density(problem.base, orbitals)
    blocks = Matrix{ComplexF64}[]
    basis = orbitals
    history = NamedTuple[]
    converged = false

    for refresh in 1:config.refresh_iterations
        filtered = corrected_polynomial_raw_block(
            problem,
            rho,
            chart,
            orbitals,
            state,
            config.moment_depth,
            config.probe_width,
        )
        push!(blocks, filtered.block)
        while config.window_blocks > 0 && length(blocks) > config.window_blocks
            popfirst!(blocks)
        end
        window = window_basis(blocks, config.subspace_ranktol)
        window.rank >= occupied || error(
            "combined accumulated basis rank is below the occupied count",
        )
        basis = window.basis
        densities = Vector{Float64}[]
        residuals = Vector{Float64}[]
        reduced = quadratic_reduced_state(problem, chart, basis, rho)
        density_defect = Inf
        performed = 0
        for inner in 1:config.inner_iterations
            reduced = quadratic_reduced_state(problem, chart, basis, rho)
            fixed_point_residual = reduced.density - rho
            density_defect = norm(fixed_point_residual) / max(norm(rho), eps(Float64))
            performed = inner
            if density_defect <= config.density_tolerance &&
                    reduced.residual <= config.residual_tolerance
                converged = true
                break
            end
            push!(densities, copy(rho))
            push!(residuals, fixed_point_residual)
            while length(densities) > config.inner_history_depth + 1
                popfirst!(densities)
                popfirst!(residuals)
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
            normalize_density!(problem.base, rho)
        end
        reduced = quadratic_reduced_state(problem, chart, basis, rho)
        orbitals = reduced.orbitals
        state = reduced.state
        density_defect = norm(reduced.density - rho) / max(norm(rho), eps(Float64))
        converged = density_defect <= config.density_tolerance &&
            reduced.residual <= config.residual_tolerance
        push!(history, (
            refresh=refresh,
            inner_iterations=performed,
            density_defect=density_defect,
            residual=reduced.residual,
            basis_dimension=size(basis, 2),
            solve_count=filtered.solve_count,
            rhs_count=filtered.rhs_count,
        ))
        converged && break
    end

    final_state = quadratic_reduced_state(problem, chart, basis, rho)
    (
        variant=config.window_blocks == 0 ? :combined_growing : :combined_window,
        inner_method=:anderson,
        orbitals=final_state.orbitals,
        state=final_state.state,
        density=rho,
        basis=basis,
        history=history,
        converged=converged,
        density_defect=norm(final_state.density - rho) / max(norm(rho), eps(Float64)),
        residual=final_state.residual,
        refresh_count=length(history),
        inner_iterations=sum(record.inner_iterations for record in history),
        solve_count=sum(record.solve_count for record in history),
        rhs_count=sum(record.rhs_count for record in history),
    )
end
