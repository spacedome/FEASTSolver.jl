struct QuadraticDensityNEP{P<:ContactMeanField1D,M<:AbstractMatrix{Float64}}
    base::P
    quadratic::M
end

function quadratic_density_nep(base::ContactMeanField1D; strength=0.02)
    strength >= 0 || throw(ArgumentError("quadratic strength must be nonnegative"))
    profile = strength .* (1 .+ 0.05 .* base.grid .^ 2)
    QuadraticDensityNEP(base, Diagonal(profile))
end

function polynomial_matrix(problem::QuadraticDensityNEP, rho, z)
    z^2 .* problem.quadratic + z * I - hamiltonian(problem.base, rho)
end

function polynomial_invariant_residual(problem::QuadraticDensityNEP, rho, orbitals, state)
    residual = -hamiltonian(problem.base, rho) * orbitals +
        orbitals * state + problem.quadratic * orbitals * state^2
    scale = norm(hamiltonian(problem.base, rho) * orbitals) +
        norm(orbitals * state) + norm(problem.quadratic * orbitals * state^2)
    norm(residual) / max(scale, eps(Float64))
end

function corrected_polynomial_moment_step(
    problem::QuadraticDensityNEP,
    rho,
    chart::CircularChart,
    orbitals,
    state,
    moment_depth,
    probe_width;
    ranktol=1e-12,
)
    state_count = size(orbitals, 2)
    moment_depth * probe_width >= state_count || throw(ArgumentError(
        "moment depth and probe width cannot represent the target count",
    ))
    tangent = moment_tangent(state_count, probe_width)
    probe = orbitals * tangent
    moments = [
        zeros(ComplexF64, size(orbitals, 1), probe_width)
        for _ in 1:(2moment_depth)
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

    H0, H1 = block_hankel(moments, moment_depth, probe)
    factorization = svd(H0)
    factorization.S[state_count] >= ranktol * factorization.S[1] || error(
        "polynomial moment pencil has numerical rank below the target count",
    )
    U = factorization.U[:, 1:state_count]
    V = factorization.V[:, 1:state_count]
    inverse_singulars = Diagonal(1.0 ./ factorization.S[1:state_count])
    coordinate_state = adjoint(U) * H1 * V * inverse_singulars
    realized_state = chart.center .* I + chart.radius .* coordinate_state
    moment_row = reduce(hcat, moments[1:moment_depth])
    raw_map = moment_row * V * inverse_singulars
    map_qr = qr(raw_map)
    basis = Matrix(map_qr.Q)[:, 1:state_count]
    gauge = Matrix(map_qr.R)[1:state_count, 1:state_count]
    basis_state = gauge * realized_state / gauge
    values = eigvals(basis_state)
    count(value -> abs(value - chart.center) < chart.radius, values) == state_count || error(
        "realized polynomial state leaves the target contour",
    )
    (
        orbitals=basis,
        state=basis_state,
        values=values,
        singular_values=Float64.(factorization.S),
        count_diagnostic=count_diagnostic,
        solve_count=length(chart.nodes),
        rhs_count=length(chart.nodes) * (state === nothing ? probe_width : state_count),
    )
end

Base.@kwdef struct CombinedSCFConfig
    moment_depth::Int
    probe_width::Int
    iterations::Int = 60
    mixing::Float64 = 0.3
    outer_method::Symbol = :mixing
    history_depth::Int = 8
    regularization::Float64 = 1e-12
    maximum_step_ratio::Float64 = 4.0
    density_tolerance::Float64 = 1e-10
    residual_tolerance::Float64 = 1e-10
    ranktol::Float64 = 1e-12
end

function solve_combined_scf(
    problem::QuadraticDensityNEP,
    chart::CircularChart,
    initial;
    config::CombinedSCFConfig,
)
    0 < config.mixing <= 1 || throw(ArgumentError("mixing must lie in (0,1]"))
    config.outer_method in (:mixing, :anderson) || throw(ArgumentError(
        "outer_method must be :mixing or :anderson",
    ))
    orbitals = orthonormalize(initial, problem.base.occupied)
    state = nothing
    rho = density(problem.base, orbitals)
    history = NamedTuple[]
    densities = Vector{Float64}[]
    residuals = Vector{Float64}[]
    converged = false

    for iteration in 1:config.iterations
        step = corrected_polynomial_moment_step(
            problem,
            rho,
            chart,
            orbitals,
            state,
            config.moment_depth,
            config.probe_width;
            ranktol=config.ranktol,
        )
        output_density = density(problem.base, step.orbitals)
        fixed_point_residual = output_density - rho
        density_defect = norm(fixed_point_residual) / max(norm(rho), eps(Float64))
        push!(densities, copy(rho))
        push!(residuals, fixed_point_residual)
        while length(densities) > config.history_depth + 1
            popfirst!(densities)
            popfirst!(residuals)
        end
        correction = config.outer_method === :anderson ?
            anderson_correction(
                densities,
                residuals,
                config.mixing,
                config.regularization,
            ) : config.mixing .* fixed_point_residual
        maximum_step = config.maximum_step_ratio * max(norm(fixed_point_residual), eps(Float64))
        norm(correction) > maximum_step && (correction .*= maximum_step / norm(correction))
        next_density = rho + correction
        if minimum(next_density) < 0
            scale = min(1.0, 0.95 * minimum(rho ./ max.(rho .- next_density, eps(Float64))))
            next_density = rho + max(scale, 0.0) .* correction
        end
        residual = polynomial_invariant_residual(
            problem,
            next_density,
            step.orbitals,
            step.state,
        )
        push!(history, (
            iteration=iteration,
            density_defect=density_defect,
            residual=residual,
            values=step.values,
            singular_ratio=step.singular_values[problem.base.occupied] /
                step.singular_values[1],
            outer_method=config.outer_method,
            solve_count=step.solve_count,
            rhs_count=step.rhs_count,
        ))
        orbitals = step.orbitals
        state = step.state
        rho = next_density
        if density_defect <= config.density_tolerance && residual <= config.residual_tolerance
            converged = true
            break
        end
    end

    (
        orbitals=orbitals,
        state=state,
        density=rho,
        history=history,
        converged=converged,
        residual=polynomial_invariant_residual(problem, rho, orbitals, state),
        solve_count=sum(record.solve_count for record in history),
        rhs_count=sum(record.rhs_count for record in history),
    )
end
