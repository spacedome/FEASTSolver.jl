Base.@kwdef struct AndersonConfig
    moment_depth::Int
    probe_width::Int
    iterations::Int = 60
    history_depth::Int = 6
    mixing::Float64 = 0.5
    regularization::Float64 = 1e-12
    maximum_step_ratio::Float64 = 4.0
    density_tolerance::Float64 = 1e-10
    residual_tolerance::Float64 = 1e-10
    ranktol::Float64 = 1e-12
    preserve_chart::Bool = false
    chart_safety::Float64 = 0.8
end

function anderson_correction(densities, residuals, mixing, regularization)
    residual = residuals[end]
    length(residuals) == 1 && return mixing .* residual
    density_differences = reduce(hcat, [
        densities[index + 1] - densities[index] for index in 1:(length(densities) - 1)
    ])
    residual_differences = reduce(hcat, [
        residuals[index + 1] - residuals[index] for index in 1:(length(residuals) - 1)
    ])
    gram = adjoint(residual_differences) * residual_differences
    scale = max(opnorm(gram), 1.0)
    coefficients = (gram + regularization * scale * I) \ (
        adjoint(residual_differences) * residual
    )
    mixing .* residual - (density_differences + mixing .* residual_differences) * coefficients
end

function solve_anderson_scf(
    problem::ContactMeanField1D,
    chart_spec,
    initial;
    config::AndersonConfig,
)
    config.moment_depth * config.probe_width >= problem.occupied || throw(ArgumentError(
        "moment capacity is below the occupied count",
    ))
    config.history_depth > 0 || throw(ArgumentError("history_depth must be positive"))
    0 < config.mixing <= 1 || throw(ArgumentError("mixing must lie in (0,1]"))
    orbitals = orthonormalize(initial, problem.occupied)
    rho = density(problem, orbitals)
    tangent = moment_tangent(problem.occupied, config.probe_width)
    densities = Vector{Float64}[]
    residuals = Vector{Float64}[]
    history = NamedTuple[]
    converged = false

    for iteration in 1:config.iterations
        H = hamiltonian(problem, rho)
        chart_resolution = resolve_chart(chart_spec, H, problem.occupied)
        chart = chart_resolution.chart
        step = moment_projector_step(
            H,
            chart,
            orbitals * tangent,
            config.moment_depth,
            problem.occupied;
            ranktol=config.ranktol,
        )
        output_density = density(problem, step.orbitals)
        fixed_point_residual = output_density - rho
        density_defect = norm(fixed_point_residual) / max(norm(rho), eps(Float64))
        push!(densities, copy(rho))
        push!(residuals, fixed_point_residual)
        while length(densities) > config.history_depth + 1
            popfirst!(densities)
            popfirst!(residuals)
        end
        correction = anderson_correction(
            densities,
            residuals,
            config.mixing,
            config.regularization,
        )
        maximum_step = config.maximum_step_ratio * max(norm(fixed_point_residual), eps(Float64))
        norm(correction) > maximum_step && (correction .*= maximum_step / norm(correction))
        correction, step_scale = limit_density_correction(
            problem,
            chart,
            step.values,
            rho,
            correction;
            preserve_chart=config.preserve_chart,
            chart_safety=config.chart_safety,
        )
        next_density = rho + correction
        residual = invariant_residual(problem, next_density, step.orbitals)
        push!(history, (
            iteration=iteration,
            density_defect=density_defect,
            residual=residual,
            energy=contact_energy(problem, step.orbitals),
            values=step.values,
            singular_ratio=step.singular_values[problem.occupied] / step.singular_values[1],
            history_size=length(residuals),
            step_scale=step_scale,
            solve_count=step.solve_count,
            solve_application_count=step.solve_count,
            rhs_count=step.rhs_count,
            chart_factorization_count=chart_resolution.factorization_count,
            chart_center=chart.center,
            chart_radius=chart.radius,
        ))
        orbitals = step.orbitals
        rho = next_density
        if density_defect <= config.density_tolerance && residual <= config.residual_tolerance
            converged = true
            break
        end
    end

    (
        orbitals=orbitals,
        density=rho,
        history=history,
        converged=converged,
        residual=invariant_residual(problem, rho, orbitals),
        energy=contact_energy(problem, orbitals),
        solve_count=sum(record.solve_count for record in history),
        solve_application_count=sum(record.solve_application_count for record in history),
        rhs_count=sum(record.rhs_count for record in history),
        chart_factorization_count=sum(record.chart_factorization_count for record in history),
    )
end
