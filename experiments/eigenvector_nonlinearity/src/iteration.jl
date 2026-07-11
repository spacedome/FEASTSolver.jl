Base.@kwdef struct MomentSCFConfig
    moment_depth::Int
    probe_width::Int
    iterations::Int = 100
    mixing::Float64 = 0.5
    density_tolerance::Float64 = 1e-10
    residual_tolerance::Float64 = 1e-10
    ranktol::Float64 = 1e-12
    preserve_chart::Bool = false
    chart_safety::Float64 = 0.8
end

Base.@kwdef struct ResponseNewtonConfig
    moment_depth::Int
    probe_width::Int
    iterations::Int = 30
    warmup_iterations::Int = 2
    warmup_method::Symbol = :mixing
    warmup_mixing::Float64 = 0.2
    warmup_history_depth::Int = 6
    damping::Float64 = 1.0
    density_tolerance::Float64 = 1e-10
    residual_tolerance::Float64 = 1e-10
    ranktol::Float64 = 1e-12
    krylov_tolerance::Float64 = 1e-8
    maximum_krylov_tolerance::Float64 = 0.1
    krylov_iterations::Int = 30
    krylov_method::Symbol = :auto
    adaptive_krylov::Bool = true
    preserve_chart::Bool = false
    chart_safety::Float64 = 0.8
end

function limit_density_correction(
    problem::ContactMeanField1D,
    chart::CircularChart,
    values,
    rho,
    correction;
    preserve_chart,
    chart_safety,
)
    0 < chart_safety < 1 || throw(ArgumentError("chart_safety must lie in (0,1)"))
    scale = 1.0
    if preserve_chart && problem.coupling > 0
        margin = minimum(chart.radius .- abs.(values .- chart.center))
        margin > 0 || error("the current occupied state touches or leaves the contour")
        hamiltonian_step = problem.coupling * norm(correction, Inf)
        hamiltonian_step > 0 && (scale = min(scale, chart_safety * margin / hamiltonian_step))
    end
    candidate = rho + scale .* correction
    if minimum(candidate) < 0
        positivity_scale = 0.95 * minimum(rho ./ max.(rho .- candidate, eps(Float64)))
        scale *= clamp(positivity_scale, 0.0, 1.0)
    end
    scale .* correction, scale
end

function solve_moment_scf(
    problem::ContactMeanField1D,
    chart_spec,
    initial;
    config::MomentSCFConfig,
)
    0 < config.mixing <= 1 || throw(ArgumentError("mixing must lie in (0,1]"))
    config.moment_depth * config.probe_width >= problem.occupied || throw(ArgumentError(
        "moment capacity is below the occupied count",
    ))
    orbitals = orthonormalize(initial, problem.occupied)
    rho = density(problem, orbitals)
    tangent = moment_tangent(problem.occupied, config.probe_width)
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
        density_defect = norm(output_density - rho) / max(norm(rho), eps(Float64))
        correction, step_scale = limit_density_correction(
            problem,
            chart,
            step.values,
            rho,
            config.mixing .* (output_density - rho);
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
            state_values=step.state_values,
            singular_ratio=step.singular_values[problem.occupied] / step.singular_values[1],
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

function solve_response_newton(
    problem::ContactMeanField1D,
    chart_spec,
    initial;
    config::ResponseNewtonConfig,
)
    config.moment_depth * config.probe_width >= problem.occupied || throw(ArgumentError(
        "moment capacity is below the occupied count",
    ))
    0 < config.warmup_mixing <= 1 || throw(ArgumentError(
        "warmup_mixing must lie in (0,1]",
    ))
    config.warmup_method in (:mixing, :anderson) || throw(ArgumentError(
        "warmup_method must be :mixing or :anderson",
    ))
    config.krylov_method in (:auto, :cg, :gmres) || throw(ArgumentError(
        "krylov_method must be :auto, :cg, or :gmres",
    ))
    0 < config.damping <= 1 || throw(ArgumentError("damping must lie in (0,1]"))
    orbitals = orthonormalize(initial, problem.occupied)
    rho = density(problem, orbitals)
    tangent = moment_tangent(problem.occupied, config.probe_width)
    warmup_densities = Vector{Float64}[]
    warmup_residuals = Vector{Float64}[]
    history = NamedTuple[]
    converged = false

    for iteration in 1:config.iterations
        H = hamiltonian(problem, rho)
        chart_resolution = resolve_chart(chart_spec, H, problem.occupied)
        chart = chart_resolution.chart
        factors = contour_factorizations(H, chart)
        step = moment_projector_step(
            H,
            chart,
            orbitals * tangent,
            config.moment_depth,
            problem.occupied;
            ranktol=config.ranktol,
            factorizations=factors,
        )
        output_density = density(problem, step.orbitals)
        fixed_point_residual = output_density - rho
        density_defect = norm(fixed_point_residual) / max(norm(rho), eps(Float64))

        correction, krylov_steps, krylov_residual, mode, krylov_method = if iteration <= config.warmup_iterations
            push!(warmup_densities, copy(rho))
            push!(warmup_residuals, fixed_point_residual)
            while length(warmup_densities) > config.warmup_history_depth + 1
                popfirst!(warmup_densities)
                popfirst!(warmup_residuals)
            end
            warmup_correction = config.warmup_method === :anderson ?
                anderson_correction(
                    warmup_densities,
                    warmup_residuals,
                    config.warmup_mixing,
                    1e-12,
                ) :
                config.warmup_mixing .* fixed_point_residual
            (warmup_correction, 0, 0.0, config.warmup_method, :none)
        else
            response = direction -> contour_density_response(
                problem,
                step.orbitals,
                step.values,
                direction,
                chart;
                factorizations=factors,
            )
            krylov_tolerance = config.adaptive_krylov ?
                min(
                    config.maximum_krylov_tolerance,
                    max(config.krylov_tolerance, sqrt(density_defect)),
                ) :
                config.krylov_tolerance
            action = direction -> direction - response(direction)
            selected_method = config.krylov_method === :auto ? :cg : config.krylov_method
            krylov = selected_method === :cg ?
                cg_action(
                    action,
                    fixed_point_residual;
                    tolerance=krylov_tolerance,
                    iterations=config.krylov_iterations,
                ) :
                gmres_action(
                    action,
                    fixed_point_residual;
                    tolerance=krylov_tolerance,
                    iterations=config.krylov_iterations,
                )
            (
                config.damping .* krylov.solution,
                krylov.iterations,
                krylov.residual,
                :response_newton,
                selected_method,
            )
        end

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
            mode=mode,
            density_defect=density_defect,
            residual=residual,
            energy=contact_energy(problem, step.orbitals),
            values=step.values,
            singular_ratio=step.singular_values[problem.occupied] / step.singular_values[1],
            krylov_steps=krylov_steps,
            krylov_residual=krylov_residual,
            krylov_method=krylov_method,
            step_scale=step_scale,
            solve_count=step.solve_count,
            solve_application_count=step.solve_count * (1 + krylov_steps),
            rhs_count=step.rhs_count + krylov_steps * length(chart.nodes) * problem.occupied,
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
