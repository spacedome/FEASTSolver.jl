Base.@kwdef struct OrbitalSCFConfig
    moment_depth::Int
    probe_width::Int
    iterations::Int = 100
    mixing::Float64 = 0.5
    projector_tolerance::Float64 = 1e-10
    residual_tolerance::Float64 = 1e-10
    ranktol::Float64 = 1e-12
end

function align_orbitals(reference, candidate)
    factorization = svd(adjoint(candidate) * reference)
    candidate * (factorization.U * adjoint(factorization.V))
end

function solve_orbital_scf(
    problem::ContactMeanField1D,
    chart_spec,
    initial;
    config::OrbitalSCFConfig,
)
    config.moment_depth * config.probe_width >= problem.occupied || throw(ArgumentError(
        "moment capacity is below the occupied count",
    ))
    0 < config.mixing <= 1 || throw(ArgumentError("mixing must lie in (0,1]"))
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
        aligned = align_orbitals(orbitals, step.orbitals)
        next_orbitals = orthonormalize(
            (1 - config.mixing) .* orbitals .+ config.mixing .* aligned,
            problem.occupied,
        )
        next_density = density(problem, next_orbitals)
        projector_defect = subspace_gap(step.orbitals, orbitals) / sqrt(problem.occupied)
        residual = invariant_residual(problem, next_density, next_orbitals)
        push!(history, (
            iteration=iteration,
            projector_defect=projector_defect,
            residual=residual,
            energy=contact_energy(problem, next_orbitals),
            singular_ratio=step.singular_values[problem.occupied] / step.singular_values[1],
            solve_count=step.solve_count,
            solve_application_count=step.solve_count,
            rhs_count=step.rhs_count,
            chart_factorization_count=chart_resolution.factorization_count,
            chart_center=chart.center,
            chart_radius=chart.radius,
        ))
        orbitals = next_orbitals
        rho = next_density
        if projector_defect <= config.projector_tolerance && residual <= config.residual_tolerance
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
