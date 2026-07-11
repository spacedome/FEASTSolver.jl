Base.@kwdef struct LevelShiftConfig
    moment_depth::Int
    probe_width::Int
    shift::Float64
    iterations::Int = 100
    projector_tolerance::Float64 = 1e-10
    residual_tolerance::Float64 = 1e-10
    ranktol::Float64 = 1e-12
end

function solve_level_shifted_scf(
    problem::ContactMeanField1D,
    shifted_chart::CircularChart,
    initial;
    config::LevelShiftConfig,
)
    config.shift > 0 || throw(ArgumentError("level shift must be positive"))
    config.moment_depth * config.probe_width >= problem.occupied || throw(ArgumentError(
        "moment capacity is below the occupied count",
    ))
    orbitals = orthonormalize(initial, problem.occupied)
    rho = density(problem, orbitals)
    tangent = moment_tangent(problem.occupied, config.probe_width)
    history = NamedTuple[]
    converged = false

    for iteration in 1:config.iterations
        shifted_hamiltonian = Hermitian(
            Matrix(hamiltonian(problem, rho)) -
            config.shift .* (orbitals * adjoint(orbitals)),
        )
        step = moment_projector_step(
            shifted_hamiltonian,
            shifted_chart,
            orbitals * tangent,
            config.moment_depth,
            problem.occupied;
            ranktol=config.ranktol,
        )
        next_orbitals = step.orbitals
        next_density = density(problem, next_orbitals)
        projector_defect = subspace_gap(next_orbitals, orbitals) / sqrt(problem.occupied)
        density_defect = norm(next_density - rho) / max(norm(rho), eps(Float64))
        residual = invariant_residual(problem, next_density, next_orbitals)
        push!(history, (
            iteration=iteration,
            projector_defect=projector_defect,
            density_defect=density_defect,
            residual=residual,
            energy=contact_energy(problem, next_orbitals),
            values=step.values,
            singular_ratio=step.singular_values[problem.occupied] / step.singular_values[1],
            solve_count=step.solve_count,
            solve_application_count=step.solve_count,
            rhs_count=step.rhs_count,
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
    )
end

