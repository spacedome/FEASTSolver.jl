Base.@kwdef struct ReducedNLFEASTConfig
    moment_depth::Int
    probe_width::Int
    subspace_dimension::Int
    outer_iterations::Int = 20
    inner_iterations::Int = 60
    inner_history_depth::Int = 8
    inner_mixing::Float64 = 0.5
    density_tolerance::Float64 = 1e-10
    residual_tolerance::Float64 = 1e-10
    ranktol::Float64 = 1e-12
    seed::Int = 4817
end

function augmented_basis(initial, dimension, seed)
    size(initial, 2) <= dimension || throw(DimensionMismatch(
        "initial orbital count exceeds the requested reduced dimension",
    ))
    size(initial, 2) == dimension && return orthonormalize(initial, dimension)
    rng = MersenneTwister(seed)
    extra = randn(rng, ComplexF64, size(initial, 1), dimension - size(initial, 2))
    orthonormalize(hcat(initial, extra), dimension)
end

function solve_reduced_density(
    problem,
    basis,
    initial_density,
    config::ReducedNLFEASTConfig,
)
    rho = copy(initial_density)
    densities = Vector{Float64}[]
    residuals = Vector{Float64}[]
    orbitals = basis[:, 1:problem.occupied]
    defect = Inf
    for iteration in 1:config.inner_iterations
        reduced_hamiltonian = Hermitian(adjoint(basis) * hamiltonian(problem, rho) * basis)
        decomposition = eigen(reduced_hamiltonian)
        orbitals = basis * decomposition.vectors[:, 1:problem.occupied]
        output_density = density(problem, orbitals)
        fixed_point_residual = output_density - rho
        defect = norm(fixed_point_residual) / max(norm(rho), eps(Float64))
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
            1e-12,
        )
        rho += correction
        minimum(rho) < 0 && (rho .= max.(rho, 0); rho .*= problem.occupied / (problem.spacing * sum(rho)))
        defect <= config.density_tolerance && return (
            orbitals=orbitals,
            density=rho,
            iterations=iteration,
            defect=defect,
        )
    end
    (orbitals=orbitals, density=rho, iterations=config.inner_iterations, defect=defect)
end

function solve_reduced_nlfeast(
    problem::ContactMeanField1D,
    chart::CircularChart,
    initial;
    config::ReducedNLFEASTConfig,
)
    config.subspace_dimension >= problem.occupied || throw(ArgumentError(
        "the reduced subspace must contain the occupied space",
    ))
    config.moment_depth * config.probe_width >= config.subspace_dimension || throw(ArgumentError(
        "moment capacity is below the reduced subspace dimension",
    ))
    basis = augmented_basis(initial, config.subspace_dimension, config.seed)
    orbitals = orthonormalize(initial, problem.occupied)
    rho = density(problem, orbitals)
    tangent = moment_tangent(config.subspace_dimension, config.probe_width)
    history = NamedTuple[]
    converged = false

    for iteration in 1:config.outer_iterations
        H = hamiltonian(problem, rho)
        step = moment_projector_step(
            H,
            chart,
            basis * tangent,
            config.moment_depth,
            config.subspace_dimension;
            ranktol=config.ranktol,
        )
        basis = orthonormalize(step.orbitals, config.subspace_dimension)
        reduced = solve_reduced_density(problem, basis, rho, config)
        orbitals = reduced.orbitals
        rho = reduced.density
        residual = invariant_residual(problem, rho, orbitals)
        push!(history, (
            iteration=iteration,
            density_defect=reduced.defect,
            residual=residual,
            energy=contact_energy(problem, orbitals),
            inner_iterations=reduced.iterations,
            singular_ratio=step.singular_values[config.subspace_dimension] /
                step.singular_values[1],
            solve_count=step.solve_count,
            solve_application_count=step.solve_count,
            rhs_count=step.rhs_count,
        ))
        if reduced.defect <= config.density_tolerance && residual <= config.residual_tolerance
            converged = true
            break
        end
    end

    (
        orbitals=orbitals,
        density=rho,
        basis=basis,
        history=history,
        converged=converged,
        residual=invariant_residual(problem, rho, orbitals),
        energy=contact_energy(problem, orbitals),
        solve_count=sum(record.solve_count for record in history),
        solve_application_count=sum(record.solve_application_count for record in history),
        rhs_count=sum(record.rhs_count for record in history),
    )
end
