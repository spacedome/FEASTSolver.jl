@testset "dual cache and response" begin
    problem = PMF.similarity_mean_field_1d(
        points=32,
        half_length=6.0,
        coupling=1.0,
        occupied=3,
        nonnormality=0.5,
    )
    initial_right, initial_left = PMF.initial_biorthogonal_orbitals(problem; seed=33)
    initial_density = PMF.oblique_density(problem, initial_right, initial_left)
    reference = PMF.reference_scf(problem.base, initial_density; mixing=0.5)
    H = PMF.hamiltonian(problem, reference.density)
    decomposition = eigen(H)
    indices = sortperm(real.(decomposition.values))[1:problem.base.occupied]
    right = decomposition.vectors[:, indices]
    left = adjoint(inv(decomposition.vectors))[:, indices]
    values = decomposition.values[indices]
    spectrum = sort(real.(decomposition.values))
    chart = PMF.CircularChart(
        (spectrum[1] + spectrum[3]) / 2,
        (spectrum[3] - spectrum[1]) / 2 + 0.3 * (spectrum[4] - spectrum[3]),
        128,
    )
    cache = build_dual_cache(problem, chart, reference.density)
    tangent = PMF.moment_tangent(problem.base.occupied, 1)
    sampled = dual_positive_moments!(cache, right * tangent, left * tangent, 3)
    standalone = PMF.dual_raw_moment_block(
        H,
        chart,
        right * tangent,
        left * tangent,
        3,
    )
    @test sampled.right == standalone.right
    @test sampled.left == standalone.left

    direction = randn(MersenneTwister(47), length(reference.density))
    direction .-= sum(direction) / length(direction)
    response = dual_density_response!(cache, right, left, values, direction)
    function exact_density(rho)
        state = eigen(PMF.hamiltonian(problem, rho))
        occupied = sortperm(real.(state.values))[1:problem.base.occupied]
        modal_right = state.vectors[:, occupied]
        modal_left = adjoint(inv(state.vectors))[:, occupied]
        PMF.oblique_density(problem, modal_right, modal_left)
    end
    step = 1e-6
    finite_difference = (
        exact_density(reference.density + step .* direction) -
        exact_density(reference.density - step .* direction)
    ) ./ (2step)
    @test norm(response - finite_difference) / norm(finite_difference) <= 5e-7
    @test dual_cache_stats(cache).response_rhs_count ==
        2length(chart.nodes) * problem.base.occupied
end

@testset "nonlocal projector cache response" begin
    problem = PMF.nonlocal_projector_1d(
        points=32,
        half_length=6.0,
        coupling=3.0,
        occupied=3,
    )
    initial = PMF.initial_orbitals(problem; seed=33)
    initial_projector = PMF.projector_state(initial)
    reference = PMF.reference_projector_scf(
        problem,
        initial_projector;
        mixing=0.1,
        tolerance=1e-10,
    )
    decomposition = eigen(PMF.hamiltonian(problem, reference.projector))
    orbitals = ComplexF64.(decomposition.vectors[:, 1:problem.occupied])
    values = Float64.(decomposition.values[1:problem.occupied])
    gap = decomposition.values[problem.occupied + 1] - values[end]
    chart = PMF.CircularChart(
        (values[1] + values[end]) / 2,
        (values[end] - values[1]) / 2 + 0.3gap,
        128,
    )
    cache = build_projector_cache(problem, chart, reference.projector)
    rng = MersenneTwister(91)
    raw_direction = randn(rng, ComplexF64, size(reference.projector))
    direction = Matrix(Hermitian(raw_direction + adjoint(raw_direction)))
    direction ./= norm(direction)
    response = projector_response!(cache, orbitals, values, direction)

    function exact_projector(projector)
        state = eigen(PMF.hamiltonian(problem, projector))
        PMF.projector_state(state.vectors[:, 1:problem.occupied])
    end
    step = 1e-6
    finite_difference = (
        exact_projector(reference.projector + step .* direction) -
        exact_projector(reference.projector - step .* direction)
    ) ./ (2step)
    @test norm(response - finite_difference) / norm(finite_difference) <= 5e-7
    @test count(>(1e-9 * opnorm(response)), svdvals(response)) <= 2problem.occupied
    @test projector_cache_stats(cache).response_rhs_count ==
        length(chart.nodes) * problem.occupied
end
