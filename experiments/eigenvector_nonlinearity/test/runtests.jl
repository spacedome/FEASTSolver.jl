using LinearAlgebra
using Random
using SparseArrays
using Test

include(joinpath(@__DIR__, "..", "src", "ProjectorMomentFEAST.jl"))
using .ProjectorMomentFEAST

@testset "eigenvector-nonlinear projector moment FEAST" begin
    problem = contact_mean_field_1d(points=72, half_length=7.0, coupling=5.0, occupied=4)
    initial = initial_orbitals(problem; seed=903)
    initial_density = density(problem, initial)
    reference = reference_scf(problem, initial_density; mixing=0.2)
    @test reference.converged
    @test reference.residual <= 1e-11

    spectrum = eigen(hamiltonian(problem, reference.density)).values
    occupied_values = spectrum[1:problem.occupied]
    gap = spectrum[problem.occupied + 1] - spectrum[problem.occupied]
    chart = CircularChart(
        (first(occupied_values) + last(occupied_values)) / 2,
        (last(occupied_values) - first(occupied_values)) / 2 + 0.30gap,
        96,
    )

    H = hamiltonian(problem, initial_density)
    rng = MersenneTwister(11)
    X = randn(rng, ComplexF64, length(problem.grid), problem.occupied)
    S = randn(rng, ComplexF64, problem.occupied, problem.occupied)
    z = chart.nodes[3]
    corrected = rii_response(H, X, S, z)
    filtered = (z .* I - H) \ X
    @test norm(corrected - filtered) / norm(filtered) <= 1e-11

    tangent = ProjectorMomentFEAST.moment_tangent(problem.occupied, 2)
    narrow = moment_projector_step(H, chart, initial * tangent, 2, problem.occupied)
    wide = moment_projector_step(H, chart, initial, 1, problem.occupied)
    frozen_decomposition = eigen(H)
    @test subspace_gap(narrow.orbitals, wide.orbitals) <= 1e-7
    @test subspace_gap(
        narrow.orbitals,
        frozen_decomposition.vectors[:, 1:problem.occupied],
    ) <= 1e-7
    @test narrow.probe_width == 2
    @test narrow.count_diagnostic.count == problem.occupied
    @test narrow.count_diagnostic.integer_error <= 1e-10
    @test narrow.rhs_count == 2length(chart.nodes)
    @test wide.rhs_count == 4length(chart.nodes)
    expanded_chart = CircularChart(
        (frozen_decomposition.values[1] + frozen_decomposition.values[problem.occupied + 1]) / 2,
        (frozen_decomposition.values[problem.occupied + 1] - frozen_decomposition.values[1]) / 2 +
        0.2 * (
            frozen_decomposition.values[problem.occupied + 2] -
            frozen_decomposition.values[problem.occupied + 1]
        ),
        96,
    )
    @test_throws ContourCountError moment_projector_step(
        H,
        expanded_chart,
        initial,
        1,
        problem.occupied,
    )

    config = ResponseNewtonConfig(
        moment_depth=2,
        probe_width=2,
        iterations=30,
        warmup_iterations=2,
        warmup_method=:anderson,
        warmup_mixing=0.6,
        warmup_history_depth=10,
        krylov_method=:cg,
        density_tolerance=1e-10,
        residual_tolerance=1e-10,
        krylov_tolerance=1e-9,
    )
    result = solve_response_newton(problem, chart, initial; config=config)
    @test result.converged
    @test result.residual <= 1e-10
    @test norm(adjoint(result.orbitals) * result.orbitals - I) <= 1e-11
    @test norm(result.density - reference.density) / norm(reference.density) <= 1e-8
    @test subspace_gap(result.orbitals, reference.orbitals) <= 1e-7
    @test abs(result.energy - reference.energy) <= 1e-8
    @test all(record.singular_ratio >= 1e-10 for record in result.history)
    @test all(
        record.krylov_method in (:none, :cg)
        for record in result.history
    )

    exact_jacobian = projector_density_jacobian(problem, reference.density)
    jacobian_values = eigvals(Symmetric(exact_jacobian))
    @test maximum(abs, jacobian_values) > 1
    @test maximum(jacobian_values) <= 1e-12
    @test minimum(eigvals(Symmetric(I - exact_jacobian))) >= 1 - 1e-12
    derivative_chart = chart
    reference_hamiltonian = hamiltonian(problem, reference.density)
    factors = ProjectorMomentFEAST.contour_factorizations(reference_hamiltonian, derivative_chart)
    directions = Matrix{Float64}(I, length(problem.grid), length(problem.grid))
    response_jacobian = hcat([
        contour_density_response(
            problem,
            reference.orbitals,
            reference.values,
            view(directions, :, column),
            derivative_chart;
            factorizations=factors,
        )
        for column in axes(directions, 2)
    ]...)
    @test norm(response_jacobian - exact_jacobian) / norm(exact_jacobian) <= 1e-6

    anderson = solve_anderson_scf(
        problem,
        chart,
        initial;
        config=AndersonConfig(
            moment_depth=2,
            probe_width=2,
            iterations=40,
            history_depth=10,
            mixing=0.6,
            density_tolerance=1e-10,
            residual_tolerance=1e-10,
        ),
    )
    @test anderson.converged
    @test anderson.residual <= 1e-10
    @test norm(anderson.density - reference.density) / norm(reference.density) <= 1e-8
    @test anderson.rhs_count < result.rhs_count
    @test result.solve_count < anderson.solve_count
end

@testset "level-shifted projector FEAST" begin
    problem = contact_mean_field_1d(points=48, half_length=6.0, coupling=1.0, occupied=3)
    initial = initial_orbitals(problem; seed=33)
    reference = reference_scf(problem, density(problem, initial); mixing=0.5)
    @test reference.converged
    shifted_chart = CircularChart(2.0, 1.3, 128)
    result = solve_level_shifted_scf(
        problem,
        shifted_chart,
        initial;
        config=LevelShiftConfig(
            moment_depth=3,
            probe_width=1,
            shift=0.25,
            iterations=30,
        ),
    )
    @test result.converged
    @test result.residual <= 1e-10
    @test subspace_gap(result.orbitals, reference.orbitals) <= 1e-7
end

@testset "canonical reduced NLFEAST branch" begin
    problem = contact_mean_field_1d(points=48, half_length=6.0, coupling=1.0, occupied=3)
    initial = initial_orbitals(problem; seed=33)
    reference = reference_scf(problem, density(problem, initial); mixing=0.5)
    spectrum = eigvals(hamiltonian(problem, reference.density))
    reduced_dimension = 5
    chart = CircularChart(
        (spectrum[1] + spectrum[reduced_dimension]) / 2,
        (spectrum[reduced_dimension] - spectrum[1]) / 2 +
        0.25 * (spectrum[reduced_dimension + 1] - spectrum[reduced_dimension]),
        128,
    )
    result = solve_reduced_nlfeast(
        problem,
        chart,
        initial;
        config=ReducedNLFEASTConfig(
            moment_depth=3,
            probe_width=2,
            subspace_dimension=reduced_dimension,
            outer_iterations=20,
            inner_iterations=80,
            density_tolerance=1e-9,
            residual_tolerance=1e-9,
        ),
    )
    @test result.converged
    @test result.residual <= 1e-9
    @test subspace_gap(result.orbitals, reference.orbitals) <= 1e-7
end

@testset "non-Hermitian dual projector closure" begin
    problem = similarity_mean_field_1d(
        points=48,
        half_length=6.0,
        coupling=1.0,
        occupied=3,
        nonnormality=0.5,
    )
    initial_right, initial_left = initial_biorthogonal_orbitals(problem; seed=33)
    initial_density = oblique_density(problem, initial_right, initial_left)
    reference = reference_scf(problem.base, initial_density; mixing=0.5)
    initial_spectrum = eigvals(hamiltonian(problem.base, initial_density))
    final_spectrum = eigvals(hamiltonian(problem.base, reference.density))
    lower = min(initial_spectrum[1], final_spectrum[1])
    upper = max(initial_spectrum[problem.base.occupied], final_spectrum[problem.base.occupied])
    outside = min(
        initial_spectrum[problem.base.occupied + 1],
        final_spectrum[problem.base.occupied + 1],
    )
    chart = CircularChart(
        (lower + upper) / 2,
        (upper - lower) / 2 + 0.25 * (outside - upper),
        128,
    )
    result = solve_dual_scf(
        problem,
        chart,
        initial_right,
        initial_left;
        config=DualSCFConfig(iterations=60, mixing=0.4),
    )
    target_right = problem.similarity .* reference.orbitals
    right_only = ProjectorMomentFEAST.orthonormalize(target_right, problem.base.occupied)
    right_only_density = density(problem.base, right_only)

    @test result.converged
    @test result.residual <= 1e-10
    @test norm(adjoint(result.left) * result.right - I) <= 1e-10
    @test norm(result.density - reference.density) / norm(reference.density) <= 1e-8
    @test subspace_gap(result.right, target_right) <= 1e-7
    @test norm(right_only_density - reference.density) / norm(reference.density) >= 0.2
    @test result.solve_count == length(chart.nodes) * length(result.history)
end

@testset "Gross-Pitaevskii ground state" begin
    problem = contact_mean_field_1d(points=64, half_length=7.0, coupling=10.0, occupied=1)
    initial = initial_orbitals(problem; seed=77)
    initial_density = density(problem, initial)
    reference = reference_scf(problem, initial_density; mixing=0.15)
    @test reference.converged

    chart_policy = OccupiedChartPolicy(
        lower_bound=0.0,
        upper_bound=12.0,
        node_count=64,
    )
    result = solve_response_newton(
        problem,
        chart_policy,
        initial;
        config=ResponseNewtonConfig(
            moment_depth=1,
            probe_width=1,
            iterations=30,
            warmup_iterations=5,
            warmup_mixing=0.15,
            density_tolerance=1e-10,
            residual_tolerance=1e-10,
            krylov_tolerance=1e-9,
        ),
    )

    @test result.converged
    @test result.residual <= 1e-10
    @test norm(result.density - reference.density) / norm(reference.density) <= 1e-8
    @test subspace_gap(result.orbitals, reference.orbitals) <= 1e-7
    @test abs(problem.spacing * sum(result.density) - 1) <= 1e-10
    @test result.chart_factorization_count > 0
end

@testset "sparse shifted solves remain sparse" begin
    problem = contact_mean_field_1d(
        points=128,
        half_length=9.0,
        coupling=0.0,
        occupied=4,
        sparse=true,
    )
    initial = initial_orbitals(problem; seed=19)
    H = hamiltonian(problem, density(problem, initial))
    @test parent(H) isa SparseMatrixCSC
    chart = CircularChart(2.0, 1.9, 64)
    tangent = ProjectorMomentFEAST.moment_tangent(problem.occupied, 2)
    step = moment_projector_step(H, chart, initial * tangent, 2, problem.occupied)
    @test step.count_diagnostic.count == problem.occupied
    @test invariant_residual(problem, density(problem, step.orbitals), step.orbitals) <= 1e-5
end

@testset "simultaneous spectral and density nonlinearity" begin
    base = contact_mean_field_1d(points=48, half_length=6.0, coupling=1.0, occupied=3)
    initial = initial_orbitals(base; seed=33)
    rho = density(base, initial)
    frozen_values = eigvals(hamiltonian(base, rho))
    linear_chart = CircularChart(
        (frozen_values[1] + frozen_values[base.occupied]) / 2,
        (frozen_values[base.occupied] - frozen_values[1]) / 2 +
        0.3 * (frozen_values[base.occupied + 1] - frozen_values[base.occupied]),
        128,
    )
    linear_problem = quadratic_density_nep(base; strength=0.0)
    polynomial_step = corrected_polynomial_moment_step(
        linear_problem,
        rho,
        linear_chart,
        initial,
        nothing,
        3,
        1,
    )
    tangent = ProjectorMomentFEAST.moment_tangent(base.occupied, 1)
    linear_step = moment_projector_step(
        hamiltonian(base, rho),
        linear_chart,
        initial * tangent,
        3,
        base.occupied,
    )
    @test subspace_gap(polynomial_step.orbitals, linear_step.orbitals) <= 1e-10
    @test polynomial_invariant_residual(
        linear_problem,
        rho,
        polynomial_step.orbitals,
        polynomial_step.state,
    ) <= 1e-10

    nonlinear_problem = quadratic_density_nep(base; strength=0.02)
    result = solve_combined_scf(
        nonlinear_problem,
        CircularChart(1.8, 1.5, 128),
        initial;
        config=CombinedSCFConfig(
            moment_depth=3,
            probe_width=1,
            iterations=30,
            outer_method=:anderson,
            history_depth=10,
            mixing=0.6,
            density_tolerance=1e-9,
            residual_tolerance=1e-9,
        ),
    )
    @test result.converged
    @test result.residual <= 1e-9
    @test abs(base.spacing * sum(result.density) - base.occupied) <= 1e-10
end
