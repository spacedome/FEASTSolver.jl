using LinearAlgebra
using Test

isdefined(Main, :ProjectorMomentFEAST) || include(
    joinpath(@__DIR__, "..", "src", "ProjectorMomentFEAST.jl"),
)
using .ProjectorMomentFEAST

@testset "nonlocal projector state cannot collapse to density" begin
    problem = nonlocal_projector_1d(points=16, coupling=2.0, occupied=1)
    uniform = fill(inv(sqrt(16)), 16, 1)
    phases = reshape(ComplexF64[cis(2pi * index / 16) for index in 1:16], 16, 1)
    twisted = uniform .* phases
    first_projector = projector_state(uniform)
    second_projector = projector_state(twisted)

    @test diag(first_projector) ≈ diag(second_projector)
    @test norm(first_projector - second_projector) >= 1
    @test norm(
        hamiltonian(problem, first_projector) -
        hamiltonian(problem, second_projector),
    ) >= 1e-2
end

@testset "nonlocal projector two-timescale schedules" begin
    problem = nonlocal_projector_1d(
        points=28,
        half_length=7.0,
        coupling=4.2,
        occupied=3,
    )
    initial = initial_orbitals(problem; seed=33)
    initial_projector = projector_state(initial)
    reference = reference_projector_scf(
        problem,
        initial_projector;
        mixing=0.1,
        tolerance=1e-9,
    )
    unstable = reference_projector_scf(
        problem,
        initial_projector;
        mixing=0.2,
        tolerance=1e-8,
        iterations=300,
    )
    chart = OccupiedChartPolicy(lower_bound=0.0, upper_bound=15.0, node_count=64)
    adaptive = solve_projector_two_timescale(
        problem,
        chart,
        initial;
        config=TwoTimescaleConfig(
            moment_depth=3,
            probe_width=1,
            subspace_dimension=3,
            maximum_subspace_dimension=9,
            growth_increment=3,
            stagnation_ratio=0.4,
            refresh_iterations=25,
            maximum_inner_iterations=30,
            forcing_ratio=1.0,
            inner_mixing=0.3,
            density_tolerance=1e-6,
            residual_tolerance=1e-6,
        ),
    )
    windowed = solve_projector_windowed_nlfeast(
        problem,
        chart,
        initial;
        config=ProjectorWindowConfig(
            moment_depth=3,
            probe_width=1,
            window_blocks=4,
            refresh_iterations=20,
            inner_iterations=5,
            inner_method=:anderson,
            inner_mixing=0.6,
            projector_tolerance=1e-6,
            residual_tolerance=1e-6,
        ),
    )
    projector_values = eigvals(Hermitian(adaptive.projector))

    @test reference.converged
    @test !unstable.converged
    @test adaptive.converged
    @test adaptive.variant === :nonlocal_adaptive_dimension
    @test size(adaptive.basis, 2) > problem.occupied
    @test subspace_gap(adaptive.orbitals, reference.orbitals) <= 1e-5
    @test windowed.converged
    @test windowed.variant === :nonlocal_pre_extraction_window
    @test subspace_gap(windowed.orbitals, reference.orbitals) <= 1e-5
    @test windowed.refresh_count < adaptive.refresh_count
    @test abs(real(tr(adaptive.projector)) - problem.occupied) <= 1e-10
    @test minimum(projector_values) >= -1e-6
    @test maximum(projector_values) <= 1 + 1e-6
end
