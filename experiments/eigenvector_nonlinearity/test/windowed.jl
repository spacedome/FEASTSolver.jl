using LinearAlgebra
using Random
using Test

isdefined(Main, :ProjectorMomentFEAST) || include(
    joinpath(@__DIR__, "..", "src", "ProjectorMomentFEAST.jl"),
)
using .ProjectorMomentFEAST

@testset "reduced density response derivative" begin
    problem = contact_mean_field_1d(points=32, half_length=6.0, coupling=5.0, occupied=3)
    initial = initial_orbitals(problem; seed=33)
    basis = ProjectorMomentFEAST.augmented_basis(initial, 8, 19)
    rho = density(problem, initial)
    state = ProjectorMomentFEAST.reduced_ritz_state(problem, basis, rho)
    rng = MersenneTwister(9)
    direction = randn(rng, length(rho))
    direction .-= sum(direction) / length(direction)
    action = ProjectorMomentFEAST.reduced_density_response(
        problem,
        state,
        direction,
    )
    step = 1e-6
    plus = ProjectorMomentFEAST.reduced_ritz_state(
        problem,
        basis,
        rho + step .* direction,
    ).output_density
    minus = ProjectorMomentFEAST.reduced_ritz_state(
        problem,
        basis,
        rho - step .* direction,
    ).output_density
    finite_difference = (plus - minus) ./ (2step)

    @test norm(action - finite_difference) / norm(finite_difference) <= 1e-7
end

@testset "windowed accumulated subspaces capture strong response" begin
    problem = contact_mean_field_1d(points=36, half_length=6.0, coupling=5.0, occupied=3)
    initial = initial_orbitals(problem; seed=33)
    reference = reference_scf(problem, density(problem, initial); mixing=0.2)
    chart = OccupiedChartPolicy(lower_bound=0.0, upper_bound=15.0, node_count=48)
    common = (
        moment_depth=3,
        probe_width=1,
        refresh_iterations=15,
        inner_schedule=:fixed,
        fixed_inner_iterations=3,
        inner_mixing=0.3,
        density_tolerance=1e-7,
        residual_tolerance=1e-7,
    )
    direct_memory = solve_windowed_nlfeast(
        problem,
        chart,
        initial;
        config=WindowedNLFEASTConfig(; common..., window_blocks=1),
    )
    accumulated = solve_windowed_nlfeast(
        problem,
        chart,
        initial;
        config=WindowedNLFEASTConfig(; common..., window_blocks=3),
    )
    growing = solve_windowed_nlfeast(
        problem,
        chart,
        initial;
        config=WindowedNLFEASTConfig(
            ;
            common...,
            window_blocks=0,
            fixed_inner_iterations=5,
        ),
    )
    pre_extraction = solve_raw_windowed_nlfeast(
        problem,
        chart,
        initial;
        config=WindowedNLFEASTConfig(; common..., window_blocks=3),
    )

    @test !direct_memory.converged
    @test accumulated.converged
    @test accumulated.variant === :post_extraction_window
    @test accumulated.closure_defect <= 1e-7
    @test accumulated.residual <= 1e-7
    @test subspace_gap(accumulated.orbitals, reference.orbitals) <= 1e-6
    @test maximum(record.basis_dimension for record in accumulated.history) == 9
    @test accumulated.rhs_count == accumulated.refresh_count * 48
    @test growing.converged
    @test growing.variant === :post_extraction_growing
    @test growing.refresh_count < accumulated.refresh_count
    @test subspace_gap(growing.orbitals, reference.orbitals) <= 1e-6
    @test pre_extraction.converged
    @test pre_extraction.variant === :pre_extraction_window
    @test subspace_gap(pre_extraction.orbitals, accumulated.orbitals) <= 1e-6
    @test pre_extraction.rhs_count == accumulated.rhs_count
end
