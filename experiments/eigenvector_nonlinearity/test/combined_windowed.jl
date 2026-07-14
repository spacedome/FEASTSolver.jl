using LinearAlgebra
using Test

isdefined(Main, :ProjectorMomentFEAST) || include(
    joinpath(@__DIR__, "..", "src", "ProjectorMomentFEAST.jl"),
)
using .ProjectorMomentFEAST

@testset "combined accumulated zero-quadratic reduction" begin
    base = contact_mean_field_1d(points=32, half_length=6.0, coupling=1.0, occupied=3)
    problem = quadratic_density_nep(base; strength=0.0)
    initial = initial_orbitals(base; seed=33)
    reference = reference_scf(base, density(base, initial); mixing=0.5)
    values = eigvals(hamiltonian(base, reference.density))
    chart = CircularChart(
        (values[1] + values[3]) / 2,
        (values[3] - values[1]) / 2 + 0.3 * (values[4] - values[3]),
        64,
    )
    combined = solve_combined_windowed_nlfeast(
        problem,
        chart,
        initial;
        config=CombinedWindowConfig(
            moment_depth=3,
            probe_width=1,
            window_blocks=3,
            refresh_iterations=20,
            inner_iterations=3,
            inner_mixing=0.5,
            density_tolerance=1e-7,
            residual_tolerance=1e-7,
        ),
    )
    linear = solve_raw_windowed_nlfeast(
        base,
        chart,
        initial;
        config=WindowedNLFEASTConfig(
            moment_depth=3,
            probe_width=1,
            window_blocks=3,
            refresh_iterations=20,
            inner_schedule=:fixed,
            fixed_inner_iterations=3,
            inner_mixing=0.5,
            density_tolerance=1e-7,
            residual_tolerance=1e-7,
        ),
    )

    @test combined.converged
    @test linear.converged
    @test combined.refresh_count == linear.refresh_count
    @test subspace_gap(combined.orbitals, linear.orbitals) <= 1e-12
    @test norm(combined.density - linear.density) / norm(linear.density) <= 1e-12
end

@testset "combined accumulated nonlinear control" begin
    base = contact_mean_field_1d(points=32, half_length=6.0, coupling=1.0, occupied=3)
    problem = quadratic_density_nep(base; strength=0.02)
    initial = initial_orbitals(base; seed=33)
    chart = CircularChart(1.8, 1.5, 64)
    direct = solve_combined_scf(
        problem,
        chart,
        initial;
        config=CombinedSCFConfig(
            moment_depth=3,
            probe_width=1,
            iterations=25,
            outer_method=:anderson,
            history_depth=10,
            mixing=0.6,
            density_tolerance=1e-7,
            residual_tolerance=1e-7,
        ),
    )
    windowed = solve_combined_windowed_nlfeast(
        problem,
        chart,
        initial;
        config=CombinedWindowConfig(
            moment_depth=3,
            probe_width=1,
            window_blocks=4,
            refresh_iterations=15,
            inner_iterations=3,
            inner_mixing=0.6,
            density_tolerance=1e-7,
            residual_tolerance=1e-7,
        ),
    )

    @test direct.converged
    @test windowed.converged
    @test windowed.variant === :combined_window
    @test windowed.residual <= 1e-7
    @test subspace_gap(windowed.orbitals, direct.orbitals) <= 1e-6
    @test windowed.solve_count < direct.solve_count
    @test windowed.rhs_count < direct.rhs_count
end
