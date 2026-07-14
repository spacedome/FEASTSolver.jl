using LinearAlgebra
using Test

isdefined(Main, :ProjectorMomentFEAST) || include(
    joinpath(@__DIR__, "..", "src", "ProjectorMomentFEAST.jl"),
)
using .ProjectorMomentFEAST

function dual_test_context(; points=40, nonnormality=0.5, nodes=64)
    problem = similarity_mean_field_1d(
        points=points,
        half_length=6.0,
        coupling=1.0,
        occupied=3,
        nonnormality=nonnormality,
    )
    right, left = initial_biorthogonal_orbitals(problem; seed=33)
    rho = oblique_density(problem, right, left)
    reference = reference_scf(problem.base, rho; mixing=0.5)
    initial_values = eigvals(hamiltonian(problem.base, rho))
    final_values = eigvals(hamiltonian(problem.base, reference.density))
    lower = min(initial_values[1], final_values[1])
    upper = max(initial_values[3], final_values[3])
    exterior = min(initial_values[4], final_values[4])
    chart = CircularChart(
        (lower + upper) / 2,
        (upper - lower) / 2 + 0.25 * (exterior - upper),
        nodes,
    )
    problem, right, left, rho, reference, chart
end

@testset "dual pre-extraction depth-one reduction" begin
    problem, right, left, rho, _, chart = dual_test_context(points=32)
    H = hamiltonian(problem, rho)
    direct = dual_projector_step(H, chart, right, left)
    filtered = ProjectorMomentFEAST.dual_raw_moment_block(
        H,
        chart,
        right,
        left,
        1,
    )
    window = ProjectorMomentFEAST.dual_window_basis(
        [filtered.right],
        [filtered.left],
        1e-12,
        1e-12,
    )
    reduced = ProjectorMomentFEAST.dual_reduced_state(
        problem,
        chart,
        window.right,
        window.left,
        rho,
    )

    @test subspace_gap(reduced.right, direct.right) <= 1e-10
    @test subspace_gap(reduced.left, direct.left) <= 1e-10
    @test norm(adjoint(reduced.left) * reduced.right - I) <= 1e-10
    @test filtered.solve_count == length(chart.nodes)
end

@testset "dual accumulated moment windows" begin
    problem, right, left, _, reference, chart = dual_test_context()
    result = solve_dual_windowed_nlfeast(
        problem,
        chart,
        right,
        left;
        config=DualWindowConfig(
            moment_depth=3,
            probe_width=1,
            window_blocks=3,
            refresh_iterations=20,
            inner_iterations=3,
            inner_mixing=0.6,
            density_tolerance=1e-7,
            residual_tolerance=1e-7,
        ),
    )
    target_right = problem.similarity .* reference.orbitals

    @test result.converged
    @test result.variant === :dual_window
    @test result.residual <= 1e-7
    @test result.density_defect <= 1e-7
    @test subspace_gap(result.right, target_right) <= 1e-6
    @test norm(adjoint(result.left) * result.right - I) <= 1e-10
    @test result.solve_count == result.refresh_count * length(chart.nodes)
    @test result.rhs_count == 2result.refresh_count * length(chart.nodes)
end
