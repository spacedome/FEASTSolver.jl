using LinearAlgebra
using Test

isdefined(Main, :ProjectorMomentFEAST) || include(
    joinpath(@__DIR__, "..", "src", "ProjectorMomentFEAST.jl"),
)
using .ProjectorMomentFEAST

@testset "occupied enrichment reduces to direct FEAST at q=p" begin
    problem = contact_mean_field_1d(points=40, half_length=6.0, coupling=1.0, occupied=3)
    initial = initial_orbitals(problem; seed=33)
    rho = density(problem, initial)
    H = hamiltonian(problem, rho)
    values = eigvals(H)
    chart = CircularChart(
        (values[1] + values[3]) / 2,
        (values[3] - values[1]) / 2 + 0.3 * (values[4] - values[3]),
        64,
    )
    state = ProjectorMomentFEAST.reduced_ritz_state(problem, initial, rho)
    tangent = ProjectorMomentFEAST.moment_tangent(problem.occupied, 1)
    direct = moment_projector_step(H, chart, state.orbitals * tangent, 3, problem.occupied)
    repair = occupied_moment_enrichment(
        problem,
        chart,
        initial,
        rho;
        moment_depth=3,
        probe_width=1,
    )

    @test subspace_gap(repair.basis, direct.orbitals) <= 1e-12
    @test repair.step.rhs_count == length(chart.nodes)
end

@testset "adaptive response dimension escapes q=p" begin
    problem = contact_mean_field_1d(points=36, half_length=6.0, coupling=5.0, occupied=3)
    initial = initial_orbitals(problem; seed=33)
    reference = reference_scf(problem, density(problem, initial); mixing=0.2)
    chart = OccupiedChartPolicy(lower_bound=0.0, upper_bound=15.0, node_count=48)
    result = solve_adaptive_dimension_nlfeast(
        problem,
        chart,
        initial;
        config=TwoTimescaleConfig(
            moment_depth=3,
            probe_width=1,
            subspace_dimension=3,
            maximum_subspace_dimension=9,
            growth_increment=3,
            stagnation_ratio=0.6,
            stagnation_iterations=2,
            refresh_iterations=30,
            maximum_inner_iterations=40,
            forcing_ratio=1.0,
            inner_mixing=0.3,
            density_tolerance=1e-7,
            residual_tolerance=1e-7,
        ),
    )
    growth = filter(
        record -> record.next_basis_dimension > record.basis_dimension,
        result.refresh_history,
    )

    @test result.converged
    @test result.variant === :adaptive_dimension_enrichment
    @test size(result.basis, 2) == 6
    @test length(growth) == 1
    @test growth[1].basis_dimension == 3
    @test growth[1].next_basis_dimension == 6
    @test subspace_gap(result.orbitals, reference.orbitals) <= 1e-6
end

@testset "leakage-forced two-timescale iteration" begin
    problem = contact_mean_field_1d(points=40, half_length=6.0, coupling=1.0, occupied=3)
    initial = initial_orbitals(problem; seed=33)
    reference = reference_scf(problem, density(problem, initial); mixing=0.5)
    values = eigvals(hamiltonian(problem, reference.density))
    chart = CircularChart(
        (values[1] + values[3]) / 2,
        (values[3] - values[1]) / 2 + 0.3 * (values[4] - values[3]),
        64,
    )
    config = TwoTimescaleConfig(
        moment_depth=3,
        probe_width=1,
        subspace_dimension=5,
        refresh_iterations=15,
        maximum_inner_iterations=30,
        forcing_ratio=0.25,
        inner_mixing=0.5,
        density_tolerance=1e-8,
        residual_tolerance=1e-8,
    )
    result = solve_two_timescale_nlfeast(problem, chart, initial; config=config)
    forced = filter(record -> record.stop_reason === :leakage_forcing, result.refresh_history)

    @test result.converged
    @test result.variant === :fixed_dimension_enrichment
    @test result.closure_defect <= config.density_tolerance
    @test result.residual <= config.residual_tolerance
    @test subspace_gap(result.orbitals, reference.orbitals) <= 1e-7
    @test !isempty(forced)
    @test all(
        record.closure_defect <= config.forcing_ratio * record.leakage * (1 + 1e-12)
        for record in forced
    )
    @test result.rhs_count == result.refresh_count * length(chart.nodes)
    @test result.solve_count == result.refresh_count * length(chart.nodes)
end

@testset "adaptive dimension handles rank-one strong coupling" begin
    problem = contact_mean_field_1d(points=36, half_length=7.0, coupling=10.0, occupied=1)
    initial = initial_orbitals(problem; seed=77)
    reference = reference_scf(problem, density(problem, initial); mixing=0.15)
    chart = OccupiedChartPolicy(lower_bound=0.0, upper_bound=20.0, node_count=40)
    result = solve_adaptive_dimension_nlfeast(
        problem,
        chart,
        initial;
        config=TwoTimescaleConfig(
            moment_depth=1,
            probe_width=1,
            subspace_dimension=1,
            maximum_subspace_dimension=6,
            growth_increment=1,
            stagnation_ratio=0.4,
            refresh_iterations=30,
            maximum_inner_iterations=40,
            forcing_ratio=1.0,
            inner_mixing=0.15,
            density_tolerance=1e-6,
            residual_tolerance=1e-6,
        ),
    )

    @test result.converged
    @test size(result.basis, 2) > problem.occupied
    @test result.closure_defect <= 1e-6
    @test result.residual <= 1e-6
    @test subspace_gap(result.orbitals, reference.orbitals) <= 1e-5
    @test abs(problem.spacing * sum(result.density) - 1) <= 1e-10
end
