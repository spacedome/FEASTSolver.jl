using Printf

include(joinpath(@__DIR__, "src", "ProjectorMomentFEAST.jl"))
using .ProjectorMomentFEAST

problem = contact_mean_field_1d(points=48, half_length=6.0, coupling=5.0, occupied=3)
initial = initial_orbitals(problem; seed=33)
reference = reference_scf(problem, density(problem, initial); mixing=0.2)
occupied_chart = OccupiedChartPolicy(lower_bound=0.0, upper_bound=15.0, node_count=64)
virtual_chart = OccupiedChartPolicy(lower_bound=0.0, upper_bound=20.0, node_count=64)

window_config(; blocks, inner) = WindowedNLFEASTConfig(
    moment_depth=3,
    probe_width=1,
    window_blocks=blocks,
    refresh_iterations=35,
    inner_schedule=:fixed,
    fixed_inner_iterations=inner,
    inner_mixing=0.3,
    density_tolerance=1e-8,
    residual_tolerance=1e-8,
)

variants = [
    (
        id="EV-DIRECT",
        run=() -> solve_anderson_scf(
            problem,
            occupied_chart,
            initial;
            config=AndersonConfig(
                moment_depth=3,
                probe_width=1,
                iterations=50,
                history_depth=10,
                mixing=0.6,
                density_tolerance=1e-8,
                residual_tolerance=1e-8,
            ),
        ),
    ),
    (
        id="EV-RESPONSE",
        run=() -> solve_response_newton(
            problem,
            occupied_chart,
            initial;
            config=ResponseNewtonConfig(
                moment_depth=3,
                probe_width=1,
                iterations=30,
                warmup_iterations=2,
                warmup_method=:anderson,
                warmup_mixing=0.6,
                krylov_method=:cg,
                krylov_tolerance=1e-7,
                density_tolerance=1e-8,
                residual_tolerance=1e-8,
            ),
        ),
    ),
    (
        id="EV-FULLQ q=8",
        run=() -> solve_reduced_nlfeast(
            problem,
            virtual_chart,
            initial;
            config=ReducedNLFEASTConfig(
                moment_depth=4,
                probe_width=2,
                subspace_dimension=8,
                outer_iterations=40,
                inner_iterations=80,
                density_tolerance=1e-8,
                residual_tolerance=1e-8,
            ),
        ),
    ),
    (
        id="EV-TWO q=8",
        run=() -> solve_two_timescale_nlfeast(
            problem,
            occupied_chart,
            initial;
            config=TwoTimescaleConfig(
                moment_depth=3,
                probe_width=1,
                subspace_dimension=8,
                refresh_iterations=35,
                maximum_inner_iterations=50,
                forcing_ratio=1.0,
                inner_mixing=0.3,
                density_tolerance=1e-8,
                residual_tolerance=1e-8,
            ),
        ),
    ),
    (
        id="EV-ADAPT q=3",
        run=() -> solve_adaptive_dimension_nlfeast(
            problem,
            occupied_chart,
            initial;
            config=TwoTimescaleConfig(
                moment_depth=3,
                probe_width=1,
                subspace_dimension=3,
                maximum_subspace_dimension=9,
                growth_increment=3,
                stagnation_ratio=0.6,
                refresh_iterations=35,
                maximum_inner_iterations=50,
                forcing_ratio=1.0,
                inner_mixing=0.3,
                density_tolerance=1e-8,
                residual_tolerance=1e-8,
            ),
        ),
    ),
    (
        id="EV-WINDOW-POST h=3",
        run=() -> solve_windowed_nlfeast(
            problem,
            occupied_chart,
            initial;
            config=window_config(blocks=3, inner=3),
        ),
    ),
    (
        id="EV-WINDOW-PRE h=3",
        run=() -> solve_raw_windowed_nlfeast(
            problem,
            occupied_chart,
            initial;
            config=window_config(blocks=3, inner=3),
        ),
    ),
    (
        id="EV-GROWING-PRE",
        run=() -> solve_raw_windowed_nlfeast(
            problem,
            occupied_chart,
            initial;
            config=window_config(blocks=0, inner=5),
        ),
    ),
]

rows = map(variants) do variant
    result = variant.run()
    dimension = hasproperty(result, :basis) ? size(result.basis, 2) : problem.occupied
    outer = hasproperty(result, :refresh_count) ? result.refresh_count : length(result.history)
    inner = hasproperty(result, :inner_iterations) ? result.inner_iterations : 0
    (
        id=variant.id,
        result=result,
        dimension=dimension,
        outer=outer,
        inner=inner,
        gap=subspace_gap(result.orbitals, reference.orbitals),
    )
end

println("strong-control categorical sweep")
println("variant                 ok  outer  inner    q  factors    RHS      residual          gap")
for row in rows
    @printf(
        "%-23s %-5s %5d %6d %4d %8d %6d   %10.3e   %10.3e\n",
        row.id,
        string(row.result.converged),
        row.outer,
        row.inner,
        row.dimension,
        row.result.solve_count,
        row.result.rhs_count,
        row.result.residual,
        row.gap,
    )
end
