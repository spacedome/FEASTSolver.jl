using Printf

include(joinpath(@__DIR__, "src", "ProjectorMomentFEAST.jl"))
using .ProjectorMomentFEAST

cases = [
    (name="g1-p3", coupling=1.0, occupied=3, q=5, mixing=0.5, points=40),
    (name="g5-p3", coupling=5.0, occupied=3, q=6, mixing=0.3, points=40),
    (name="g10-p1", coupling=10.0, occupied=1, q=3, mixing=0.15, points=48),
    (name="g5-p4", coupling=5.0, occupied=4, q=8, mixing=0.3, points=48),
]

println("closure/leakage forcing sweep")
println("case      η     ok  refresh  inner      closure      leakage")
for case in cases
    problem = contact_mean_field_1d(
        points=case.points,
        half_length=7.0,
        coupling=case.coupling,
        occupied=case.occupied,
    )
    initial = initial_orbitals(problem; seed=77)
    chart = OccupiedChartPolicy(lower_bound=0.0, upper_bound=20.0, node_count=48)
    width = min(2, case.occupied)
    depth = cld(case.occupied, width)
    for forcing in (0.25, 0.5, 1.0, 2.0, 4.0)
        result = solve_two_timescale_nlfeast(
            problem,
            chart,
            initial;
            config=TwoTimescaleConfig(
                moment_depth=depth,
                probe_width=width,
                subspace_dimension=case.q,
                refresh_iterations=35,
                maximum_inner_iterations=50,
                forcing_ratio=forcing,
                inner_mixing=case.mixing,
                density_tolerance=1e-7,
                residual_tolerance=1e-7,
            ),
        )
        @printf(
            "%-8s %4.2f  %-5s %7d %6d   %10.3e   %10.3e\n",
            case.name,
            forcing,
            string(result.converged),
            result.refresh_count,
            result.inner_iterations,
            result.closure_defect,
            result.residual,
        )
    end
end
