using Printf

include(joinpath(@__DIR__, "src", "ProjectorMomentFEAST.jl"))
using .ProjectorMomentFEAST

cases = [
    (coupling=1.0, occupied=3, dimension=5, mixing=0.5),
    (coupling=5.0, occupied=3, dimension=6, mixing=0.3),
    (coupling=10.0, occupied=1, dimension=5, mixing=0.15),
]

println("cross-refresh Anderson history")
println("g      p   q   policy     ok    refresh  inner      closure      leakage")
for case in cases
    problem = contact_mean_field_1d(
        points=48,
        half_length=7.0,
        coupling=case.coupling,
        occupied=case.occupied,
    )
    initial = initial_orbitals(problem; seed=77)
    chart = OccupiedChartPolicy(lower_bound=0.0, upper_bound=20.0, node_count=48)
    width = min(2, case.occupied)
    depth = cld(case.occupied, width)
    for policy in (:restart, :physical)
        result = solve_two_timescale_nlfeast(
            problem,
            chart,
            initial;
            config=TwoTimescaleConfig(
                moment_depth=depth,
                probe_width=width,
                subspace_dimension=case.dimension,
                refresh_iterations=35,
                maximum_inner_iterations=50,
                forcing_ratio=1.0,
                inner_mixing=case.mixing,
                density_tolerance=1e-7,
                residual_tolerance=1e-7,
                history_policy=policy,
            ),
        )
        @printf(
            "%-5.1f %2d %3d   %-8s %-5s %7d %6d   %.3e   %.3e\n",
            case.coupling,
            case.occupied,
            case.dimension,
            string(policy),
            string(result.converged),
            result.refresh_count,
            result.inner_iterations,
            result.closure_defect,
            result.residual,
        )
    end
end
