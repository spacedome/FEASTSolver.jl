using Printf

include(joinpath(@__DIR__, "src", "ProjectorMomentFEAST.jl"))
using .ProjectorMomentFEAST

problem = contact_mean_field_1d(points=48, half_length=6.0, coupling=5.0, occupied=3)
initial = initial_orbitals(problem; seed=33)
forms = ((3, 1), (2, 2), (3, 2), (2, 3), (1, 5))

println("finite-quadrature moment/oversampling sweep")
println(" N  memory  d×ℓ   ok  refresh    q    RHS")
for nodes in (32, 64), window in (3, 0), (depth, width) in forms
    chart = OccupiedChartPolicy(lower_bound=0.0, upper_bound=15.0, node_count=nodes)
    result = solve_raw_windowed_nlfeast(
        problem,
        chart,
        initial;
        config=WindowedNLFEASTConfig(
            moment_depth=depth,
            probe_width=width,
            window_blocks=window,
            refresh_iterations=25,
            inner_schedule=:fixed,
            fixed_inner_iterations=window == 0 ? 5 : 3,
            inner_mixing=0.3,
            density_tolerance=1e-8,
            residual_tolerance=1e-8,
        ),
    )
    @printf(
        "%2d  %-6s  %d×%d  %-5s %7d %4d %6d\n",
        nodes,
        window == 0 ? "grow" : "h=3",
        depth,
        width,
        string(result.converged),
        result.refresh_count,
        size(result.basis, 2),
        result.rhs_count,
    )
end
