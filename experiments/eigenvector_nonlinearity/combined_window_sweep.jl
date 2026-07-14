using Printf

include(joinpath(@__DIR__, "src", "ProjectorMomentFEAST.jl"))
using .ProjectorMomentFEAST

base = contact_mean_field_1d(points=48, half_length=6.0, coupling=1.0, occupied=3)
problem = quadratic_density_nep(base; strength=0.02)
initial = initial_orbitals(base; seed=33)
chart = CircularChart(1.8, 1.5, 128)

direct = solve_combined_scf(
    problem,
    chart,
    initial;
    config=CombinedSCFConfig(
        moment_depth=3,
        probe_width=1,
        iterations=30,
        outer_method=:anderson,
        history_depth=10,
        mixing=0.6,
        density_tolerance=1e-8,
        residual_tolerance=1e-8,
    ),
)
rows = NamedTuple[(name="direct", result=direct)]
for blocks in (1, 2, 3, 4, 0)
    result = solve_combined_windowed_nlfeast(
        problem,
        chart,
        initial;
        config=CombinedWindowConfig(
            moment_depth=3,
            probe_width=1,
            window_blocks=blocks,
            refresh_iterations=25,
            inner_iterations=3,
            inner_mixing=0.6,
            density_tolerance=1e-8,
            residual_tolerance=1e-8,
        ),
    )
    name = blocks == 0 ? "growing" : "window h=$blocks"
    push!(rows, (name=name, result=result))
end

println("combined spectral/state window sweep")
println("method       ok  refresh   q   factors   RHS      residual")
for row in rows
    result = row.result
    refreshes = hasproperty(result, :refresh_count) ? result.refresh_count : length(result.history)
    dimension = hasproperty(result, :basis) ? size(result.basis, 2) : base.occupied
    @printf(
        "%-12s %-5s %7d %3d %9d %5d   %10.3e\n",
        row.name,
        string(result.converged),
        refreshes,
        dimension,
        result.solve_count,
        result.rhs_count,
        result.residual,
    )
end
