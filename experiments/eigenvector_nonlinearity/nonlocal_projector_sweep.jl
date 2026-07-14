using Printf

include(joinpath(@__DIR__, "src", "ProjectorMomentFEAST.jl"))
using .ProjectorMomentFEAST

problem = nonlocal_projector_1d(points=40, half_length=7.0, coupling=4.2, occupied=3)
initial = initial_orbitals(problem; seed=33)
reference = reference_projector_scf(
    problem,
    projector_state(initial);
    mixing=0.1,
    tolerance=1e-10,
)
chart = OccupiedChartPolicy(lower_bound=0.0, upper_bound=15.0, node_count=96)

rows = NamedTuple[]
for q in (5, 6, 8, 10)
    result = solve_projector_two_timescale(
        problem,
        chart,
        initial;
        config=TwoTimescaleConfig(
            moment_depth=3,
            probe_width=1,
            subspace_dimension=q,
            refresh_iterations=30,
            maximum_inner_iterations=40,
            forcing_ratio=1.0,
            inner_mixing=0.3,
            density_tolerance=1e-7,
            residual_tolerance=1e-7,
        ),
    )
    push!(rows, (name="fixed q=$q", result=result))
end

adaptive = solve_projector_two_timescale(
    problem,
    chart,
    initial;
    config=TwoTimescaleConfig(
        moment_depth=3,
        probe_width=1,
        subspace_dimension=3,
        maximum_subspace_dimension=12,
        growth_increment=3,
        stagnation_ratio=0.4,
        refresh_iterations=30,
        maximum_inner_iterations=40,
        forcing_ratio=1.0,
        inner_mixing=0.3,
        density_tolerance=1e-7,
        residual_tolerance=1e-7,
    ),
)
push!(rows, (name="adaptive", result=adaptive))

for blocks in (2, 3, 4, 0)
    result = solve_projector_windowed_nlfeast(
        problem,
        chart,
        initial;
        config=ProjectorWindowConfig(
            moment_depth=3,
            probe_width=1,
            window_blocks=blocks,
            refresh_iterations=30,
            inner_iterations=5,
            inner_method=:anderson,
            inner_mixing=0.6,
            projector_tolerance=1e-7,
            residual_tolerance=1e-7,
        ),
    )
    name = blocks == 0 ? "growing" : "window h=$blocks"
    push!(rows, (name=name, result=result))
end

println("nonlocal projector schedule sweep")
println("method          ok  refresh  inner   q   factors   RHS      residual          gap")
for row in rows
    result = row.result
    @printf(
        "%-14s %-5s %7d %6d %3d %9d %5d   %10.3e   %10.3e\n",
        row.name,
        string(result.converged),
        result.refresh_count,
        result.inner_iterations,
        size(result.basis, 2),
        result.solve_count,
        result.rhs_count,
        result.residual,
        subspace_gap(result.orbitals, reference.orbitals),
    )
end
