using Printf
using SparseArrays

include(joinpath(@__DIR__, "src", "ProjectorMomentFEAST.jl"))
using .ProjectorMomentFEAST

problem = contact_mean_field_1d(
    points=256,
    half_length=8.0,
    coupling=5.0,
    occupied=4,
    sparse=true,
)
initial = initial_orbitals(problem; seed=903)
chart = OccupiedChartPolicy(lower_bound=0.0, upper_bound=20.0, node_count=48)

anderson_config = AndersonConfig(
    moment_depth=2,
    probe_width=2,
    iterations=50,
    history_depth=10,
    mixing=0.6,
    density_tolerance=1e-8,
    residual_tolerance=1e-8,
)
response_config = ResponseNewtonConfig(
    moment_depth=2,
    probe_width=2,
    iterations=25,
    warmup_iterations=2,
    warmup_method=:anderson,
    warmup_mixing=0.6,
    warmup_history_depth=10,
    krylov_method=:cg,
    adaptive_krylov=true,
    krylov_tolerance=1e-7,
    maximum_krylov_tolerance=0.1,
    density_tolerance=1e-8,
    residual_tolerance=1e-8,
)
growing_config = WindowedNLFEASTConfig(
    moment_depth=2,
    probe_width=2,
    window_blocks=0,
    refresh_iterations=20,
    inner_schedule=:fixed,
    fixed_inner_iterations=5,
    inner_mixing=0.3,
    density_tolerance=1e-8,
    residual_tolerance=1e-8,
)
growing_response_config = WindowedNLFEASTConfig(
    moment_depth=2,
    probe_width=2,
    window_blocks=0,
    refresh_iterations=20,
    inner_schedule=:fixed,
    inner_method=:response,
    fixed_inner_iterations=1,
    inner_krylov_tolerance=0.03,
    inner_krylov_iterations=20,
    density_tolerance=1e-8,
    residual_tolerance=1e-8,
)

solvers = [
    (
        name="Anderson",
        run=() -> solve_anderson_scf(problem, chart, initial; config=anderson_config),
    ),
    (
        name="response",
        run=() -> solve_response_newton(problem, chart, initial; config=response_config),
    ),
    (
        name="growing moments",
        run=() -> solve_raw_windowed_nlfeast(
            problem,
            chart,
            initial;
            config=growing_config,
        ),
    ),
    (
        name="growing response",
        run=() -> solve_raw_windowed_nlfeast(
            problem,
            chart,
            initial;
            config=growing_response_config,
        ),
    ),
]

for solver in solvers
    solver.run()
end
rows = map(solvers) do solver
    elapsed = @elapsed result = solver.run()
    dimension = hasproperty(result, :basis) ? size(result.basis, 2) : problem.occupied
    (
        name=solver.name,
        elapsed=elapsed,
        result=result,
        dimension=dimension,
    )
end

println("warmed sparse memory-policy comparison")
println("method             time   factors    RHS  inertia  reduced  inner  response   residual")
for row in rows
    @printf(
        "%-16s %6.3f %9d %6d %8d %8d %6d %9d   %.2e\n",
        row.name,
        row.elapsed,
        row.result.solve_count,
        row.result.rhs_count,
        row.result.chart_factorization_count,
        row.dimension,
        hasproperty(row.result, :inner_iterations) ? row.result.inner_iterations : 0,
        hasproperty(row.result, :reduced_response_actions) ?
            row.result.reduced_response_actions : 0,
        row.result.residual,
    )
end
