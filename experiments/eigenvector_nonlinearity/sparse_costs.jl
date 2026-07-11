using LinearAlgebra
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
chart = CircularChart(5.3, 1.35, 48)
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

# Compile both paths before timing.
solve_anderson_scf(problem, chart, initial; config=anderson_config)
solve_response_newton(problem, chart, initial; config=response_config)

anderson_time = @elapsed anderson = solve_anderson_scf(
    problem,
    chart,
    initial;
    config=anderson_config,
)
response_time = @elapsed response = solve_response_newton(
    problem,
    chart,
    initial;
    config=response_config,
)

println("warmed sparse outer-update comparison")
@printf(
    "Anderson  %.3f s  outer=%d  factors=%d  RHS=%d  residual=%.2e\n",
    anderson_time,
    length(anderson.history),
    anderson.solve_count,
    anderson.rhs_count,
    anderson.residual,
)
@printf(
    "response  %.3f s  outer=%d  factors=%d  RHS=%d  residual=%.2e\n",
    response_time,
    length(response.history),
    response.solve_count,
    response.rhs_count,
    response.residual,
)
@printf("response speedup %.2f×\n", anderson_time / response_time)

