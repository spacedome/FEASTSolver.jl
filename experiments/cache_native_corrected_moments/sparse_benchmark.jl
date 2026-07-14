using Printf

include(joinpath(@__DIR__, "src", "CacheNativeCorrectedMoments.jl"))
using .CacheNativeCorrectedMoments

const PMF = CacheNativeCorrectedMoments.ProjectorMomentFEAST

problem = PMF.contact_mean_field_1d(
    points=256,
    half_length=8.0,
    coupling=5.0,
    occupied=4,
    sparse=true,
)
initial = PMF.initial_orbitals(problem; seed=903)
chart = PMF.OccupiedChartPolicy(
    lower_bound=0.0,
    upper_bound=20.0,
    node_count=48,
)

cases = [
    (name="growing i=5", blocks=0, inner=5, policy=:none, period=1),
    (name="growing adaptive", blocks=0, inner=3, policy=:adaptive_response, period=3),
    (name="h4 i=2", blocks=4, inner=2, policy=:none, period=1),
    (name="h4 response/3", blocks=4, inner=2, policy=:response, period=3),
    (name="h4 adaptive", blocks=4, inner=2, policy=:adaptive_response, period=3),
    (name="h4 i=3", blocks=4, inner=3, policy=:none, period=1),
]

function run_case(case)
    solve_cache_native_nlfeast(
        problem,
        chart,
        initial;
        config=CacheNativeConfig(
            moment_depth=2,
            probe_width=2,
            window_blocks=case.blocks,
            refresh_iterations=20,
            pre_update=case.policy,
            response_warmup_refreshes=2,
            response_warmup_mixing=0.6,
            response_period=case.period,
            response_forcing_ratio=1.0,
            response_krylov_tolerance=0.03,
            response_krylov_iterations=20,
            reduced_inner_iterations=case.inner,
            reduced_mixing=0.3,
            density_tolerance=1e-8,
            residual_tolerance=1e-8,
        ),
    )
end

for case in cases
    run_case(case)
end

println("warmed sparse cache-policy comparison")
println("method             time  ok  refresh  factors  base RHS  response RHS   q  inner")
for case in cases
    elapsed = @elapsed result = run_case(case)
    @printf(
        "%-16s %6.3f %-5s %7d %8d %9d %13d %3d %6d\n",
        case.name,
        elapsed,
        string(result.converged),
        result.refresh_count,
        result.factorization_count,
        result.base_rhs_count,
        result.response_rhs_count,
        size(result.basis, 2),
        sum(record.reduced_inner_iterations for record in result.history),
    )
end
