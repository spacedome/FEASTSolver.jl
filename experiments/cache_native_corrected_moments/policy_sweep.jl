using Printf

include(joinpath(@__DIR__, "src", "CacheNativeCorrectedMoments.jl"))
using .CacheNativeCorrectedMoments

const PMF = CacheNativeCorrectedMoments.ProjectorMomentFEAST

problem = PMF.contact_mean_field_1d(
    points=40,
    half_length=7.0,
    coupling=5.0,
    occupied=3,
)
initial = PMF.initial_orbitals(problem; seed=77)
chart = PMF.OccupiedChartPolicy(
    lower_bound=0.0,
    upper_bound=20.0,
    node_count=48,
)

cases = [
    (name="window h=4", blocks=4, inner=2, policy=:none, period=1),
    (name="cache h=4/k=3", blocks=4, inner=2, policy=:response, period=3),
    (name="adaptive h=4", blocks=4, inner=2, policy=:adaptive_response, period=3),
    (name="window h=3", blocks=3, inner=2, policy=:none, period=1),
    (name="cache h=3/k=3", blocks=3, inner=2, policy=:response, period=3),
    (name="growing", blocks=0, inner=3, policy=:none, period=1),
    (name="cache grow/k=3", blocks=0, inner=3, policy=:response, period=3),
    (name="adaptive grow", blocks=0, inner=3, policy=:adaptive_response, period=3),
]

println("cache-native corrected-moment policy sweep")
println("variant           ok  refresh  factors  base RHS  response RHS   q")
results = NamedTuple[]
for case in cases
    result = solve_cache_native_nlfeast(
        problem,
        chart,
        initial;
        config=CacheNativeConfig(
            moment_depth=3,
            probe_width=1,
            window_blocks=case.blocks,
            refresh_iterations=25,
            pre_update=case.policy,
            response_warmup_refreshes=2,
            response_warmup_mixing=0.6,
            response_period=case.period,
            response_krylov_tolerance=0.03,
            reduced_inner_iterations=case.inner,
            reduced_mixing=0.5,
            density_tolerance=1e-7,
            residual_tolerance=1e-7,
        ),
    )
    push!(results, merge(case, (result=result,)))
    @printf(
        "%-17s %-5s %7d %8d %9d %13d %3d\n",
        case.name,
        string(result.converged),
        result.refresh_count,
        result.factorization_count,
        result.base_rhs_count,
        result.response_rhs_count,
        size(result.basis, 2),
    )
end

println("\nresponse crossover relative to the matching window")
println("variant           saved factors  added RHS  RHS per saved factor")
for (baseline_index, combined_index) in ((1, 2), (4, 5), (6, 7))
    baseline = results[baseline_index]
    combined = results[combined_index]
    saved = baseline.result.factorization_count - combined.result.factorization_count
    added = combined.result.base_rhs_count + combined.result.response_rhs_count -
        baseline.result.base_rhs_count
    @printf(
        "%-17s %13d %10d %21.2f\n",
        combined.name,
        saved,
        added,
        saved > 0 ? added / saved : Inf,
    )
end

println("\nparametric winner when one node factorization costs χ RHS solves")
println("χ       winner             weighted work")
for factor_value in (1, 5, 10, 20, 50, 100, 500)
    costs = [
        row.result.factorization_count * factor_value +
        row.result.base_rhs_count + row.result.response_rhs_count
        for row in results
    ]
    winner = argmin(costs)
    @printf(
        "%-7d %-18s %13d\n",
        factor_value,
        results[winner].name,
        costs[winner],
    )
end
