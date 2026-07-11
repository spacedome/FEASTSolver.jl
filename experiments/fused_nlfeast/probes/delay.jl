using LinearAlgebra
using Printf
using Random

include(joinpath(@__DIR__, "..", "src", "FusedNLFEAST.jl"))
using .FusedNLFEAST

function multiset_distance(actual, expected)
    length(actual) == length(expected) || return Inf
    count = length(actual)
    count == 0 && return 0.0
    count <= 20 || throw(ArgumentError("assignment metric is intended for small probe sets"))
    costs = abs.(ComplexF64.(actual) .- transpose(ComplexF64.(expected)))
    current = Dict(0 => 0.0)
    for row in 1:count
        next = Dict{Int,Float64}()
        for (mask, cost) in current, column in 1:count
            bit = 1 << (column - 1)
            iszero(mask & bit) || continue
            next_mask = mask | bit
            next_cost = max(cost, costs[row, column])
            next[next_mask] = min(get(next, next_mask, Inf), next_cost)
        end
        current = next
    end
    current[(1 << count) - 1]
end

a = 0.4
b = 2.0
tau = 1.0
f = z -> z + a - b * exp(-tau * z)
df = z -> one(z) + b * tau * exp(-tau * z)
T = z -> fill(ComplexF64(f(z)), 1, 1)
right_solve = (z, rhs) -> rhs ./ f(z)
left_solve = (z, rhs) -> rhs ./ conj(f(z))

count_chart = CircularChart(0.0 + 0.0im, 6.0, 256)
count_estimate = argument_principle_count(
    z -> df(z) / f(z),
    count_chart;
    integrality_tolerance=1e-10,
    refinement_tolerance=1e-10,
)
count_estimate.stable || error("delay reference count estimate is not stable")

reference = fused_nlfeast(
    T,
    right_solve,
    left_solve,
    count_chart,
    ones(ComplexF64, 1, 1),
    ones(ComplexF64, 1, 1);
    config=FusedConfig(
        moment_count=count_estimate.count,
        iterations=5,
        ranktol=1e-12,
        residual_tol=1e-13,
        residual_scale=1.0,
        target_count=count_estimate.count,
        coupling=:independent,
        maxrank=6,
    ),
)
reference_values = reference.extraction.values

println("count: ", count_estimate.count, ", reference values: ", reference_values)
println("modal iteration:")
println("nodes  successes  worst λ error  worst |f(λ)|")
for nodes in (12, 16, 24, 32, 48)
    rows = Any[]
    for seed in 1:10
        rng = MersenneTwister(seed)
        chart = CircularChart(0.0 + 0.0im, 6.0, nodes)
        result = try
            fused_nlfeast(
                T,
                right_solve,
                left_solve,
                chart,
                randn(rng, ComplexF64, 1, 1),
                randn(rng, ComplexF64, 1, 1);
                config=FusedConfig(
                    moment_count=count_estimate.count,
                    iterations=8,
                    ranktol=1e-10,
                    residual_tol=1e-11,
                    residual_scale=1.0,
                    target_count=count_estimate.count,
                    coupling=:independent,
                    maxrank=6,
                ),
            )
        catch error
            error
        end
        if result isa Exception
            push!(rows, result)
        else
            push!(rows, (
                value_error=multiset_distance(result.extraction.values, reference_values),
                residual=maximum(abs.(f.(result.extraction.values))),
            ))
        end
    end
    valid = filter(row -> !(row isa Exception), rows)
    successes = count(row -> row.value_error <= 1e-8 && row.residual <= 1e-10, valid)
    @printf(
        "%-6d %-10d %-14.3e %.3e\n",
        nodes,
        successes,
        isempty(valid) ? Inf : maximum(row.value_error for row in valid),
        isempty(valid) ? Inf : maximum(row.residual for row in valid),
    )
    exceptions = [sprint(showerror, row) for row in rows if row isa Exception]
    isempty(exceptions) || println("  exceptions: ", exceptions)
end

coefficients = (
    ones(ComplexF64, 1, 1),
    fill(ComplexF64(a), 1, 1),
    fill(ComplexF64(-b), 1, 1),
)
functions = (identity, one, z -> exp((-tau) * z))

function state_run(nodes, seed; updates=6)
    rng = MersenneTwister(seed)
    chart = CircularChart(0.0 + 0.0im, 6.0, nodes)
    cache = ContourSampleCache(chart, right_solve, left_solve)
    right_initial = add_right_probe!(cache, :initial, randn(rng, ComplexF64, 1, 1))
    left_initial = add_left_probe!(cache, :initial, randn(rng, ComplexF64, 1, 1))
    right_moments = probe_moments(cache, :initial, :right, 6)
    left_moments = probe_moments(cache, :initial, :left, 6)
    state = common_structured_realization(
        coefficients,
        functions,
        chart,
        right_moments,
        left_moments,
        left_initial.probe,
        right_initial.probe,
        3;
        ranktol=1e-10,
        maxrank=6,
        target_count=count_estimate.count,
    )
    for iteration in 1:updates
        factors = state_residual_factors(
            state.right_residual,
            state.left_residual;
            ranktol=1e-13,
        )
        right_id = Symbol(:right_, iteration)
        left_id = Symbol(:left_, iteration)
        add_right_probe!(cache, right_id, factors.right_basis; role=:residual)
        add_left_probe!(cache, left_id, factors.left_basis; role=:residual)
        right_moments, left_moments = state_corrected_moments(
            cache,
            state.state,
            state.state,
            state.right,
            state.left,
            right_id,
            left_id,
            factors.right_coefficients,
            factors.left_coefficients,
            6,
        )
        state = common_structured_realization(
            coefficients,
            functions,
            chart,
            right_moments,
            left_moments,
            left_initial.probe,
            right_initial.probe,
            3;
            ranktol=1e-10,
            maxrank=6,
            fixed_rank=state.rank,
            restrict_to_chart=false,
        )
    end
    values = eigvals(state.state)
    (
        value_error=multiset_distance(values, reference_values),
        residual=maximum(abs.(f.(values))),
    )
end

println("state iteration:")
println("nodes  successes  worst λ error  worst |f(λ)|")
for nodes in (12, 16, 24, 32, 48)
    rows = Any[
        try
            state_run(nodes, seed)
        catch error
            error
        end
        for seed in 1:10
    ]
    valid = filter(row -> !(row isa Exception), rows)
    successes = count(row -> row.value_error <= 1e-8 && row.residual <= 1e-10, valid)
    @printf(
        "%-6d %-10d %-14.3e %.3e\n",
        nodes,
        successes,
        isempty(valid) ? Inf : maximum(row.value_error for row in valid),
        isempty(valid) ? Inf : maximum(row.residual for row in valid),
    )
    exceptions = [sprint(showerror, row) for row in rows if row isa Exception]
    isempty(exceptions) || println("  exceptions: ", exceptions)
end
