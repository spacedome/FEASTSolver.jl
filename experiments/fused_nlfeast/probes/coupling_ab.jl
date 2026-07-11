using Statistics

include(joinpath(@__DIR__, "state_policy_matrix.jl"))

println("loaded-string divided-difference coupling A/B, 10 seeds")
println("nodes  coupling     converged  median updates  worst λ error  worst backward error")

for nodes in (16, 24, 32), coupling in (:common, :independent)
    rows = [
        run_driver(
            nodes,
            seed;
            updates=10,
            residual_tol=1e-11,
            coupling=coupling,
        )
        for seed in 1:10
    ]
    updates = [length(row.result.history) - 1 for row in rows]
    @printf(
        "%-6d %-12s %-10d %-15.1f %-14.3e %.3e\n",
        nodes,
        String(coupling),
        count(row -> row.result.converged, rows),
        median(updates),
        maximum(row.λ_error for row in rows),
        maximum(row.backward_error for row in rows),
    )
end
