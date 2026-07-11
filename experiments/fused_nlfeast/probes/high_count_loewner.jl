using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "..", "src", "FusedNLFEAST.jl"))
using .FusedNLFEAST

case = scalar_sine_case()

println("states  nodes  points  rank  λ error       σₘ/σ₁        time")
for states in (15, 31, 63, 127)
    half = (states - 1) ÷ 2
    radius = (half + 0.45) * pi
    nodes = 8 * states
    point_count = states + 8
    chart = CircularChart(0.0, radius, nodes)
    cache = ContourSampleCache(chart, case.right_solve, case.left_solve)
    add_right_probe!(cache, :initial, ones(ComplexF64, 1, 1))
    left_points = ComplexF64[
        1.15 * cis(2pi * j / point_count) for j in 0:(point_count - 1)
    ]
    right_points = ComplexF64[
        1.35 * cis(2pi * (j + 0.5) / point_count) for j in 0:(point_count - 1)
    ]
    realization = nothing
    seconds = @elapsed begin
        full_left = rational_probe_samples(cache, :initial, :right, left_points)
        full_right = rational_probe_samples(cache, :initial, :right, right_points)
        realization = loewner_realization(
            left_points,
            full_left,
            right_points,
            full_right,
            full_right;
            ranktol=1e-12,
            maxrank=states,
        )
    end
    values = chart.center .+ chart.radius .* eigvals(realization.state)
    expected = ComplexF64[k * pi for k in -half:half]
    error = length(values) == states ?
        maximum(last(FusedNLFEAST.bottleneck_match(values, expected))) : Inf
    singular_ratio = length(realization.singular_values) >= states ?
        realization.singular_values[states] / realization.singular_values[1] : 0.0
    @printf(
        "%-7d %-6d %-7d %-5d %-13.3e %-13.3e %.3f\n",
        states,
        nodes,
        point_count,
        realization.rank,
        error,
        singular_ratio,
        seconds,
    )
end
