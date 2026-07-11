using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "..", "src", "FusedNLFEAST.jl"))
using .FusedNLFEAST

function side_singular_ratio(case, chart, right_probe, left_probe)
    cache = ContourSampleCache(chart, case.right_solve, case.left_solve)
    add_right_probe!(cache, :initial, right_probe)
    add_left_probe!(cache, :initial, left_probe)
    right_moments = probe_moments(cache, :initial, :right, 4)
    left_moments = probe_moments(cache, :initial, :left, 4)
    right_H0, _ = FusedNLFEAST.block_hankel(right_moments, 2; observer=left_probe)
    left_H0, _ = FusedNLFEAST.block_hankel(left_moments, 2; observer=right_probe)
    right_singulars = svdvals(right_H0)
    left_singulars = svdvals(left_H0)
    raw_right = streaming_hankel_realization(
        right_moments,
        2;
        ranktol=1e-10,
        maxrank=2,
    )
    raw_left = streaming_hankel_realization(
        left_moments,
        2;
        ranktol=1e-10,
        maxrank=2,
    )
    raw_right_singulars = raw_right.singular_values
    raw_left_singulars = raw_left.singular_values
    (
        right=right_singulars[end] / right_singulars[1],
        left=left_singulars[end] / left_singulars[1],
        right_rank=FusedNLFEAST.numerical_rank(right_singulars, 1e-10, 2),
        left_rank=FusedNLFEAST.numerical_rank(left_singulars, 1e-10, 2),
        raw_right_rank=FusedNLFEAST.numerical_rank(raw_right_singulars, 1e-10, 2),
        raw_left_rank=FusedNLFEAST.numerical_rank(raw_left_singulars, 1e-10, 2),
    )
end

case = linear_case(Diagonal(ComplexF64[-0.5, 0.5]))
chart = CircularChart(0.0 + 0.0im, 1.0, 16)
strong = ComplexF64[1; 1;;]

println("ε          crossed σ₂/σ₁           crossed ranks  raw ranks  extraction")
for epsilon in (1.0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 0.0)
    weak = ComplexF64[1; epsilon;;]
    diagnostics = side_singular_ratio(case, chart, weak, strong)
    result = try
        fused_nlfeast(
            case.T,
            case.right_solve,
            case.left_solve,
            chart,
            weak,
            strong;
            config=FusedConfig(
                moment_count=2,
                iterations=0,
                ranktol=1e-10,
                residual_tol=1e-12,
                target_count=2,
                coupling=:independent,
            ),
        )
    catch error
        error
    end
    outcome = result isa Exception ? sprint(showerror, result) :
        "count=$(length(result.extraction.values)), $(result.termination_reason)"
    @printf(
        "%-10.1e %-13.3e %-13.3e (%d,%d)         (%d,%d)     %s\n",
        epsilon,
        diagnostics.right,
        diagnostics.left,
        diagnostics.right_rank,
        diagnostics.left_rank,
        diagnostics.raw_right_rank,
        diagnostics.raw_left_rank,
        outcome,
    )
end

full_probe = Matrix{ComplexF64}(I, 2, 2)
full = fused_nlfeast(
    case.T,
    case.right_solve,
    case.left_solve,
    chart,
    full_probe,
    full_probe;
    config=FusedConfig(
        moment_count=1,
        iterations=0,
        ranktol=1e-10,
        residual_tol=1e-12,
        target_count=2,
        coupling=:independent,
    ),
)
println("augmented probe: count=$(length(full.extraction.values)), $(full.termination_reason)")
