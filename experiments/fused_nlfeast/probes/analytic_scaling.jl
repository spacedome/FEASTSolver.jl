using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "..", "src", "FusedNLFEAST.jl"))
using .FusedNLFEAST

function multiset_distance(actual, expected)
    length(actual) == length(expected) || return Inf
    order(values) = sort(ComplexF64.(values); by=value -> (real(value), imag(value)))
    maximum(abs.(order(actual) .- order(expected)))
end

function scaled_sine(alpha)
    scale = z -> exp(alpha * z)
    T = z -> fill(ComplexF64(scale(z) * sin(z)), 1, 1)
    right_solve = (z, B) -> B ./ (scale(z) * sin(z))
    left_solve = (z, B) -> B ./ conj(scale(z) * sin(z))
    (T=T, right_solve=right_solve, left_solve=left_solve)
end

chart = CircularChart(0.0 + 0.0im, 4.0, 128)
expected = ComplexF64[-pi, 0, pi]

println("α      σ₃/σ₁       detected rank  returned  λ error       status")
for alpha in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    case = scaled_sine(alpha)
    cache = ContourSampleCache(chart, case.right_solve, case.left_solve)
    probe = ones(ComplexF64, 1, 1)
    add_right_probe!(cache, :initial, probe)
    add_left_probe!(cache, :initial, probe)
    moments = probe_moments(cache, :initial, :right, 6)
    H0, _ = FusedNLFEAST.block_hankel(moments, 3; observer=probe)
    singulars = svdvals(H0)
    detected = FusedNLFEAST.numerical_rank(singulars, 1e-12, 3)
    result = try
        fused_nlfeast(
            case.T,
            case.right_solve,
            case.left_solve,
            chart,
            probe,
            probe;
            config=FusedConfig(
                moment_count=3,
                iterations=8,
                ranktol=1e-12,
                residual_tol=1e-12,
                residual_metric=(Tvalue, vector, side, value) -> begin
                    action = side === :right ? Tvalue * vector : adjoint(Tvalue) * vector
                    norm(action) / (abs(exp(alpha * value)) * norm(vector))
                end,
                target_count=3,
                coupling=:independent,
            ),
        )
    catch error
        error
    end
    if result isa Exception
        @printf(
            "%-6.1f %-12.3e %-14d %-9s %-13s %s\n",
            alpha,
            singulars[3] / singulars[1],
            detected,
            "error",
            "Inf",
            sprint(showerror, result),
        )
    else
        @printf(
            "%-6.1f %-12.3e %-14d %-9d %-13.3e %s\n",
            alpha,
            singulars[3] / singulars[1],
            detected,
            length(result.extraction.values),
            multiset_distance(result.extraction.values, expected),
            string(result.termination_reason),
        )
    end
end
