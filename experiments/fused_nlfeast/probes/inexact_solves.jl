using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "..", "src", "FusedNLFEAST.jl"))
using .FusedNLFEAST

function multiset_distance(actual, expected)
    length(actual) == length(expected) || return Inf
    order(values) = sort(ComplexF64.(values); by=value -> (real(value), imag(value)))
    maximum(abs.(order(actual) .- order(expected)))
end

function inexact_sine_case(relative_error)
    T = z -> fill(ComplexF64(sin(z)), 1, 1)
    multiplier = z -> one(ComplexF64) + relative_error * cis(7 * angle(z) + 0.31)
    right_solve = (z, B) -> multiplier(z) .* B ./ sin(z)
    left_solve = (z, B) -> conj(multiplier(z)) .* B ./ conj(sin(z))
    (T=T, right_solve=right_solve, left_solve=left_solve)
end

expected = ComplexF64[k * pi for k in -3:3]
chart = CircularChart(0.0 + 0.0im, 10.0, 64)

println("η          converged  iterations  λ error       final residual  observed ratios")
for relative_error in (0.0, 1.0, 5e-1, 3e-1, 1e-1, 3e-2, 1e-2, 1e-3, 1e-4)
    case = inexact_sine_case(relative_error)
    result = fused_nlfeast(
        case.T,
        case.right_solve,
        case.left_solve,
        chart,
        ones(ComplexF64, 1, 1),
        ones(ComplexF64, 1, 1);
        config=FusedConfig(
            moment_count=7,
            iterations=15,
            ranktol=1e-11,
            residual_ranktol=1e-14,
            residual_tol=1e-12,
            residual_metric=(Tvalue, vector, side, value) -> begin
                action = side === :right ? Tvalue * vector : adjoint(Tvalue) * vector
                norm(action) / norm(vector)
            end,
            target_count=7,
            coupling=:independent,
        ),
    )
    residuals = [record.max_residual for record in result.history]
    ratios = [residuals[index + 1] / residuals[index] for index in 1:(length(residuals) - 1) if residuals[index] > 1e-13]
    @printf(
        "%-10.1e %-10s %-11d %-13.3e %-15.3e %s\n",
        relative_error,
        string(result.converged),
        length(result.history) - 1,
        multiset_distance(result.extraction.values, expected),
        maximum(result.extraction.residuals),
        repr(round.(ratios; sigdigits=3)),
    )
end
