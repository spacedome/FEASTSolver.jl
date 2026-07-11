using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "..", "src", "FusedNLFEAST.jl"))
using .FusedNLFEAST

function inexact_sine_case(relative_error)
    multiplier = z -> one(ComplexF64) + relative_error * cis(7 * angle(z) + 0.31)
    (
        right_solve=(z, block) -> multiplier(z) .* block ./ sin(z),
        left_solve=(z, block) -> conj(multiplier(z)) .* block ./ conj(sin(z)),
    )
end

function bottleneck_distance(actual, expected)
    length(actual) == length(expected) || return Inf
    _, errors = FusedNLFEAST.bottleneck_match(actual, expected)
    maximum(errors)
end

expected = ComplexF64[k * pi for k in -3:3]
chart = CircularChart(0.0, 10.0, 64)
coefficients = (ones(ComplexF64, 1, 1),)
functions = (sin,)

println("η          converged  updates  λ error       residual       ratios")
for η in (0.0, 0.5, 0.3, 0.1, 0.03, 0.01, 0.001)
    case = inexact_sine_case(η)
    error_metric = function (state)
        components = state_components(state)
        lifted_residual_error(
            components.right_state,
            components.right,
            components.right_residual,
            components.left_state,
            components.left,
            components.left_residual;
            lift_depth=7,
        )
    end
    result = structured_fused_state_nlfeast(
        coefficients,
        functions,
        case.right_solve,
        case.left_solve,
        chart,
        ones(ComplexF64, 1, 1),
        ones(ComplexF64, 1, 1);
        config=StateIterationConfig(
            moment_count=7,
            iterations=15,
            ranktol=1e-11,
            residual_ranktol=1e-14,
            residual_tol=1e-12,
            target_count=7,
            maxrank=7,
            rollback_ratio=1.05,
        ),
        state_error=error_metric,
    )
    components = state_components(result.state)
    values = eigvals(components.right_state)
    residuals = [record.error for record in result.history]
    ratios = [
        residuals[k + 1] / residuals[k]
        for k in 1:(length(residuals) - 1) if residuals[k] > 1e-13
    ]
    @printf(
        "%-10.2e %-10s %-8d %-13.3e %-14.3e %s\n",
        η,
        string(result.converged),
        length(result.history) - 1,
        bottleneck_distance(values, expected),
        result.history[end].error,
        repr(round.(ratios; sigdigits=3)),
    )
end
