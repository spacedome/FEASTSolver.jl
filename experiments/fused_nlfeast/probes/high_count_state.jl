using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "..", "src", "FusedNLFEAST.jl"))
using .FusedNLFEAST

case = scalar_sine_case()
coefficients = (ones(ComplexF64, 1, 1),)
functions = (sin,)

function state_error(state, depth)
    components = state_components(state)
    lifted_residual_error(
        components.right_state,
        components.right,
        components.right_residual,
        components.left_state,
        components.left,
        components.left_residual;
        lift_depth=depth,
    )
end

println("states  nodes  status       recovered  λ error       residual       time")
for states in (15, 31, 63, 127)
    half = (states - 1) ÷ 2
    radius = (half + 0.45) * pi
    nodes = 4 * states
    chart = CircularChart(0.0, radius, nodes)
    expected = ComplexF64[k * pi for k in -half:half]
    result = nothing
    seconds = @elapsed result = try
        structured_fused_state_nlfeast(
            coefficients,
            functions,
            case.right_solve,
            case.left_solve,
            chart,
            ones(ComplexF64, 1, 1),
            ones(ComplexF64, 1, 1);
            config=StateIterationConfig(
                moment_count=states,
                iterations=4,
                ranktol=1e-12,
                residual_ranktol=1e-14,
                residual_tol=1e-10,
                target_count=states,
                maxrank=states,
                tangent_width=1,
                compress_after=0,
                rollback_ratio=1.05,
            ),
            state_error=state -> state_error(state, states),
        )
    catch error
        error
    end
    if result isa Exception
        @printf(
            "%-7d %-6d %-12s %-10d %-13s %-14s %.3f  %s\n",
            states,
            nodes,
            "error",
            0,
            "Inf",
            "Inf",
            seconds,
            sprint(showerror, result),
        )
        continue
    end
    values = eigvals(state_components(result.state).right_state)
    _, errors = FusedNLFEAST.bottleneck_match(values, expected)
    @printf(
        "%-7d %-6d %-12s %-10d %-13.3e %-14.3e %.3f\n",
        states,
        nodes,
        string(result.termination_reason),
        length(values),
        maximum(errors),
        result.history[end].error,
        seconds,
    )
end
