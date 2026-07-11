using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "..", "src", "FusedNLFEAST.jl"))
using .FusedNLFEAST

case = scalar_sine_case()
coefficients = (ones(ComplexF64, 1, 1),)
functions = (sin,)

println("δ        nodes  count reliable  certified  updates  λ error       residual       margin")
for δ in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5), nodes in (16, 32, 64, 128, 256)
    center = pi / 2 + δ
    radius = pi / 2
    chart = CircularChart(center, radius, nodes)
    count = argument_principle_count(
        z -> cos(z) / sin(z),
        chart;
        integrality_tolerance=1e-7,
        refinement_tolerance=1e-7,
    )
    result = try
        action_error = function (state)
            components = state_components(state)
            lifted_residual_error(
                components.right_state,
                components.right,
                components.right_residual,
                components.left_state,
                components.left,
                components.left_residual;
                lift_depth=3,
            )
        end
        structured_fused_state_nlfeast(
            coefficients,
            functions,
            case.right_solve,
            case.left_solve,
            chart,
            ones(ComplexF64, 1, 1),
            ones(ComplexF64, 1, 1);
            config=StateIterationConfig(
                moment_count=3,
                iterations=15,
                ranktol=1e-11,
                residual_ranktol=1e-14,
                residual_tol=1e-11,
                target_count=1,
                target_count_certified=count.reliable && count.count == 1,
                maxrank=3,
                rollback_ratio=1.05,
            ),
            state_error=action_error,
        )
    catch error
        error
    end
    if result isa Exception
        @printf(
            "%-8.1e %-6d %-5d %-9s %-10s %-8s %-13s %-14s %s\n",
            δ,
            nodes,
            count.count,
            string(count.reliable),
            "error",
            "-",
            "Inf",
            "Inf",
            sprint(showerror, result),
        )
        continue
    end
    components = state_components(result.state)
    value = only(eigvals(components.right_state))
    margin = 1 - abs(FusedNLFEAST.chart_coordinate(chart, value))
    @printf(
        "%-8.1e %-6d %-5d %-9s %-10s %-8d %-13.3e %-14.3e %.3e\n",
        δ,
        nodes,
        count.count,
        string(count.reliable),
        string(result.certified),
        length(result.history) - 1,
        abs(value - pi),
        result.history[end].error,
        margin,
    )
end
