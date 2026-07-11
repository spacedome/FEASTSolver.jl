using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "..", "src", "FusedNLFEAST.jl"))
using .FusedNLFEAST

function boundary_case(delta)
    merge(
        scalar_sine_case(),
        (
            center=pi / 2 + delta,
            radius=pi / 2,
            inside=pi,
            outside=0.0,
        ),
    )
end

println("δ        nodes  count reliable  converged  iterations  λ error       residual       status")
for delta in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3), nodes in (16, 32, 64, 128, 256)
    case = boundary_case(delta)
    chart = CircularChart(case.center, case.radius, nodes)
    count = argument_principle_count(
        z -> cos(z) / sin(z),
        chart;
        integrality_tolerance=1e-6,
        refinement_tolerance=1e-6,
    )
    result = try
        fused_nlfeast(
            case.T,
            case.right_solve,
            case.left_solve,
            chart,
            ones(ComplexF64, 1, 1),
            ones(ComplexF64, 1, 1);
            config=FusedConfig(
                moment_count=3,
                iterations=15,
                ranktol=1e-11,
                residual_ranktol=1e-14,
                residual_tol=1e-11,
                target_count=1,
                coupling=:independent,
                maxrank=3,
            ),
        )
    catch error
        error
    end
    if result isa Exception
        @printf(
            "%-8.1e %-6d %-5d %-9s %-10s %-11s %-13s %-14s %s\n",
            delta,
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
    value_error = length(result.extraction.values) == 1 ?
        abs(only(result.extraction.values) - case.inside) : Inf
    residual = isempty(result.extraction.residuals) ? Inf : maximum(result.extraction.residuals)
    @printf(
        "%-8.1e %-6d %-5d %-9s %-10s %-11d %-13.3e %-14.3e %s\n",
        delta,
        nodes,
        count.count,
        string(count.reliable),
        string(result.converged),
        length(result.history) - 1,
        value_error,
        residual,
        string(result.termination_reason),
    )
end
