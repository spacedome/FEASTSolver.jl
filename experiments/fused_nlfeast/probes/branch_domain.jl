using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "..", "src", "FusedNLFEAST.jl"))
using .FusedNLFEAST

function branch_case(branch_point, beta)
    f = z -> sqrt(z - branch_point) - beta
    df = z -> inv(2 * sqrt(z - branch_point))
    T = z -> fill(ComplexF64(f(z)), 1, 1)
    (
        f=f,
        df=df,
        T=T,
        right_solve=(z, rhs) -> rhs ./ f(z),
        left_solve=(z, rhs) -> rhs ./ conj(f(z)),
        root=branch_point + beta^2,
    )
end

function run_case(label, branch_point, beta, radius)
    case = branch_case(ComplexF64(branch_point), ComplexF64(beta))
    println(label, ": root=", case.root, ", branch point=", branch_point)
    for nodes in (32, 64, 128, 256)
        chart = CircularChart(0.0 + 0.0im, radius, nodes)
        count = argument_principle_count(
            z -> case.df(z) / case.f(z),
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
                    iterations=5,
                    ranktol=1e-10,
                    residual_tol=1e-11,
                    residual_scale=1.0,
                    coupling=:independent,
                    maxrank=6,
                ),
                analytic_domain=input_chart -> begin
                    left_edge = real(input_chart.center) - input_chart.radius
                    (
                        valid=left_edge > real(branch_point),
                        reason="principal square-root branch cut crosses the chart",
                    )
                end,
            )
        catch error
            error
        end
        if result isa Exception
            @printf(
                "  N=%-4d index=% .6f%+.2ei reliable=%-5s extraction=%s\n",
                nodes,
                real(count.value),
                imag(count.value),
                string(count.reliable),
                sprint(showerror, result),
            )
        else
            values = result.extraction.values
            root_error = isempty(values) ? Inf : minimum(abs.(values .- case.root))
            residual = isempty(values) ? Inf : maximum(abs.(case.f.(values)))
            @printf(
                "  N=%-4d index=% .6f%+.2ei reliable=%-5s states=%-2d root error=%.3e residual=%.3e\n",
                nodes,
                real(count.value),
                imag(count.value),
                string(count.reliable),
                length(values),
                root_error,
                residual,
            )
        end
    end
end

run_case("valid sheet", -2.0, 1.5, 1.0)
run_case("near branch point", -1.01, sqrt(1.25), 1.0)
run_case("invalid cross-cut chart", -0.5, 1.0, 1.0)
