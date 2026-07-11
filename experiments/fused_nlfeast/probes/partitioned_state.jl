using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "..", "src", "FusedNLFEAST.jl"))
using .FusedNLFEAST

case = scalar_sine_case()
coefficients = (ones(ComplexF64, 1, 1),)
functions = (sin,)
T = z -> fill(ComplexF64(sin(z)), 1, 1)

function certified_sine_count(chart::RectangularChart)
    lower_x = real(chart.center) - chart.half_width
    upper_x = real(chart.center) + chart.half_width
    lower_y = imag(chart.center) - chart.half_height
    upper_y = imag(chart.center) + chart.half_height
    root_count = if lower_y < 0 < upper_y
        Base.count(k -> lower_x < k * pi < upper_x, -1000:1000)
    else
        0
    end
    (count=root_count, stable=true, reliable=true)
end

count_mode = get(ENV, "FUSED_SINE_COUNT", "winding")
count_backend = if count_mode == "exact"
    certified_sine_count
elseif count_mode == "winding"
    chart -> determinant_winding_count(
        T,
        chart;
        derivative_bound=(start, stop) -> cosh(max(abs(imag(start)), abs(imag(stop)))),
    )
else
    error("FUSED_SINE_COUNT must be exact or winding")
end

function solve_leaf(chart, estimate)
    count = estimate.count
    error_metric = function (state)
        components = state_components(state)
        lifted_residual_error(
            components.right_state,
            components.right,
            components.right_residual,
            components.left_state,
            components.left,
            components.left_residual;
            lift_depth=count,
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
            moment_count=count,
            iterations=6,
            ranktol=1e-11,
            residual_ranktol=1e-14,
            residual_tol=1e-10,
            target_count=count,
            target_count_certified=true,
            maxrank=count,
            rollback_ratio=1.05,
        ),
        state_error=error_metric,
    )
end

parent = RectangularChart(0.0, 49.0, 10.0, 128)
seconds = @elapsed result = partitioned_state_solve(
    parent;
    count_backend=count_backend,
    solve_leaf=solve_leaf,
    accept_leaf=(leaf, estimate) -> leaf.certified,
    capacity=6,
    max_depth=8,
    max_count_refinements=2,
)

modal_leaves = [state_modal_output(leaf.result.state) for leaf in result.leaves]
all(modal -> modal.available, modal_leaves) || error(
    "a sine partition leaf did not admit modal output",
)
values = reduce(vcat, (modal.values for modal in modal_leaves); init=ComplexF64[])
assembled = partitioned_invariant_state(result)
assembled_components = state_components(assembled)
worst_residual = maximum(
    leaf.result.history[end].error for leaf in result.leaves;
    init=0.0,
)
expected = ComplexF64[k * pi for k in -15:15]
_, errors = FusedNLFEAST.bottleneck_match(values, expected)

println("partitioned common-state sine")
println("count backend: ", count_mode)
@printf("recovered: %d\n", length(values))
@printf("max λ error: %.3e\n", maximum(errors))
@printf("max residual: %.3e\n", worst_residual)
println("count conserved: ", result.count_conserved)
println("assembled state rank: ", size(assembled_components.right_state, 1))
@printf("assembled residual norm: %.3e\n", max(
    norm(assembled_components.right_residual),
    norm(assembled_components.left_residual),
))
println("unresolved: ", length(result.unresolved))
println("stats: ", result.stats)
@printf("time: %.3f s\n", seconds)
for (index, leaf) in pairs(result.leaves)
    modal = modal_leaves[index]
    @printf(
        "leaf %d x=[%.3f, %.3f] count=%d values=%s\n",
        index,
        real(leaf.chart.center) - leaf.chart.half_width,
        real(leaf.chart.center) + leaf.chart.half_width,
        leaf.count,
        repr(round.(sort(real.(modal.values)); digits=6)),
    )
end
