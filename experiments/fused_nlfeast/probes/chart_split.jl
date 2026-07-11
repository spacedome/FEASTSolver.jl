using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "..", "src", "FusedNLFEAST.jl"))
using .FusedNLFEAST

function scalar_problem(f, df; backward_scale=_ -> one(Float64))
    T = z -> fill(ComplexF64(f(z)), 1, 1)
    (
        f=f,
        df=df,
        T=T,
        right_solve=(z, rhs) -> rhs ./ f(z),
        left_solve=(z, rhs) -> rhs ./ conj(f(z)),
        residual_metric=(Tvalue, vector, side, value) -> abs(f(value)) / backward_scale(value),
    )
end

function child_charts(chart, depth; overlap=0.025, nodes=128)
    phase = pi / 11 + depth * pi / 17
    offset = chart.radius / sqrt(2)
    child_radius = (1 + overlap) * chart.radius / sqrt(2)
    [
        CircularChart(
            chart.center + offset * cis(phase + pi / 4 + quadrant * pi / 2),
            child_radius,
            nodes,
        )
        for quadrant in 0:3
    ]
end

function merge_candidates(candidates, parent; tolerance=1e-7)
    retained = filter(candidate -> FusedNLFEAST.in_chart(parent, candidate.value), candidates)
    sort!(retained; by=candidate -> (real(candidate.value), imag(candidate.value)))
    merged = NamedTuple[]
    for candidate in retained
        nearby = findfirst(item -> abs(item.value - candidate.value) <= tolerance, merged)
        if nearby === nothing
            push!(merged, candidate)
        elseif candidate.residual < merged[nearby].residual
            merged[nearby] = candidate
        end
    end
    merged
end

function split_solve(problem, parent; capacity=6, max_depth=7)
    candidates = NamedTuple[]
    stats = Dict(
        :visited => 0,
        :split => 0,
        :solved => 0,
        :empty => 0,
        :unresolved => 0,
    )

    function visit(chart, depth)
        stats[:visited] += 1
        estimate = argument_principle_count(
            z -> problem.df(z) / problem.f(z),
            chart;
            integrality_tolerance=1e-6,
            refinement_tolerance=1e-6,
        )
        if estimate.stable && estimate.count == 0
            stats[:empty] += 1
            return
        end
        should_split = !estimate.stable || estimate.count > capacity
        result = nothing
        if !should_split
            result = try
                fused_nlfeast(
                    problem.T,
                    problem.right_solve,
                    problem.left_solve,
                    chart,
                    ones(ComplexF64, 1, 1),
                    ones(ComplexF64, 1, 1);
                    config=FusedConfig(
                        moment_count=max(estimate.count, 1),
                        iterations=8,
                        ranktol=1e-11,
                        residual_ranktol=1e-14,
                        residual_tol=1e-10,
                        residual_metric=problem.residual_metric,
                        target_count=estimate.count,
                        coupling=:independent,
                        maxrank=max(estimate.count, 1),
                    ),
                )
            catch
                nothing
            end
            should_split = result === nothing || !result.converged
        end
        if !should_split
            stats[:solved] += 1
            for index in eachindex(result.extraction.values)
                push!(candidates, (
                    value=result.extraction.values[index],
                    residual=result.extraction.residuals[index],
                    depth=depth,
                ))
            end
            return
        end
        if depth >= max_depth
            stats[:unresolved] += 1
            return
        end
        stats[:split] += 1
        for child in child_charts(chart, depth)
            abs(child.center - parent.center) <= parent.radius + child.radius || continue
            visit(child, depth + 1)
        end
    end

    visit(parent, 0)
    merged = merge_candidates(candidates, parent)
    (candidates=candidates, merged=merged, stats=stats)
end

function report(label, problem, parent, expected)
    global_count = argument_principle_count(
        z -> problem.df(z) / problem.f(z),
        parent;
        refinement_factor=2,
        integrality_tolerance=1e-6,
        refinement_tolerance=1e-6,
    )
    result = split_solve(problem, parent)
    values = ComplexF64[item.value for item in result.merged]
    errors = if length(values) == length(expected)
        last(FusedNLFEAST.bottleneck_match(values, expected))
    else
        Float64[Inf]
    end
    @printf(
        "%s: count=%d reliable=%s recovered=%d max error=%.3e max residual=%.3e stats=%s\n",
        label,
        global_count.count,
        string(global_count.reliable),
        length(values),
        isempty(errors) ? Inf : maximum(errors),
        isempty(result.merged) ? Inf : maximum(item.residual for item in result.merged),
        repr(result.stats),
    )
end

large_sine = scalar_problem(sin, cos)
large_parent = CircularChart(0.0, 50.0, 512)
large_expected = ComplexF64[k * pi for k in -15:15]
report("31-root sine", large_sine, large_parent, large_expected)

alpha = 5.0
scale = z -> exp(alpha * z)
scaled_sine = scalar_problem(
    z -> scale(z) * sin(z),
    z -> scale(z) * (alpha * sin(z) + cos(z));
    backward_scale=z -> abs(scale(z)),
)
scaled_parent = CircularChart(0.0, 4.0, 128)
scaled_expected = ComplexF64[-pi, 0, pi]
report("analytically scaled sine", scaled_sine, scaled_parent, scaled_expected)

function vertical_children(chart::RectangularChart, depth; nodes=128)
    fraction = iseven(depth) ? 0.47 : 0.53
    left_edge = real(chart.center) - chart.half_width
    right_edge = real(chart.center) + chart.half_width
    split = left_edge + fraction * (right_edge - left_edge)
    left_width = (split - left_edge) / 2
    right_width = (right_edge - split) / 2
    (
        RectangularChart(
            complex((left_edge + split) / 2, imag(chart.center)),
            left_width,
            chart.half_height,
            nodes,
        ),
        RectangularChart(
            complex((split + right_edge) / 2, imag(chart.center)),
            right_width,
            chart.half_height,
            nodes,
        ),
    )
end

function rectangular_split_solve(problem, parent; capacity=6, max_depth=7)
    candidates = NamedTuple[]
    stats = Dict(:visited => 0, :split => 0, :solved => 0, :empty => 0, :unresolved => 0, :count_refinements => 0)

    function counted_chart(chart)
        current = chart
        estimate = nothing
        for refinement in 0:2
            estimate = determinant_winding_count(
                problem.T,
                current;
                integrality_tolerance=1e-6,
                refinement_tolerance=1e-6,
                phase_resolution_certified=true,
            )
            estimate.reliable && return estimate, current
            refinement == 2 && break
            stats[:count_refinements] += 1
            current = FusedNLFEAST.refine_chart(current, 2 * length(current.nodes))
        end
        estimate, current
    end

    function visit(input_chart, depth)
        stats[:visited] += 1
        estimate, chart = counted_chart(input_chart)
        if estimate.reliable && estimate.count == 0
            stats[:empty] += 1
            return
        end
        should_split = !estimate.reliable || estimate.count > capacity
        result = nothing
        if !should_split
            result = try
                fused_nlfeast(
                    problem.T,
                    problem.right_solve,
                    problem.left_solve,
                    chart,
                    ones(ComplexF64, 1, 1),
                    ones(ComplexF64, 1, 1);
                    config=FusedConfig(
                        moment_count=max(estimate.count, 1),
                        iterations=8,
                        ranktol=1e-11,
                        residual_ranktol=1e-14,
                        residual_tol=1e-10,
                        residual_metric=problem.residual_metric,
                        target_count=estimate.count,
                        coupling=:independent,
                        maxrank=max(estimate.count, 1),
                    ),
                )
            catch
                nothing
            end
            should_split = result === nothing || !result.converged
        end
        if !should_split
            stats[:solved] += 1
            for index in eachindex(result.extraction.values)
                push!(candidates, (
                    value=result.extraction.values[index],
                    residual=result.extraction.residuals[index],
                    depth=depth,
                ))
            end
            return
        end
        if depth >= max_depth
            stats[:unresolved] += 1
            return
        end
        stats[:split] += 1
        for child in vertical_children(input_chart, depth)
            visit(child, depth + 1)
        end
    end

    visit(parent, 0)
    merged = merge_candidates(candidates, parent)
    (candidates=candidates, merged=merged, stats=stats)
end

function rectangular_report(label, problem, parent, expected)
    result = rectangular_split_solve(problem, parent)
    values = ComplexF64[item.value for item in result.merged]
    errors = length(values) == length(expected) ?
        last(FusedNLFEAST.bottleneck_match(values, expected)) : Float64[Inf]
    @printf(
        "%s: recovered=%d max error=%.3e max residual=%.3e stats=%s\n",
        label,
        length(values),
        maximum(errors),
        isempty(result.merged) ? Inf : maximum(item.residual for item in result.merged),
        repr(result.stats),
    )
end

rectangular_report(
    "31-root sine, rectangular partition",
    large_sine,
    RectangularChart(0.0, 49.0, 10.0, 256),
    large_expected,
)
rectangular_report(
    "analytically scaled sine, rectangular partition",
    scaled_sine,
    RectangularChart(0.0, 4.0, 2.0, 128),
    scaled_expected,
)
