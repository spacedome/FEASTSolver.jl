struct PartitionedStateResult
    leaves::Vector{Any}
    unresolved::Vector{Any}
    stats::NamedTuple
    root_count::Union{Nothing,Int}
    leaf_count::Int
    count_conserved::Bool
end

struct PartitionedInvariantState
    right_state::Matrix{ComplexF64}
    left_state::Matrix{ComplexF64}
    right::Matrix{ComplexF64}
    left::Matrix{ComplexF64}
    right_residual::Matrix{ComplexF64}
    left_residual::Matrix{ComplexF64}
    leaf_ranges::Vector{UnitRange{Int}}
end

state_components(state::PartitionedInvariantState) = (
    right_state=state.right_state,
    left_state=state.left_state,
    right=state.right,
    left=state.left,
    right_residual=state.right_residual,
    left_residual=state.left_residual,
)

function partitioned_invariant_state(result::PartitionedStateResult; require_certified=true)
    isempty(result.unresolved) || error("cannot assemble unresolved partition leaves")
    result.count_conserved || error("cannot assemble a partition without count conservation")
    isempty(result.leaves) && error("cannot assemble an empty partition state")
    components = Any[]
    ranges = UnitRange{Int}[]
    offset = 0
    physical_dimension = nothing
    for leaf in result.leaves
        if require_certified
            hasproperty(leaf.result, :certified) || error(
                "partition leaf result does not expose certification",
            )
            leaf.result.certified || error("partition leaf is not certified")
        end
        leaf_state = hasproperty(leaf.result, :state) ? leaf.result.state : leaf.result
        leaf_components = state_components(leaf_state)
        count = size(leaf_components.right_state, 1)
        count == leaf.count || throw(DimensionMismatch(
            "partition leaf count does not match its invariant-pair rank",
        ))
        size(leaf_components.right_state) == (count, count) || throw(DimensionMismatch(
            "partition right state must be square",
        ))
        size(leaf_components.left_state) == (count, count) || throw(DimensionMismatch(
            "partition left state must match the right state",
        ))
        all(value -> in_chart(leaf.chart, value), eigvals(leaf_components.right_state)) ||
            error("partition right state lies outside its leaf chart")
        all(value -> in_chart(leaf.chart, value), eigvals(leaf_components.left_state)) ||
            error("partition left state lies outside its leaf chart")
        dimension = size(leaf_components.right, 1)
        physical_dimension === nothing || dimension == physical_dimension ||
            throw(DimensionMismatch("partition leaves have different physical dimensions"))
        physical_dimension = dimension
        push!(components, leaf_components)
        push!(ranges, (offset + 1):(offset + count))
        offset += count
    end
    block_diagonal(field) = cat((getproperty(component, field) for component in components)...; dims=(1, 2))
    PartitionedInvariantState(
        Matrix{ComplexF64}(block_diagonal(:right_state)),
        Matrix{ComplexF64}(block_diagonal(:left_state)),
        Matrix{ComplexF64}(reduce(hcat, (component.right for component in components))),
        Matrix{ComplexF64}(reduce(hcat, (component.left for component in components))),
        Matrix{ComplexF64}(reduce(hcat, (component.right_residual for component in components))),
        Matrix{ComplexF64}(reduce(hcat, (component.left_residual for component in components))),
        ranges,
    )
end

function bisect_chart(
    chart::RectangularChart,
    depth;
    node_count=length(chart.nodes),
    fraction=nothing,
)
    split_fraction = something(fraction, iseven(depth) ? 0.47 : 0.53)
    0 < split_fraction < 1 || throw(ArgumentError("split fraction must lie in (0, 1)"))
    if chart.half_width >= chart.half_height
        low = real(chart.center) - chart.half_width
        high = real(chart.center) + chart.half_width
        split = low + split_fraction * (high - low)
        first_half = (split - low) / 2
        second_half = (high - split) / 2
        return (
            RectangularChart(
                complex((low + split) / 2, imag(chart.center)),
                first_half,
                chart.half_height,
                node_count,
            ),
            RectangularChart(
                complex((split + high) / 2, imag(chart.center)),
                second_half,
                chart.half_height,
                node_count,
            ),
        )
    end
    low = imag(chart.center) - chart.half_height
    high = imag(chart.center) + chart.half_height
    split = low + split_fraction * (high - low)
    first_half = (split - low) / 2
    second_half = (high - split) / 2
    (
        RectangularChart(
            complex(real(chart.center), (low + split) / 2),
            chart.half_width,
            first_half,
            node_count,
        ),
        RectangularChart(
            complex(real(chart.center), (split + high) / 2),
            chart.half_width,
            second_half,
            node_count,
        ),
    )
end

function partitioned_state_solve(
    parent::RectangularChart;
    count_backend,
    solve_leaf,
    accept_leaf,
    capacity,
    max_depth=8,
    max_count_refinements=2,
    max_leaf_refinements=1,
    split_fractions=(0.5, 0.47, 0.53, 0.43, 0.57, 0.37, 0.63),
)
    capacity > 0 || throw(ArgumentError("partition capacity must be positive"))
    max_depth >= 0 || throw(ArgumentError("max_depth must be nonnegative"))
    max_count_refinements >= 0 || throw(ArgumentError(
        "max_count_refinements must be nonnegative",
    ))
    max_leaf_refinements >= 0 || throw(ArgumentError(
        "max_leaf_refinements must be nonnegative",
    ))
    isempty(split_fractions) && throw(ArgumentError("split_fractions must not be empty"))
    all(fraction -> 0 < fraction < 1, split_fractions) || throw(ArgumentError(
        "every split fraction must lie in (0, 1)",
    ))
    leaves = Any[]
    unresolved = Any[]
    visited = 0
    split_count = 0
    empty_count = 0
    count_refinements = 0
    solve_failures = 0
    leaf_refinements = 0
    split_attempts = 0
    split_rejections = 0

    function count_chart(input_chart)
        chart = input_chart
        estimate = count_backend(chart)
        for _ in 1:max_count_refinements
            estimate.reliable && break
            count_refinements += 1
            chart = refine_chart(chart, 2 * length(chart.nodes))
            estimate = count_backend(chart)
        end
        estimate, chart
    end

    root_estimate, root_chart = count_chart(parent)
    root_count = root_estimate.reliable ? root_estimate.count : nothing

    function additive_children(chart, depth, estimate)
        for fraction in split_fractions
            split_attempts += 1
            raw_children = bisect_chart(chart, depth; fraction=fraction)
            first_estimate, first_chart = count_chart(raw_children[1])
            second_estimate, second_chart = count_chart(raw_children[2])
            if first_estimate.reliable && second_estimate.reliable &&
               first_estimate.count + second_estimate.count == estimate.count
                return (
                    (chart=first_chart, estimate=first_estimate),
                    (chart=second_chart, estimate=second_estimate),
                )
            end
            split_rejections += 1
        end
        nothing
    end

    function visit(input_chart, depth; known_estimate=nothing)
        visited += 1
        estimate, chart = known_estimate === nothing ?
            count_chart(input_chart) : (known_estimate, input_chart)
        if estimate.reliable && estimate.count == 0
            empty_count += 1
            return
        end
        should_split = !estimate.reliable || estimate.count > capacity
        leaf_result = nothing
        failure = nothing
        if !should_split
            accepted = false
            for refinement in 0:max_leaf_refinements
                leaf_result = try
                    solve_leaf(chart, estimate)
                catch error
                    failure = error
                    nothing
                end
                if leaf_result !== nothing && accept_leaf(leaf_result, estimate)
                    accepted = true
                    break
                end
                solve_failures += 1
                refinement == max_leaf_refinements && break
                leaf_refinements += 1
                refined_chart = refine_chart(chart, 2 * length(chart.nodes))
                refined_estimate, chart = count_chart(refined_chart)
                if !refined_estimate.reliable || refined_estimate.count != estimate.count
                    estimate = refined_estimate
                    should_split = true
                    break
                end
                estimate = refined_estimate
            end
            should_split = should_split || !accepted
        end
        if !should_split
            push!(leaves, (chart=chart, count=estimate.count, result=leaf_result))
            return
        end
        if depth >= max_depth
            reason = !estimate.reliable ? :uncertified_count :
                estimate.count > capacity ? :capacity : :leaf_failure
            push!(unresolved, (
                chart=chart,
                depth=depth,
                estimate=estimate,
                reason=reason,
                error=failure,
            ))
            return
        end
        split_count += 1
        if estimate.reliable
            children = additive_children(chart, depth, estimate)
            if children === nothing
                push!(unresolved, (
                    chart=chart,
                    depth=depth,
                    estimate=estimate,
                    reason=:inconsistent_child_counts,
                    error=failure,
                ))
                return
            end
            for child in children
                visit(child.chart, depth + 1; known_estimate=child.estimate)
            end
        else
            for child in bisect_chart(chart, depth)
                visit(child, depth + 1)
            end
        end
    end

    visit(root_chart, 0; known_estimate=root_estimate)
    leaf_count = sum(leaf.count for leaf in leaves)
    count_conserved = root_count !== nothing && isempty(unresolved) && leaf_count == root_count
    PartitionedStateResult(
        leaves,
        unresolved,
        (
            visited=visited,
            split=split_count,
            empty=empty_count,
            count_refinements=count_refinements,
            solve_failures=solve_failures,
            leaf_refinements=leaf_refinements,
            split_attempts=split_attempts,
            split_rejections=split_rejections,
        ),
        root_count,
        leaf_count,
        count_conserved,
    )
end
