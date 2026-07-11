@testset "disjoint partition conserves a certified count" begin
    roots = ComplexF64[-1.2, 0.1, 1.4]
    T = z -> Diagonal(z .- roots)
    count_backend = chart -> determinant_winding_count(
        T,
        chart;
        phase_resolution_certified=true,
    )
    result = partitioned_state_solve(
        RectangularChart(0.0, 2.0, 1.0, 64);
        count_backend=count_backend,
        solve_leaf=(chart, estimate) -> estimate.count,
        accept_leaf=(leaf, estimate) -> leaf == estimate.count,
        capacity=1,
        max_depth=6,
    )
    @test isempty(result.unresolved)
    @test result.root_count == 3
    @test result.leaf_count == 3
    @test result.count_conserved
    @test all(leaf.count == 1 for leaf in result.leaves)
end

@testset "leaf quadrature refines before chart subdivision" begin
    root = 0.2 + 0.0im
    function exact_count(chart)
        count = FusedNLFEAST.in_chart(chart, root) ? 1 : 0
        ArgumentCountEstimate(
            count,
            ComplexF64(count),
            ComplexF64(count),
            0.0,
            0.0,
            true,
            true,
            true,
            length(chart.nodes),
            length(chart.nodes),
        )
    end
    result = partitioned_state_solve(
        RectangularChart(0.0, 1.0, 1.0, 32);
        count_backend=exact_count,
        solve_leaf=(chart, estimate) -> length(chart.nodes),
        accept_leaf=(nodes, estimate) -> nodes >= 64,
        capacity=1,
        max_depth=2,
        max_leaf_refinements=1,
    )
    @test result.count_conserved
    @test length(result.leaves) == 1
    @test length(result.leaves[1].chart.nodes) == 64
    @test result.stats.leaf_refinements == 1
    @test result.stats.split == 0
end

@testset "disjoint leaves assemble one block invariant pair" begin
    function scalar_common_state(value)
        state = fill(ComplexF64(value), 1, 1)
        vector = ones(ComplexF64, 2, 1)
        residual = zeros(ComplexF64, 2, 1)
        CommonStateRealization(
            state,
            vector,
            vector,
            residual,
            residual,
            ones(ComplexF64, 1, 1),
            state,
            [1.0],
            [1.0],
            [1.0],
            1,
        )
    end
    first_state = scalar_common_state(-0.5)
    second_state = scalar_common_state(0.7)
    leaves = Any[
        (
            chart=RectangularChart(-0.5, 0.2, 0.2, 16),
            count=1,
            result=(state=first_state, certified=true),
        ),
        (
            chart=RectangularChart(0.7, 0.2, 0.2, 16),
            count=1,
            result=(state=second_state, certified=true),
        ),
    ]
    partition = PartitionedStateResult(leaves, Any[], (;), 2, 2, true)
    assembled = partitioned_invariant_state(partition)
    components = state_components(assembled)

    @test eigvals(components.right_state) ≈ ComplexF64[-0.5, 0.7]
    @test size(components.right) == (2, 2)
    @test components.right_residual == zeros(ComplexF64, 2, 2)
    @test assembled.leaf_ranges == [1:1, 2:2]
end

@testset "partition shifts a cut away from boundary spectrum" begin
    roots = ComplexF64[-0.8, 0.0, 0.7]
    function exact_count(chart)
        boundary = any(abs(chart_boundary_margin(chart, root)) <= 1e-12 for root in roots)
        count = Base.count(root -> FusedNLFEAST.in_chart(chart, root), roots)
        reliable = !boundary
        ArgumentCountEstimate(
            count,
            ComplexF64(count),
            ComplexF64(count),
            0.0,
            0.0,
            reliable,
            reliable,
            reliable,
            length(chart.nodes),
            length(chart.nodes),
        )
    end
    result = partitioned_state_solve(
        RectangularChart(0.0, 1.0, 0.5, 32);
        count_backend=exact_count,
        solve_leaf=(chart, estimate) -> estimate.count,
        accept_leaf=(leaf, estimate) -> leaf == estimate.count,
        capacity=1,
        max_depth=6,
        max_count_refinements=0,
    )
    @test isempty(result.unresolved)
    @test result.count_conserved
    @test result.leaf_count == 3
    @test result.stats.split_rejections >= 1
end
