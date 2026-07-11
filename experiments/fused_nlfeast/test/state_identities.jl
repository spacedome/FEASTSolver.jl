@testset "state realization fixed point" begin
    rng = MersenneTwister(721)
    chart = CircularChart(0.2 - 0.1im, 1.7, 12)
    state = ComplexF64[0.3 0.4; 0 0.3]
    physical_state = chart.center .* Matrix{ComplexF64}(I, 2, 2) .+ chart.radius .* state
    output = rand(rng, ComplexF64, 3, 2)
    expected_scale = inv(Matrix{ComplexF64}(I, 2, 2) + state^length(chart.nodes))
    moments = [zeros(ComplexF64, size(output)) for _ in 1:8]
    for (z, weight) in zip(chart.nodes, chart.weights)
        coordinate = FusedNLFEAST.chart_coordinate(chart, z)
        sample = output / (z .* Matrix{ComplexF64}(I, 2, 2) .- physical_state)
        power = one(ComplexF64)
        for moment in moments
            moment .+= (weight * power) .* sample
            power *= coordinate
        end
    end
    @test all(
        isapprox(moments[k + 1], output * state^k * expected_scale; rtol=2e-12, atol=2e-12)
        for k in 0:7
    )

    linear_A = rand(rng, ComplexF64, 3, 3)
    right = rand(rng, ComplexF64, 3, 2)
    residual = right * physical_state - linear_A * right
    for z in chart.nodes
        corrected = (right - (z .* I - linear_A) \ residual) /
            (z .* Matrix{ComplexF64}(I, 2, 2) .- physical_state)
        @test isapprox(corrected, (z .* I - linear_A) \ right; rtol=2e-12, atol=2e-12)
    end
end

@testset "lift normalization makes common overlap gauge independent" begin
    values = ComplexF64[0.2, 0.7]
    state_matrix = Diagonal(values) |> Matrix
    coefficient_identity = Matrix{ComplexF64}(I, 2, 2)
    coefficients = (-state_matrix, coefficient_identity)
    right_gauge = Diagonal(ComplexF64[1e12, 1e-12])
    left_gauge = Diagonal(ComplexF64[1e12, 1e-12])
    right = Matrix{ComplexF64}(right_gauge)
    left = Matrix{ComplexF64}(left_gauge)
    residual = zeros(ComplexF64, 2, 2)
    independent = IndependentStateRealization(
        state_matrix,
        state_matrix,
        right,
        left,
        residual,
        residual,
        [1.0, 1.0],
        [1.0, 1.0],
        2,
        2,
        2,
    )
    overlap = (left_state, left_map, right_state, right_map) ->
        polynomial_state_divided_overlap(
            coefficients,
            left_state,
            left_map,
            right_state,
            right_map,
        )
    right_residual = (map, state) -> polynomial_invariant_residual(
        coefficients,
        map,
        state,
    )
    left_residual = (map, state) -> polynomial_left_invariant_residual(
        coefficients,
        map,
        state,
    )

    @test_throws ErrorException common_state_from_independent(
        independent;
        overlap_ranktol=1e-10,
        right_residual=right_residual,
        left_residual=left_residual,
        state_overlap=overlap,
    )
    normalized = lift_normalized_independent_state(independent, 1)
    common = common_state_from_independent(
        normalized;
        overlap_ranktol=1e-10,
        right_residual=right_residual,
        left_residual=left_residual,
        state_overlap=overlap,
    )
    @test common.overlap_singular_values[end] /
        common.overlap_singular_values[1] >= 1 - 1e-12
    @test norm(common.right_residual) <= 1e-12
    @test norm(common.left_residual) <= 1e-12
end

@testset "contour separation detects a nonnormal state resolvent" begin
    state_matrix = ComplexF64[0 1e8; 0 0]
    vectors = Matrix{ComplexF64}(I, 2, 2)
    residuals = zeros(ComplexF64, 2, 2)
    state = CommonStateRealization(
        state_matrix,
        vectors,
        vectors,
        residuals,
        residuals,
        Matrix{ComplexF64}(I, 2, 2),
        state_matrix,
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        2,
    )
    chart = CircularChart(0.0, 1.0, 16)
    separation = state_contour_separation(chart, state)

    @test minimum(chart_boundary_margin.(Ref(chart), eigvals(state_matrix))) ≈ 1
    @test separation.minimum <= 1.1e-8
end

@testset "structured modal condition is gauge invariant" begin
    coefficients = (
        ComplexF64[-1 40; 0 -2],
        Matrix{ComplexF64}(I, 2, 2),
    )
    values = ComplexF64[1, 2]
    right = ComplexF64[1 40; 0 1]
    left = ComplexF64[1 0; 40 1]
    baseline = structured_eigenvalue_condition_numbers(
        coefficients,
        (_ -> 0.0, _ -> 1.0),
        values,
        right,
        left,
    )
    right_scaling = Diagonal(ComplexF64[3 - 2im, -0.4im])
    left_scaling = Diagonal(ComplexF64[-2im, 5 + im])
    scaled = structured_eigenvalue_condition_numbers(
        coefficients,
        (_ -> 0.0, _ -> 1.0),
        values,
        right * right_scaling,
        left * left_scaling,
    )
    @test baseline ≈ scaled
    @test minimum(baseline) > 30
end

@testset "lift-normalized backward error is similarity invariant" begin
    coefficients = (
        ComplexF64[1.0 0.2; -0.3 0.7],
        ComplexF64[-0.8 0.1; 0.4 -1.1],
        ComplexF64[0.2 -0.1; 0.3 0.5],
    )
    functions = (one, identity, value -> value^2)
    right_state = ComplexF64[0.2 0.4; 0.0 -0.3]
    left_state = ComplexF64[-0.1 -0.2; 0.0 0.35]
    right = ComplexF64[1.0 0.2; -0.3 0.8]
    left = ComplexF64[0.7 -0.4; 0.1 1.1]
    right_gauge = ComplexF64[1.2 0.3; -0.2 0.9]
    left_gauge = ComplexF64[0.8 -0.4; 0.1 1.3]
    baseline = structured_pair_backward_error(
        coefficients,
        functions,
        right_state,
        right,
        left_state,
        left;
        lift_depth=2,
    )
    transformed = structured_pair_backward_error(
        coefficients,
        functions,
        right_gauge \ right_state * right_gauge,
        right * right_gauge,
        adjoint(left_gauge) * left_state / adjoint(left_gauge),
        left * left_gauge;
        lift_depth=2,
    )
    @test isapprox(transformed.right, baseline.right; rtol=1e-12)
    @test isapprox(transformed.left, baseline.left; rtol=1e-12)
    @test isapprox(transformed.maximum, baseline.maximum; rtol=1e-12)

    degenerate = structured_pair_backward_error(
        (Matrix{ComplexF64}(I, 1, 1),),
        (sin,),
        zeros(ComplexF64, 1, 1),
        ones(ComplexF64, 1, 1),
        zeros(ComplexF64, 1, 1),
        ones(ComplexF64, 1, 1);
        lift_depth=1,
    )
    @test degenerate.scale_degenerate
    @test isinf(degenerate.maximum)

    baseline_residual = lifted_residual_error(
        right_state,
        right,
        structured_invariant_residual(coefficients, functions, right, right_state),
        left_state,
        left,
        structured_left_invariant_residual(coefficients, functions, left, left_state);
        lift_depth=2,
    )
    transformed_residual = lifted_residual_error(
        right_gauge \ right_state * right_gauge,
        right * right_gauge,
        structured_invariant_residual(
            coefficients,
            functions,
            right * right_gauge,
            right_gauge \ right_state * right_gauge,
        ),
        adjoint(left_gauge) * left_state / adjoint(left_gauge),
        left * left_gauge,
        structured_left_invariant_residual(
            coefficients,
            functions,
            left * left_gauge,
            adjoint(left_gauge) * left_state / adjoint(left_gauge),
        );
        lift_depth=2,
    )
    @test isapprox(transformed_residual.maximum, baseline_residual.maximum; rtol=1e-12)

    baseline_components = lift_normalized_components(
        right_state,
        right,
        structured_invariant_residual(coefficients, functions, right, right_state),
        left_state,
        left,
        structured_left_invariant_residual(coefficients, functions, left, left_state);
        lift_depth=2,
    )
    transformed_components = lift_normalized_components(
        right_gauge \ right_state * right_gauge,
        right * right_gauge,
        structured_invariant_residual(
            coefficients,
            functions,
            right * right_gauge,
            right_gauge \ right_state * right_gauge,
        ),
        adjoint(left_gauge) * left_state / adjoint(left_gauge),
        left * left_gauge,
        structured_left_invariant_residual(
            coefficients,
            functions,
            left * left_gauge,
            adjoint(left_gauge) * left_state / adjoint(left_gauge),
        );
        lift_depth=2,
    )
    @test isapprox(
        svdvals(transformed_components.right_residual),
        svdvals(baseline_components.right_residual);
        rtol=1e-12,
    )
    @test isapprox(
        svdvals(transformed_components.left_residual),
        svdvals(baseline_components.left_residual);
        rtol=1e-12,
    )
    baseline_gauges = lifted_pair_gauges(right_state, right, left_state, left, 2)
    shifted_gauges = lifted_pair_gauges(
        1e5 .* Matrix{ComplexF64}(I, 2, 2) .+ 300 .* right_state,
        right,
        1e5 .* Matrix{ComplexF64}(I, 2, 2) .+ 300 .* left_state,
        left,
        2;
        center=1e5,
        radius=300,
    )
    @test isapprox(shifted_gauges.right, baseline_gauges.right; rtol=1e-12)
    @test isapprox(shifted_gauges.left, baseline_gauges.left; rtol=1e-12)
end

@testset "left Schur restriction takes a trailing invariant block" begin
    operator = ComplexF64[0.2 3.0; 0.0 2.0]
    chart = CircularChart(0.0, 1.0, 16)
    realization = FusedNLFEAST.MomentRealization(
        Matrix{ComplexF64}(I, 2, 2),
        adjoint(operator),
        [1.0, 0.5],
        2,
    )
    selected = FusedNLFEAST.interior_state_realization(
        chart,
        realization;
        adjoint_coordinate_state=true,
    )
    coefficients = (-operator, Matrix{ComplexF64}(I, 2, 2))
    residual = polynomial_left_invariant_residual(
        coefficients,
        selected.output,
        selected.state,
    )
    @test selected.count == 1
    @test isapprox(selected.values[1], 0.2; atol=1e-12)
    @test norm(residual) <= 1e-12
end

@testset "state divided overlap identity" begin
    rng = MersenneTwister(722)
    coefficients = [rand(rng, ComplexF64, 4, 4) for _ in 1:4]
    right = rand(rng, ComplexF64, 4, 3)
    left = rand(rng, ComplexF64, 4, 3)
    right_state = rand(rng, ComplexF64, 3, 3)
    left_state = rand(rng, ComplexF64, 3, 3)
    right_residual = polynomial_invariant_residual(coefficients, right, right_state)
    left_residual = polynomial_left_invariant_residual(coefficients, left, left_state)
    gram = polynomial_state_divided_overlap(
        coefficients,
        left_state,
        left,
        right_state,
        right,
    )
    numerator = adjoint(left_residual) * right - adjoint(left) * right_residual
    @test isapprox(left_state * gram - gram * right_state, numerator; rtol=2e-12, atol=2e-12)
    from_right = gram * right_state - adjoint(left) * right_residual
    from_left = left_state * gram - adjoint(left_residual) * right
    @test isapprox(from_right, from_left; rtol=2e-12, atol=2e-12)

    common = balanced_common_state(gram, (from_right + from_left) / 2, right, left; ranktol=1e-12)
    @test isapprox(
        adjoint(common.left_gauge) * gram * common.right_gauge,
        Matrix{ComplexF64}(I, 3, 3);
        rtol=2e-12,
        atol=2e-12,
    )
end

@testset "structured holomorphic state overlap" begin
    rng = MersenneTwister(723)
    coefficients = [rand(rng, ComplexF64, 3, 3), rand(rng, ComplexF64, 3, 3)]
    functions = (sin, exp)
    right = rand(rng, ComplexF64, 3, 2)
    left = rand(rng, ComplexF64, 3, 2)
    right_state = 0.2 .* rand(rng, ComplexF64, 2, 2)
    left_state = 0.2 .* rand(rng, ComplexF64, 2, 2)
    right_residual = structured_invariant_residual(coefficients, functions, right, right_state)
    left_residual = structured_left_invariant_residual(coefficients, functions, left, left_state)
    gram = structured_state_divided_overlap(
        coefficients,
        functions,
        left_state,
        left,
        right_state,
        right,
    )
    numerator = adjoint(left_residual) * right - adjoint(left) * right_residual
    @test isapprox(left_state * gram - gram * right_state, numerator; rtol=2e-11, atol=2e-11)

    direction = rand(rng, ComplexF64, 2, 2)
    sine_divided = matrix_divided_action(sin, left_state, direction, right_state)
    @test isapprox(
        left_state * sine_divided - sine_divided * right_state,
        sin(left_state) * direction - direction * sin(right_state);
        rtol=2e-11,
        atol=2e-11,
    )
end
