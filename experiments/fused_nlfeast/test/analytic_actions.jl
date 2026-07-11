@testset "Cauchy actions recover invariant residuals and divided overlap" begin
    coefficients = (
        ComplexF64[2.0 0.3; -0.1 1.0],
        ComplexF64[-1.0 0.2; 0.4 -0.7],
        ComplexF64[0.3 -0.2; 0.1 0.5],
    )
    right_state = ComplexF64[0.2 0.4; 0.0 -0.3]
    left_state = ComplexF64[-0.1 -0.2; 0.0 0.35]
    right = ComplexF64[1.0 0.2; -0.3 0.8]
    left = ComplexF64[0.7 -0.4; 0.1 1.1]
    chart = CircularChart(0.0, 2.0, 64)
    T = z -> coefficients[1] + z .* coefficients[2] + z^2 .* coefficients[3]
    data = cauchy_invariant_data(
        chart,
        right_state,
        right,
        left_state,
        left,
        (z, block) -> T(z) * block,
        (z, block) -> adjoint(T(z)) * block,
    )
    @test isapprox(
        data.right_residual,
        polynomial_invariant_residual(coefficients, right, right_state);
        rtol=1e-12,
        atol=1e-12,
    )
    @test isapprox(
        data.left_residual,
        polynomial_left_invariant_residual(coefficients, left, left_state);
        rtol=1e-12,
        atol=1e-12,
    )
    @test isapprox(
        data.overlap,
        polynomial_state_divided_overlap(
            coefficients,
            left_state,
            left,
            right_state,
            right,
        );
        rtol=1e-12,
        atol=1e-12,
    )
end

@testset "independent action quadrature exposes circular aliasing" begin
    chart = CircularChart(0.0, 1.0, 8)
    refined_chart = FusedNLFEAST.refine_chart(chart, 16)
    state = zeros(ComplexF64, 1, 1)
    vector = ones(ComplexF64, 1, 1)
    action = (z, block) -> (1 + z^8) .* block

    aliased = cauchy_right_invariant_residual(chart, state, vector, action)
    resolved = cauchy_right_invariant_residual(refined_chart, state, vector, action)
    @test norm(aliased) <= 1e-12
    @test resolved ≈ ones(ComplexF64, 1, 1)
end
