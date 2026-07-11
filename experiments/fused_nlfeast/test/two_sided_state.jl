@testset "common spectral restriction preserves both invariant pairs" begin
    operator = ComplexF64[0.2 3.0; 0.0 2.0]
    identity_map = Matrix{ComplexF64}(I, 2, 2)
    chart = CircularChart(0.0, 1.0, 16)
    restricted = common_spectral_restriction(
        chart,
        operator,
        identity_map,
        identity_map,
    )
    coefficients = (-operator, identity_map)
    right_residual = polynomial_invariant_residual(
        coefficients,
        restricted.right,
        restricted.state,
    )
    left_residual = polynomial_left_invariant_residual(
        coefficients,
        restricted.left,
        restricted.state,
    )
    @test size(restricted.state) == (1, 1)
    @test isapprox(restricted.state[1, 1], 0.2; atol=1e-12)
    @test norm(right_residual) <= 1e-12
    @test norm(left_residual) <= 1e-12
    @test restricted.condition >= 1
end

@testset "one cross Hankel realizes right and dual left maps" begin
    chart = CircularChart(0.0, 3.0, 16)
    physical_state = Diagonal(ComplexF64[0.6, 1.5])
    coordinate_state = physical_state ./ chart.radius
    right_probe = ComplexF64[1.0 0.3; -0.2 0.8]
    left_probe = ComplexF64[0.7 -0.4; 0.5 1.2]
    right_moments = Matrix{ComplexF64}[
        coordinate_state^k * right_probe for k in 0:3
    ]
    left_moments = Matrix{ComplexF64}[
        adjoint(coordinate_state)^k * left_probe for k in 0:3
    ]
    identity_map = Matrix{ComplexF64}(I, 2, 2)
    realization = two_sided_structured_realization(
        (-Matrix(physical_state), identity_map),
        (one, identity),
        chart,
        right_moments,
        left_moments,
        left_probe,
        2;
        ranktol=1e-12,
        fixed_rank=2,
        restrict_to_chart=false,
    )
    @test multiset_distance(eigvals(realization.state), diag(physical_state)) <= 1e-12
    @test norm(realization.right_residual) <= 1e-12
    @test norm(realization.left_residual) <= 1e-12
    @test realization.rank == 2
end
