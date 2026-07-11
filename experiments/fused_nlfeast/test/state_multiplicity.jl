@testset "modal diagonalization loses a multiple state" begin
    coefficients = [zeros(ComplexF64, 1, 1), zeros(ComplexF64, 1, 1), ones(ComplexF64, 1, 1)]
    state = ComplexF64[0 1; 0 0]
    output = ComplexF64[1 0]
    @test polynomial_invariant_residual(coefficients, output, state) == zeros(ComplexF64, 1, 2)
    @test rank(vcat(output, output * state); atol=1e-14) == 2

    state_moments = Matrix{ComplexF64}[output * state^k for k in 0:3]
    state_H0, _ = FusedNLFEAST.block_hankel(state_moments, 2; observer=ones(ComplexF64, 1, 1))
    @test rank(state_H0; atol=1e-14) == 2

    modal_moments = [
        ComplexF64[1 1],
        zeros(ComplexF64, 1, 2),
        zeros(ComplexF64, 1, 2),
        zeros(ComplexF64, 1, 2),
    ]
    modal_H0, _ = FusedNLFEAST.block_hankel(modal_moments, 2; observer=ones(ComplexF64, 1, 1))
    @test rank(modal_H0; atol=1e-14) == 1
end

@testset "multiple state survives cache re-extraction" begin
    coefficients = [zeros(ComplexF64, 1, 1), zeros(ComplexF64, 1, 1), ones(ComplexF64, 1, 1)]
    T = z -> fill(ComplexF64(z^2), 1, 1)
    chart = CircularChart(0.0 + 0.0im, 1.0, 8)
    cache = ContourSampleCache(chart, (z, B) -> B ./ z^2, (z, B) -> B ./ conj(z)^2)
    right_initial = add_right_probe!(cache, :initial, ones(ComplexF64, 1, 1))
    left_initial = add_left_probe!(cache, :initial, ones(ComplexF64, 1, 1))
    right_moments = probe_moments(cache, :initial, :right, 4)
    left_moments = probe_moments(cache, :initial, :left, 4)
    common = common_polynomial_realization(
        coefficients,
        chart,
        right_moments,
        left_moments,
        left_initial.probe,
        right_initial.probe,
        2;
        ranktol=1e-12,
        maxrank=2,
    )
    @test common.rank == 2
    @test istriu(common.state)
    @test maximum(abs, eigvals(common.state)) <= 1e-12
    @test norm(common.right_residual) <= 1e-12
    @test norm(common.left_residual) <= 1e-12
    @test rank(vcat(common.right, common.right * common.state); atol=1e-12) == 2

    @test_throws ArgumentError moment_tangent(
        ComplexF64[0, 0],
        2,
        1;
        cluster_tolerance=1e-8,
        side=:right,
    )
    right_tangent = state_moment_tangent(common.state, 2, 1; side=:right, ranktol=1e-12)
    left_tangent = state_moment_tangent(common.state, 2, 1; side=:left, ranktol=1e-12)
    @test size(right_tangent) == (2, 1)
    @test size(left_tangent) == (2, 1)

    factors = state_residual_factors(
        common.right_residual,
        common.left_residual;
        ranktol=1e-12,
    )
    add_right_probe!(cache, :right_residual, factors.right_basis; role=:residual)
    add_left_probe!(cache, :left_residual, factors.left_basis; role=:residual)
    corrected_right, corrected_left = state_corrected_moments(
        cache,
        common.state,
        common.state,
        common.right,
        common.left,
        :right_residual,
        :left_residual,
        factors.right_coefficients,
        factors.left_coefficients,
        4,
        right_tangent=right_tangent,
        left_tangent=left_tangent,
    )
    updated = common_polynomial_realization(
        coefficients,
        chart,
        corrected_right,
        corrected_left,
        left_initial.probe,
        right_initial.probe,
        2;
        ranktol=1e-12,
        maxrank=2,
    )
    @test updated.rank == 2
    @test size(corrected_right[1], 2) == 1
    @test size(corrected_left[1], 2) == 1
    @test maximum(abs, eigvals(updated.state)) <= 1e-7
    @test norm(updated.right_residual) <= 1e-11
    @test norm(updated.left_residual) <= 1e-11
    @test rank(vcat(updated.right, updated.right * updated.state); atol=1e-11) == 2
    @test all(norm(T(value) * ones(ComplexF64, 1)) <= 1e-15 for value in eigvals(updated.state))
end
