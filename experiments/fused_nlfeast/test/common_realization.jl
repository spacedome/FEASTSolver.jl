@testset "common corrected-moment realization" begin
    lambda_expected = ComplexF64[-0.45 + 0.18im, 0.62 - 0.21im]
    left_factor = ComplexF64[0 0.65-0.2im; 0 0]
    right_factor = ComplexF64[0 0; -0.45+0.3im 0]
    identity_matrix = Matrix{ComplexF64}(I, 2, 2)
    L = z -> identity_matrix + z .* left_factor
    R = z -> identity_matrix + z .* right_factor
    T = z -> L(z) * Diagonal(z .- lambda_expected) * R(z)
    right_solve = (z, B) -> T(z) \ B
    left_solve = (z, B) -> adjoint(T(z)) \ B

    chart = CircularChart(0.0 + 0.0im, 1.2, 128)
    rng = MersenneTwister(20260709)
    right_probe = randn(rng, ComplexF64, 2, 2)
    left_probe = randn(rng, ComplexF64, 2, 2)
    cache = ContourSampleCache(chart, right_solve, left_solve)
    add_right_probe!(cache, :initial, right_probe; role=:realization)
    add_left_probe!(cache, :initial, left_probe; role=:realization)

    initial = probe_moment_data(cache, :initial, :initial, 6)
    initial_cross_error = maximum(
        norm(left_probe' * initial.right[k] - initial.left[k]' * right_probe)
        for k in eachindex(initial.right)
    )
    @test initial_cross_error <= 1e-12

    values = lambda_expected + ComplexF64[0.13 - 0.07im, -0.11 + 0.09im]
    right_exact = hcat([R(value) \ identity_matrix[:, j] for (j, value) in enumerate(lambda_expected)]...)
    left_exact = hcat([adjoint(L(value)) \ identity_matrix[:, j] for (j, value) in enumerate(lambda_expected)]...)
    right = right_exact + 0.12 .* randn(rng, ComplexF64, 2, 2)
    left = left_exact + 0.10 .* randn(rng, ComplexF64, 2, 2)
    factors = residual_factors(T, values, right, left; ranktol=0.0)
    add_right_probe!(cache, :right_residual, factors.right_basis; role=:residual)
    add_left_probe!(cache, :left_residual, factors.left_basis; role=:residual)
    moments = mixed_corrected_moments(
        T,
        cache,
        values,
        right,
        left,
        :right_residual,
        :left_residual,
        factors.right_coefficients,
        factors.left_coefficients,
        6,
    )

    naive_error = maximum(
        norm(left' * moments.right[k] - moments.left[k]' * right)
        for k in eachindex(moments.right)
    )
    naive_scale = maximum(
        max(norm(left' * moments.right[k]), norm(moments.left[k]' * right))
        for k in eachindex(moments.right)
    )
    @test naive_error / naive_scale >= 1e-2

    H0, H1 = FusedNLFEAST.block_hankel(moments.mixed, 3)
    decomposition = svd(H0)
    @test decomposition.S[3] / decomposition.S[2] <= 1e-12
    U = decomposition.U[:, 1:2]
    V = decomposition.V[:, 1:2]
    inverse_singulars = Diagonal(1.0 ./ decomposition.S[1:2])
    state = U' * H1 * V * inverse_singulars
    right_output = reduce(hcat, moments.right[1:3]) * V * inverse_singulars
    left_output = reduce(hcat, moments.left[1:3]) * U
    state_eigen = eigen(state)
    left_state_vectors = inv(state_eigen.vectors)'
    extracted_values = chart.center .+ chart.radius .* state_eigen.values
    extracted_right = right_output * state_eigen.vectors
    extracted_left = left_output * left_state_vectors

    order = sortperm(extracted_values; by=value -> (real(value), imag(value)))
    expected_order = sortperm(lambda_expected; by=value -> (real(value), imag(value)))
    @test maximum(abs.(extracted_values[order] - lambda_expected[expected_order])) <= 1e-12
    @test maximum(
        norm(T(extracted_values[j]) * extracted_right[:, j]) / norm(extracted_right[:, j])
        for j in eachindex(extracted_values)
    ) <= 1e-12
    @test maximum(
        norm(adjoint(T(extracted_values[j])) * extracted_left[:, j]) / norm(extracted_left[:, j])
        for j in eachindex(extracted_values)
    ) <= 1e-12
    @test norm(T(0.2) * T(-0.7im) - T(-0.7im) * T(0.2)) >= 1e-2
end
