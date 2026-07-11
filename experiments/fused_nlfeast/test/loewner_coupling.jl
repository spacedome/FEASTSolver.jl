@testset "divided-difference Loewner coupling" begin
    rng = MersenneTwister(1401)
    n = 5
    count = 3
    B = Matrix{ComplexF64}(I, n, n) + 0.15 .* randn(rng, ComplexF64, n, n)
    A = randn(rng, ComplexF64, n, n)
    T = z -> z .* B .- A
    divided_difference = (eta, theta) -> copy(B)
    right_values = ComplexF64[-0.8 + 0.1im, 0.2 - 0.3im, 0.9 + 0.2im]
    left_values = ComplexF64[-0.7 - 0.2im, 0.1 + 0.25im, 1.1 - 0.1im]
    right = randn(rng, ComplexF64, n, count)
    left = randn(rng, ComplexF64, n, count)
    data = modal_loewner_pencil(
        T,
        divided_difference,
        right_values,
        right,
        left_values,
        left,
    )

    @test norm(data.gram - left' * B * right) <= 1e-12
    @test norm(data.action - left' * A * right) <= 1e-12
    @test data.pencil_consistency_defect <= 1e-12
    structured_data = modal_loewner_pencil(
        T,
        (eta, beta) -> error("operator-valued divided difference should not be called"),
        right_values,
        right,
        left_values,
        left;
        divided_overlap=(alphas, Y, betas, X) -> adjoint(Y) * B * X,
    )
    @test norm(structured_data.gram - data.gram) <= 1e-12
    completed = residual_completed_overlap(
        T,
        divided_difference,
        right_values,
        right,
        left_values,
        left;
        separation_tolerance=sqrt(eps(Float64)),
    )
    @test Base.count(completed.completed) == 0
    @test norm(completed.gram - data.gram) <= 1e-12
    common = FusedNLFEAST.balanced_loewner_vectors(data, right, left)
    projected = eigen(left' * A * right, left' * B * right)
    @test multiset_distance(common.values, projected.values) <= 1e-11

    theta = 1.7 + 0.2im
    quadratic_T = z -> fill(ComplexF64(z^2 - 2), 1, 1)
    quadratic_data = modal_loewner_pencil(
        quadratic_T,
        (eta, beta) -> fill(ComplexF64(eta + beta), 1, 1),
        ComplexF64[theta],
        ones(ComplexF64, 1, 1),
        ComplexF64[theta],
        ones(ComplexF64, 1, 1),
    )
    newton_value = theta - (theta^2 - 2) / (2theta)
    @test only(quadratic_data.action) / only(quadratic_data.gram) ≈ newton_value
    quadratic_completed = residual_completed_overlap(
        quadratic_T,
        nothing,
        ComplexF64[theta],
        ones(ComplexF64, 1, 1),
        ComplexF64[theta],
        ones(ComplexF64, 1, 1);
        separation_tolerance=sqrt(eps(Float64)),
        divided_entry=(eta, y, beta, x) -> conj(y[1]) * (eta + beta) * x[1],
    )
    @test only(quadratic_completed.completed)
    @test only(quadratic_completed.gram) ≈ only(quadratic_data.gram)

    roots = ComplexF64[-2, -1, 0, 1, 2]
    coefficients = FusedNLFEAST.scalar_polynomial_coefficients(roots)
    scalar_case = polynomial_case([fill(value, 1, 1) for value in coefficients])
    scalar_vectors = ones(ComplexF64, 1, length(roots))
    scalar_data = modal_loewner_pencil(
        scalar_case.T,
        scalar_case.divided_difference,
        roots,
        scalar_vectors,
        roots,
        scalar_vectors,
    )
    scalar_singulars = svdvals(scalar_data.gram)
    @test scalar_singulars[end] / scalar_singulars[1] >= 1e-2
    @test rank(scalar_data.gram) == length(roots) > size(scalar_vectors, 1)
    scalar_common = FusedNLFEAST.balanced_loewner_vectors(
        scalar_data,
        scalar_vectors,
        scalar_vectors,
    )
    @test multiset_distance(scalar_common.values, roots) <= 1e-12

    permutation = [3, 1, 2]
    permuted = modal_loewner_pencil(
        T,
        divided_difference,
        right_values,
        right,
        left_values[permutation],
        left[:, permutation],
    )
    permuted_common = FusedNLFEAST.balanced_loewner_vectors(permuted, right, left[:, permutation])
    @test multiset_distance(permuted_common.values, common.values) <= 1e-11
    @test_throws DimensionMismatch modal_loewner_pencil(
        T,
        (eta, theta) -> zeros(ComplexF64, n - 1, n - 1),
        right_values,
        right,
        left_values,
        left,
    )
    @test_throws ArgumentError modal_loewner_pencil(
        T,
        (eta, theta) -> fill(ComplexF64(NaN), n, n),
        right_values,
        right,
        left_values,
        left,
    )
    @test_throws DimensionMismatch modal_loewner_pencil(
        T,
        nothing,
        right_values,
        right,
        left_values,
        left;
        divided_overlap=(alphas, Y, betas, X) -> zeros(ComplexF64, count - 1, count - 1),
    )
end

@testset "repeated semisimple common gauge" begin
    repeated_roots = ComplexF64[-0.4, -0.4, 0.5]
    repeated_similarity = ComplexF64[1 0.7 -0.2; 0.1 1 0.5; -0.3 0.2 1]
    repeated_inverse = inv(repeated_similarity)
    repeated_T = z -> repeated_similarity * Diagonal(sin.(z .- repeated_roots)) * repeated_inverse
    repeated_divided_difference = (eta, theta) -> repeated_similarity * Diagonal(
        FusedNLFEAST.sine_divided_difference.(eta .- repeated_roots, theta .- repeated_roots),
    ) * repeated_inverse
    repeated_functions = Tuple(
        (eta, theta) -> FusedNLFEAST.sine_divided_difference(eta - root, theta - root)
        for root in repeated_roots
    )
    repeated_overlap = (left_values, left, right_values, right) ->
        FusedNLFEAST.diagonal_divided_overlap(
            repeated_functions,
            adjoint(repeated_similarity) * left,
            repeated_inverse * right,
            left_values,
            right_values,
        )
    right_solve = (z, B) -> repeated_T(z) \ B
    left_solve = (z, B) -> adjoint(repeated_T(z)) \ B
    rng = MersenneTwister(1501)
    result = fused_nlfeast(
        repeated_T,
        right_solve,
        left_solve,
        CircularChart(0, 1.2, 8),
        rand(rng, ComplexF64, 3, 3),
        rand(rng, ComplexF64, 3, 3);
        config=FusedConfig(moment_count=1, iterations=2, ranktol=1e-10, residual_tol=1e-12),
        divided_overlap=repeated_overlap,
    )

    @test multiset_distance(result.extraction.values, repeated_roots) <= 1e-12
    @test maximum(result.extraction.residuals) <= 1e-12
    @test all(record.selection_mode === :common for record in result.history)
    @test abs(result.history[1].loewner_condition) < 10
    cluster_gram = result.extraction.coupling.gram[1:2, 1:2]
    @test abs(cluster_gram[1, 2]) + abs(cluster_gram[2, 1]) >= 1e-2
    completed_overlap = residual_completed_overlap(
        repeated_T,
        repeated_divided_difference,
        result.extraction.values,
        result.extraction.right,
        result.extraction.values,
        result.extraction.left;
        separation_tolerance=1e-6,
    )
    completed_reference = modal_loewner_pencil(
        repeated_T,
        repeated_divided_difference,
        result.extraction.values,
        result.extraction.right,
        result.extraction.values,
        result.extraction.left,
    )
    @test Base.count(completed_overlap.completed) == 5
    @test norm(completed_overlap.gram - completed_reference.gram) <= 1e-12

    compressed_rng = MersenneTwister(1501)
    compressed = fused_nlfeast(
        repeated_T,
        right_solve,
        left_solve,
        CircularChart(0, 1.2, 8),
        rand(compressed_rng, ComplexF64, 3, 3),
        rand(compressed_rng, ComplexF64, 3, 3);
        config=FusedConfig(
            moment_count=2,
            iterations=3,
            ranktol=1e-10,
            residual_tol=1e-12,
            right_moment_width=2,
            left_moment_width=2,
        ),
        divided_overlap=repeated_overlap,
    )
    @test multiset_distance(compressed.extraction.values, repeated_roots) <= 1e-11
    @test maximum(compressed.extraction.residuals) <= 1e-12
    @test all(record.right_moment_width == 2 for record in compressed.history[2:end])
    @test all(record.left_moment_width == 2 for record in compressed.history[2:end])
end
