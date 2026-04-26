@testitem "generalized FEAST: dense diagonal pencil variants" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using LinearAlgebra
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged

    A = Matrix(Diagonal(1.0:12.0))
    B = Matrix(Diagonal(2.0 .+ (1.0:12.0) ./ 10.0))
    exact = diag(A) ./ diag(B)
    expected = complex.(exact[in_contour(exact, 1.05, 0.7)])
    gen_stats = DenseFeastStats()
    dual_stats = DenseFeastStats()

    λ, _, res = gen_feast!(
        initial_subspace(12, 6, 201),
        A,
        B;
        nodes=12,
        iter=30,
        c=1.05,
        r=0.7,
        ϵ=1e-12,
        stats=gen_stats,
    )

    assert_eigenvalues_found(λ, expected; atol=1e-10)
    assert_converged(res; atol=1e-10)
    @test gen_stats.iterations == length(gen_stats.iteration_log)
    @test gen_stats.iteration_log[end].variant == :generalized

    λ, _, _, res = dual_gen_feast!(
        initial_subspace(12, 6, 202),
        initial_subspace(12, 6, 203),
        A,
        B;
        nodes=12,
        iter=30,
        c=1.05,
        r=0.7,
        ϵ=1e-12,
        stats=dual_stats,
    )

    assert_eigenvalues_found(λ, expected; atol=1e-10)
    assert_converged(res; atol=1e-10)
    @test dual_stats.iterations == length(dual_stats.iteration_log)
    @test dual_stats.iteration_log[end].variant == :dual_generalized
end

@testitem "generalized FEAST: identity mass matrix and sparse pencil" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using LinearAlgebra
    using SparseArrays
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged

    A = Matrix(Diagonal(1.0:8.0))
    expected = complex.(1.0:3.0)

    λ, _, res = gen_feast!(
        initial_subspace(8, 4, 251),
        A,
        I;
        nodes=8,
        iter=10,
        c=2.0,
        r=1.2,
        ϵ=1e-10,
    )

    assert_eigenvalues_found(λ, expected; atol=1e-10)
    assert_converged(res; atol=1e-10)

    λ, _, _, res = dual_gen_feast!(
        initial_subspace(8, 4, 252),
        initial_subspace(8, 4, 253),
        A,
        I;
        nodes=8,
        iter=10,
        c=2.0,
        r=1.2,
        ϵ=1e-10,
    )

    assert_eigenvalues_found(λ, expected; atol=1e-10)
    assert_converged(res; atol=1e-10)

    λ, _, res = gen_feast!(
        initial_subspace(8, 4, 254),
        sparse(A),
        I;
        nodes=8,
        iter=10,
        c=2.0,
        r=1.2,
        ϵ=1e-10,
    )

    assert_eigenvalues_found(λ, expected; atol=1e-10)
    assert_converged(res; atol=1e-10)
end

@testitem "generalized FEAST: sparse direct solver" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using SparseArrays
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged

    A = spdiagm(0 => collect(1.0:10.0))
    B = spdiagm(0 => 2.0 .+ collect(1.0:10.0) ./ 10.0)
    expected = complex.((1.0:3.0) ./ (2.0 .+ (1.0:3.0) ./ 10.0))
    stats = DenseFeastStats()

    λ, _, res = gen_feast!(
        initial_subspace(size(A, 1), 4, 364),
        A,
        B;
        nodes=12,
        iter=10,
        c=0.9,
        r=0.55,
        ϵ=1e-12,
        store=true,
        stats=stats,
    )

    assert_eigenvalues_found(λ, expected; atol=1e-10)
    assert_converged(res; atol=1e-10)
    @test stats.stored_factor_count == 12
    @test stats.stored_factor_bytes > 0
end
