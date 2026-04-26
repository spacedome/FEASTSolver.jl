@testitem "torture generated: dual FEAST on non-normal Grcar" tags=[:torture, :generated] setup=[FEASTTestSetup] begin
    using FEASTSolver
    using LinearAlgebra
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged, grcar_matrix

    n = 20
    A = grcar_matrix(n)
    B = Matrix{Float64}(I, n, n)
    c, r = 0.0 + 2.0im, 0.8
    exact = eigvals(A)
    expected = exact[in_contour(exact, c, r)]

    λ, _, _, res = dual_gen_feast!(
        initial_subspace(n, length(expected) + 2, 501),
        initial_subspace(n, length(expected) + 2, 502),
        A,
        B;
        nodes=16,
        iter=30,
        c=c,
        r=r,
        ϵ=1e-10,
    )

    assert_eigenvalues_found(λ, expected; atol=1e-6)
    assert_converged(res; atol=1e-6)
end

@testitem "torture generated: sparse Poisson reference solve" tags=[:torture, :generated] setup=[FEASTTestSetup] begin
    using FEASTSolver
    using LinearAlgebra
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged, poisson_matrix

    A = poisson_matrix(8)
    exact = eigvals(Matrix(A))
    c, r = 0.7, 0.35
    expected = complex.(exact[in_contour(exact, c, r)])

    λ, _, res = feast!(
        initial_subspace(size(A, 1), length(expected) + 4, 602),
        A;
        nodes=16,
        iter=20,
        c=c,
        r=r,
        ϵ=1e-10,
        store=false,
    )

    assert_eigenvalues_found(λ, expected; atol=1e-8)
    assert_converged(res; atol=1e-8)
end
