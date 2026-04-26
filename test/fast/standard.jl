@testitem "standard FEAST: diagonal sanity problem" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using LinearAlgebra
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged

    A = Matrix(Diagonal(1.0:12.0))
    expected = complex.(1.0:4.0)
    stats = DenseFeastStats()

    λ, _, res = feast!(
        initial_subspace(12, 4, 101),
        A;
        nodes=8,
        iter=10,
        c=2.5,
        r=1.6,
        ϵ=1e-12,
        stats=stats,
    )

    assert_eigenvalues_found(λ, expected; atol=1e-10)
    assert_converged(res; atol=1e-10)
    @test stats.iterations == length(stats.iteration_log)
    @test stats.iterations > 0
    @test stats.iteration_log[end].variant == :standard
    @test stats.iteration_log[end].eigenvalues_inside == length(expected)
end

@testitem "standard FEAST: symmetric dense problem" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using LinearAlgebra
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged, expected_inside_from_dense

    A = SymTridiagonal(fill(2.0, 16), fill(-1.0, 15))
    c, r = 0.25, 0.25
    expected = expected_inside_from_dense(A, c, r)

    λ, _, res = feast!(
        initial_subspace(size(A, 1), length(expected) + 2, 102),
        Matrix(A);
        nodes=12,
        iter=12,
        c=c,
        r=r,
        ϵ=1e-11,
    )

    assert_eigenvalues_found(λ, expected; atol=1e-10)
    assert_converged(res; atol=1e-10)
end

@testitem "standard FEAST: normal complex dense problem" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using LinearAlgebra
    using Random
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged

    rng = MersenneTwister(103)
    Q = Matrix(qr(randn(rng, ComplexF64, 8, 8)).Q)
    eigenvalues = ComplexF64[
        1.0 + 0.10im,
        1.4 - 0.05im,
        2.0 + 0.20im,
        2.8 - 0.10im,
        5.0 + 1.00im,
        6.0 - 1.00im,
        7.0 + 0.30im,
        8.0,
    ]
    A = Q * Diagonal(eigenvalues) * Q'
    c, r = 1.8 + 0.0im, 1.2
    expected = eigenvalues[in_contour(eigenvalues, c, r)]

    λ, _, res = feast!(
        initial_subspace(size(A, 1), length(expected) + 1, 104),
        A;
        nodes=12,
        iter=12,
        c=c,
        r=r,
        ϵ=1e-10,
    )

    assert_eigenvalues_found(λ, expected; atol=1e-9)
    assert_converged(res; atol=1e-9)
end

@testitem "standard FEAST: non-normal dense problem" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using LinearAlgebra
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged

    eigenvalues = complex.(1.0:10.0)
    A = Matrix(Diagonal(eigenvalues))
    for j in 1:9
        A[j, j+1] = 0.35 + 0.2im
    end
    c, r = 2.5 + 0.0im, 1.6
    expected = eigenvalues[in_contour(eigenvalues, c, r)]

    λ, _, res = feast!(
        initial_subspace(size(A, 1), length(expected) + 1, 105),
        A;
        nodes=12,
        iter=15,
        c=c,
        r=r,
        ϵ=1e-10,
    )

    assert_eigenvalues_found(λ, expected; atol=1e-8)
    assert_converged(res; atol=1e-8)
end

@testitem "standard FEAST: contour variants on sparse Laplacian" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using SparseArrays
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged

    n = 32
    A = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(2.0, n), 1 => fill(-1.0, n - 1))
    expected = complex.([2 - 2cos(k * pi / (n + 1)) for k in 1:4])

    contours = [
        circular_contour_trapezoidal(0.075, 0.075, 12),
        rectangular_contour_trapezoidal(0.0 - 0.05im, 0.16 + 0.05im, 12),
        rectangular_contour_gauss(0.0 - 0.05im, 0.16 + 0.05im, 12),
    ]

    for contour in contours
        λ, _, res = feast!(
            initial_subspace(n, 8, 301),
            A,
            contour;
            iter=10,
            ϵ=1e-11,
        )

        assert_eigenvalues_found(λ, expected; atol=1e-10)
        assert_converged(res; atol=1e-10)
    end
end

@testitem "standard FEAST: sparse direct solver handles missing diagonal storage" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using SparseArrays
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged

    A = spdiagm(0 => [1.0, 2.0, 0.0, 4.0, 5.0, 6.0])
    expected = complex.([1.0, 2.0])
    stats = DenseFeastStats()

    λ, _, res = feast!(
        initial_subspace(size(A, 1), 2, 361),
        A;
        nodes=8,
        iter=10,
        c=1.5,
        r=0.75,
        ϵ=1e-12,
        store=true,
        solver=SparseDirectSolver(),
        stats=stats,
    )

    assert_eigenvalues_found(λ, expected; atol=1e-10)
    assert_converged(res; atol=1e-10)
    @test stats.stored_factor_count == 8
    @test stats.stored_factor_bytes > 0
end

@testitem "standard FEAST: sparse BiCGSTAB solver policy" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using SparseArrays
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged

    A = spdiagm(0 => collect(1.0:8.0))
    expected = complex.([1.0, 2.0])

    λ, _, res = feast!(
        initial_subspace(size(A, 1), 2, 362),
        A;
        nodes=8,
        iter=10,
        c=1.5,
        r=0.75,
        ϵ=1e-9,
        solver=SparseBiCGSTABSolver(reltol=1e-12, max_mv_products=200),
    )

    assert_eigenvalues_found(λ, expected; atol=1e-8)
    assert_converged(res; atol=1e-8)
    @test_throws ErrorException feast!(
        initial_subspace(size(A, 1), 2, 363),
        A;
        nodes=8,
        iter=1,
        c=1.5,
        r=0.75,
        store=true,
        solver=SparseBiCGSTABSolver(),
    )
end

@testitem "standard FEAST: small generated Poisson problem" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using LinearAlgebra
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged, poisson_matrix

    A = poisson_matrix(5)
    exact = eigvals(Matrix(A))
    c, r = 1.3, 0.25
    expected = complex.(exact[in_contour(exact, c, r)])
    stats = DenseFeastStats()

    λ, _, res = feast!(
        initial_subspace(size(A, 1), length(expected) + 2, 365),
        A;
        nodes=12,
        iter=15,
        c=c,
        r=r,
        ϵ=1e-11,
        store=true,
        stats=stats,
    )

    assert_eigenvalues_found(λ, expected; atol=1e-9)
    assert_converged(res; atol=1e-9)
    @test stats.stored_factor_count == 12
    @test stats.stored_factor_bytes > 0
end

@testitem "standard FEAST: custom contour predicate" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using LinearAlgebra
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged

    A = Matrix(Diagonal(1.0:12.0))
    c, r = 2.5 + 0.0im, 1.6
    base = circular_contour_trapezoidal(c, r, 8)
    contour = CustomContour(
        contour_nodes(base),
        contour_weights(base);
        inside=z -> abs(z - c) <= r,
    )
    expected = complex.(1.0:4.0)

    @test in_contour(2.0 + 0.0im, contour)
    @test !in_contour(7.0 + 0.0im, contour)

    λ, _, res = feast!(
        initial_subspace(12, 4, 351),
        A;
        contour=contour,
        iter=10,
        ϵ=1e-12,
    )

    assert_eigenvalues_found(λ, expected; atol=1e-10)
    assert_converged(res; atol=1e-10)
end

@testitem "standard FEAST: custom contours require classification predicate" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using LinearAlgebra
    using .FEASTTestSetup: initial_subspace

    A = Matrix(Diagonal(1.0:8.0))
    base = circular_contour_trapezoidal(2.5, 1.6, 8)
    contour = CustomContour(contour_nodes(base), contour_weights(base))

    @test_throws ErrorException feast!(
        initial_subspace(8, 4, 352),
        A;
        contour=contour,
        iter=1,
    )
end
