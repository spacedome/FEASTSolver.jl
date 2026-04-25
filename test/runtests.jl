using FEASTSolver
using Test
using LinearAlgebra
using Random
using SparseArrays

Random.seed!(1)

function assert_eigenvalues_found(actual, expected; atol)
    @test length(actual) == length(expected)
    if !isempty(actual) && !isempty(expected)
        @test all(λ -> minimum(abs.(actual .- λ)) <= atol, expected)
    end
end

function assert_converged(residuals; atol)
    @test !isempty(residuals)
    @test maximum(residuals) <= atol
end

@testset "standard dense FEAST finds requested eigenvalues" begin
    A = Matrix(Diagonal(1.0:12.0))
    expected = complex.(1.0:4.0)

    λ, _, res = feast!(
        rand(ComplexF64, 12, 4),
        A;
        nodes=8,
        iter=10,
        c=2.5,
        r=1.6,
        ϵ=1e-12,
    )

    assert_eigenvalues_found(λ, expected; atol=1e-10)
    assert_converged(res; atol=1e-10)
end

@testset "generalized FEAST variants find requested eigenvalues" begin
    A = Matrix(Diagonal(1.0:12.0))
    B = Matrix(Diagonal(2.0 .+ (1.0:12.0) ./ 10.0))
    exact = diag(A) ./ diag(B)
    expected = complex.(exact[in_contour(exact, 1.05, 0.7)])

    λ, _, res = gen_feast!(
        rand(ComplexF64, 12, 6),
        A,
        B;
        nodes=12,
        iter=30,
        c=1.05,
        r=0.7,
        ϵ=1e-12,
    )

    assert_eigenvalues_found(λ, expected; atol=1e-10)
    assert_converged(res; atol=1e-10)

    λ, _, _, res = dual_gen_feast!(
        rand(ComplexF64, 12, 6),
        rand(ComplexF64, 12, 6),
        A,
        B;
        nodes=12,
        iter=30,
        c=1.05,
        r=0.7,
        ϵ=1e-12,
    )

    assert_eigenvalues_found(λ, expected; atol=1e-10)
    assert_converged(res; atol=1e-10)
end

@testset "contour variants find sparse Laplacian eigenvalues" begin
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
            rand(ComplexF64, n, 8),
            A,
            contour;
            iter=10,
            ϵ=1e-11,
        )

        assert_eigenvalues_found(λ, expected; atol=1e-10)
        assert_converged(res; atol=1e-10)
    end
end

@testset "nonlinear FEAST handles a linear pencil" begin
    n = 12
    A = Matrix(Diagonal(1.0:n))
    T(z) = z * Matrix{Float64}(I, n, n) - A
    expected = complex.(1.0:4.0)

    λ, _, res = nlfeast!(
        T,
        rand(ComplexF64, n, 4),
        8,
        10;
        c=2.5,
        r=1.6,
        ϵ=1e-12,
        store=true,
    )

    inside = in_contour(λ, 2.5, 1.6)
    assert_eigenvalues_found(λ[inside], expected; atol=1e-10)
    assert_converged(res[inside]; atol=1e-10)
end

@testset "dual generalized FEAST handles a small non-normal problem" begin
    Random.seed!(6)

    n = 20
    A = diagm(
        -1 => fill(-1.0, n - 1),
        0 => fill(1.0, n),
        1 => fill(1.0, n - 1),
        2 => fill(1.0, n - 2),
        3 => fill(1.0, n - 3),
    )
    B = Matrix{Float64}(I, n, n)
    c, r = 0.0 + 2.0im, 0.8
    exact = eigvals(A)
    expected = exact[in_contour(exact, c, r)]

    λ, _, _, res = dual_gen_feast!(
        rand(ComplexF64, n, length(expected) + 2),
        rand(ComplexF64, n, length(expected) + 2),
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
