using FEASTSolver
using Test
using Distributed
using LinearAlgebra
using MatrixDepot
using Random
using SparseArrays

sort_spectrum(λ) = sort(collect(λ), by=z -> (real(z), imag(z)))

function initial_subspace(n, m, seed)
    rand(MersenneTwister(seed), ComplexF64, n, m)
end

function assert_eigenvalues_found(actual, expected; atol)
    @test length(actual) == length(expected)
    if !isempty(actual) && !isempty(expected)
        @test maximum(abs.(sort_spectrum(actual) .- sort_spectrum(expected))) <= atol
    end
end

function assert_converged(residuals; atol)
    @test !isempty(residuals)
    @test maximum(residuals) <= atol
end

external_workers() = filter(!=(myid()), workers())

function ensure_workers(count)
    ids = external_workers()
    if length(ids) >= count
        return Int[]
    end
    addprocs(count - length(ids); exeflags="--project=$(Base.active_project())")
end

@testset "standard dense FEAST finds requested eigenvalues" begin
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

@testset "distributed dense FEAST finds requested eigenvalues" begin
    added = ensure_workers(2)
    try
        A = Matrix(Diagonal(1.0:12.0))
        expected = complex.(1.0:4.0)

        λ, _, res = distributed_feast!(
            initial_subspace(12, 4, 111),
            A;
            nodes=8,
            iter=10,
            c=2.5,
            r=1.6,
            ϵ=1e-12,
            worker_ids=external_workers()[1:2],
            worker_blas_threads=1,
        )

        assert_eigenvalues_found(λ, expected; atol=1e-10)
        assert_converged(res; atol=1e-10)

        λ, _, res = distributed_feast!(
            initial_subspace(12, 4, 112),
            A;
            nodes=8,
            iter=10,
            c=2.5,
            r=1.6,
            ϵ=1e-12,
            store=true,
            worker_ids=external_workers()[1:2],
            worker_blas_threads=1,
        )

        assert_eigenvalues_found(λ, expected; atol=1e-10)
        assert_converged(res; atol=1e-10)

        stats = DenseDistributedFeastStats()
        plan = DenseDistributedFeastPlan(
            A,
            4;
            nodes=8,
            c=2.5,
            r=1.6,
            store=true,
            worker_ids=external_workers()[1:2],
            worker_blas_threads=1,
            stats=stats,
        )
        try
            for seed in (113, 114)
                λ, _, res = distributed_feast!(
                    initial_subspace(12, 4, seed),
                    plan;
                    iter=10,
                    ϵ=1e-12,
                )
                assert_eigenvalues_found(λ, expected; atol=1e-10)
                assert_converged(res; atol=1e-10)
            end
            @test !isempty(stats.iteration_log)
            @test stats.iterations == length(stats.iteration_log)
        finally
            close(plan)
        end
    finally
        if !isempty(added)
            rmprocs(added)
        end
    end
end

@testset "generalized FEAST variants find requested eigenvalues" begin
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

@testset "distributed generalized FEAST variants find requested eigenvalues" begin
    added = ensure_workers(2)
    try
        A = Matrix(Diagonal(1.0:12.0))
        B = Matrix(Diagonal(2.0 .+ (1.0:12.0) ./ 10.0))
        exact = diag(A) ./ diag(B)
        expected = complex.(exact[in_contour(exact, 1.05, 0.7)])

        λ, _, res = distributed_gen_feast!(
            initial_subspace(12, 6, 211),
            A,
            B;
            nodes=12,
            iter=30,
            c=1.05,
            r=0.7,
            ϵ=1e-12,
            worker_ids=external_workers()[1:2],
            worker_blas_threads=1,
        )

        assert_eigenvalues_found(λ, expected; atol=1e-10)
        assert_converged(res; atol=1e-10)

        stats = DenseDistributedFeastStats()
        λ, _, _, res = distributed_dual_gen_feast!(
            initial_subspace(12, 6, 212),
            initial_subspace(12, 6, 213),
            A,
            B;
            nodes=12,
            iter=30,
            c=1.05,
            r=0.7,
            ϵ=1e-12,
            store=true,
            worker_ids=external_workers()[1:2],
            worker_blas_threads=1,
            stats=stats,
        )

        assert_eigenvalues_found(λ, expected; atol=1e-10)
        assert_converged(res; atol=1e-10)
        @test stats.iterations == length(stats.iteration_log)
        @test !isempty(stats.iteration_log)

        λ, _, res = distributed_gen_feast!(
            initial_subspace(8, 4, 214),
            Matrix(Diagonal(1.0:8.0)),
            I;
            nodes=8,
            iter=10,
            c=2.0,
            r=1.2,
            ϵ=1e-10,
            worker_ids=external_workers()[1:2],
            worker_blas_threads=1,
        )

        assert_eigenvalues_found(λ, complex.(1.0:3.0); atol=1e-10)
        assert_converged(res; atol=1e-10)
    finally
        if !isempty(added)
            rmprocs(added)
        end
    end
end

@testset "generalized FEAST accepts identity operator" begin
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

@testset "nonlinear FEAST handles a linear pencil" begin
    n = 12
    A = Matrix(Diagonal(1.0:n))
    T(z) = z * Matrix{Float64}(I, n, n) - A
    expected = complex.(1.0:4.0)
    stats = DenseFeastStats()

    λ, _, res = nlfeast!(
        T,
        initial_subspace(n, 4, 401),
        8,
        10;
        c=2.5,
        r=1.6,
        ϵ=1e-12,
        store=true,
        stats=stats,
    )

    inside = in_contour(λ, 2.5, 1.6)
    assert_eigenvalues_found(λ[inside], expected; atol=1e-10)
    assert_converged(res[inside]; atol=1e-10)
    @test stats.iterations == length(stats.iteration_log)
    @test stats.iteration_log[end].variant == :nonlinear
end

@testset "dual generalized FEAST handles a small non-normal problem" begin
    n = 20
    A = matrixdepot("grcar", n)
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

@testset "MatrixDepot Poisson problem" begin
    A = Matrix(matrixdepot("poisson", 5))
    exact = eigvals(A)
    c, r = 1.3, 0.25
    expected = complex.(exact[in_contour(exact, c, r)])

    λ, _, res = feast!(
        initial_subspace(size(A, 1), length(expected) + 2, 601),
        A;
        nodes=12,
        iter=15,
        c=c,
        r=r,
        ϵ=1e-11,
    )

    assert_eigenvalues_found(λ, expected; atol=1e-9)
    assert_converged(res; atol=1e-9)
end
