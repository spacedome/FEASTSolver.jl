using TestItemRunner

const TEST_FILTER = isempty(ARGS) ? nothing : Regex(ARGS[1])
const RUN_SLOW_TESTS = get(ENV, "FEAST_TEST_SLOW", "0") == "1"
const TEST_ROOT = normpath(@__DIR__)

function testitem_filter(ti)
    startswith(normpath(ti.filename), TEST_ROOT) || return false
    name_ok = TEST_FILTER === nothing || occursin(TEST_FILTER, ti.name)
    slow_ok = RUN_SLOW_TESTS || TEST_FILTER !== nothing || !(:slow in ti.tags)
    name_ok && slow_ok
end

@testmodule FEASTTestSetup begin
    using FEASTSolver
    using Test
    using Distributed
    using LinearAlgebra
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

    function butterfly_polynomial_matrices()
        N = diagm(-1 => ones(7))
        Mh0 = (4I + N + N') / 6
        Mh1 = N - N'
        Mh2 = -(2I - N - N')
        Mh3 = Mh1
        Mh4 = -Mh2
        c = [0.6 1.3; 1.3 0.1; 0.1 1.2; 1.0 1.0; 1.2 1.0]
        I8 = Matrix(I, 8, 8)
        [
            c[1, 1] * kron(I8, Mh0) + c[1, 2] * kron(Mh0, I8),
            c[2, 1] * kron(I8, Mh1) + c[2, 2] * kron(Mh1, I8),
            c[3, 1] * kron(I8, Mh2) + c[3, 2] * kron(Mh2, I8),
            c[4, 1] * kron(I8, Mh3) + c[4, 2] * kron(Mh3, I8),
            c[5, 1] * kron(I8, Mh4) + c[5, 2] * kron(Mh4, I8),
        ]
    end

    external_workers() = filter(!=(myid()), workers())

    function ensure_workers(count)
        ids = external_workers()
        if length(ids) >= count
            return Int[]
        end
        addprocs(count - length(ids); exeflags="--project=$(Base.active_project())")
    end
end

@testitem "standard dense FEAST finds requested eigenvalues" setup=[FEASTTestSetup] begin
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

@testitem "distributed dense FEAST finds requested eigenvalues" tags=[:distributed] setup=[FEASTTestSetup] begin
    using FEASTSolver
    using Distributed
    using LinearAlgebra
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged, ensure_workers, external_workers

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

@testitem "generalized FEAST variants find requested eigenvalues" setup=[FEASTTestSetup] begin
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

@testitem "distributed generalized FEAST variants find requested eigenvalues" tags=[:distributed] setup=[FEASTTestSetup] begin
    using FEASTSolver
    using Distributed
    using LinearAlgebra
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged, ensure_workers, external_workers

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

@testitem "generalized FEAST accepts identity operator" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using LinearAlgebra
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
end

@testitem "contour variants find sparse Laplacian eigenvalues" setup=[FEASTTestSetup] begin
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

@testitem "sparse FEAST direct solver handles missing diagonal storage" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using SparseArrays
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged

    A = spdiagm(0 => [1.0, 2.0, 0.0, 4.0, 5.0, 6.0])
    expected = complex.([1.0, 2.0])

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
    )

    assert_eigenvalues_found(λ, expected; atol=1e-10)
    assert_converged(res; atol=1e-10)
end

@testitem "sparse FEAST exposes a BiCGSTAB solver policy" setup=[FEASTTestSetup] begin
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

@testitem "custom contours classify eigenvalues with an explicit predicate" setup=[FEASTTestSetup] begin
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

@testitem "custom contours require a predicate for solver classification" setup=[FEASTTestSetup] begin
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

@testitem "nonlinear FEAST handles a linear pencil" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using LinearAlgebra
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged

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

@testitem "nonlinear FEAST accepts a custom contour" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using LinearAlgebra
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged

    n = 12
    A = Matrix(Diagonal(1.0:n))
    T(z) = z * Matrix{Float64}(I, n, n) - A
    c, r = 2.5 + 0.0im, 1.6
    base = circular_contour_trapezoidal(c, r, 8)
    contour = CustomContour(
        contour_nodes(base),
        contour_weights(base);
        inside=z -> abs(z - c) <= r,
    )
    expected = complex.(1.0:4.0)

    λ, _, res = nlfeast!(
        T,
        initial_subspace(n, 4, 402),
        contour,
        10;
        ϵ=1e-12,
        store=true,
    )

    inside = in_contour(λ, contour)
    assert_eigenvalues_found(λ[inside], expected; atol=1e-10)
    assert_converged(res[inside]; atol=1e-10)
end

@testitem "FEAST gallery operators materialize and apply consistently" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using LinearAlgebra
    using Random
    using SparseArrays

    rng = MersenneTwister(7781)
    operators = Any[
        matrix_operator(z -> z * Matrix{ComplexF64}(I, 8, 8) - Diagonal(complex.(1.0:8.0)), zeros(ComplexF64, 8, 8)),
        feast_gallery(
            "polynomial",
            [spdiagm(0 => complex.(-1.0:-1.0:-8)), sparse(I, 8, 8)],
        ),
        feast_gallery("nlevp_native_hadeler", 100, 8),
        feast_gallery("nlevp_native_loaded_string", 8, 1, 1),
    ]

    for op in operators
        z = 1.2 + 0.3im
        M = operator_prototype(op)
        materialize!(M, op, z)
        M2 = similar(M)
        matrix_materializer(op)(M2, z)
        V = rand(rng, ComplexF64, size(op, 2), 3)
        Y = zeros(ComplexF64, size(op, 1), 3)

        mul!(Y, op, z, V)

        @test M2 ≈ M
        @test Y ≈ M * V
    end
end

@testitem "distributed nonlinear FEAST handles small dense problems" tags=[:distributed] setup=[FEASTTestSetup] begin
    using FEASTSolver
    using Distributed
    using LinearAlgebra
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged, ensure_workers, external_workers

    added = ensure_workers(2)
    try
        worker_ids = external_workers()[1:2]

        n = 12
        A = Matrix(Diagonal(1.0:n))
        Tlinear(z) = z * Matrix{Float64}(I, n, n) - A
        expected_linear = complex.(1.0:4.0)
        stats = DenseDistributedFeastStats()

        λ, _, res = distributed_nlfeast!(
            Tlinear,
            initial_subspace(n, 4, 405),
            8,
            10;
            c=2.5,
            r=1.6,
            ϵ=1e-12,
            store=true,
            worker_ids=worker_ids,
            worker_blas_threads=1,
            stats=stats,
        )

        inside = in_contour(λ, 2.5, 1.6)
        assert_eigenvalues_found(λ[inside], expected_linear; atol=1e-10)
        assert_converged(res[inside]; atol=1e-10)
        @test stats.iterations == length(stats.iteration_log)
        @test !isempty(stats.iteration_log)

        roots1 = ComplexF64[0.6 + 0.05im, 0.75, 0.9 - 0.03im, 2.0, 2.3 + 0.1im, 2.6 - 0.1im]
        roots2 = ComplexF64[-1.0, -1.2 + 0.2im, -1.4 - 0.1im, 3.0, 3.2, 3.4]
        D1 = Diagonal(roots1 .+ roots2)
        D0 = Diagonal(roots1 .* roots2)
        V = Matrix{ComplexF64}(I, length(roots1), length(roots1))
        V[1, 2] = 0.4
        V[2, 3] = -0.2im
        Vinv = inv(V)
        Tquadratic(z) = V * (z^2 * I - z * D1 + D0) * Vinv
        c, r = 0.75 + 0.0im, 0.25
        expected_quadratic = roots1[in_contour(roots1, c, r)]

        λ, _, res = distributed_nlfeast!(
            Tquadratic,
            initial_subspace(length(roots1), length(expected_quadratic) + 1, 406),
            16,
            20;
            c=c,
            r=r,
            ϵ=1e-10,
            store=false,
            worker_ids=worker_ids,
            worker_blas_threads=1,
        )

        inside = in_contour(λ, c, r)
        assert_eigenvalues_found(λ[inside], expected_quadratic; atol=1e-9)
        assert_converged(res[inside]; atol=1e-9)
    finally
        if !isempty(added)
            rmprocs(added)
        end
    end
end

@testitem "distributed nonlinear FEAST does not densify sparse operator setup" tags=[:distributed] setup=[FEASTTestSetup] begin
    using FEASTSolver
    using Distributed
    using LinearAlgebra
    using SparseArrays
    using .FEASTTestSetup: ensure_workers, external_workers

    added = ensure_workers(1)
    try
        worker_ids = external_workers()[1:1]

        n = 10
        T = feast_gallery(
            "polynomial",
            [spdiagm(0 => complex.(-1.0:-1.0:-n)), sparse(I, n, n)],
        )

        plan = DenseDistributedNonlinearFeastPlan(
            T,
            n,
            4;
            nodes=8,
            c=2.0,
            r=1.2,
            store=false,
            materialize_nodes=false,
            worker_ids=worker_ids,
            worker_blas_threads=1,
        )
        try
            uses_generic_sparse_path = remotecall_fetch(
                Main.eval,
                worker_ids[1],
                quote
                    import SparseArrays
                    let key = $(QuoteNode(plan.key))
                        ws = FEASTSolver._DISTRIBUTED_DENSE_FEAST_WORKSPACES[key]
                        ws.T === nothing &&
                            ws.matrix_update !== nothing &&
                            ws.Tz isa SparseArrays.SparseMatrixCSC &&
                            ws.lu_ws === nothing
                    end
                end,
            )
            @test uses_generic_sparse_path
        finally
            close(plan)
        end

    finally
        if !isempty(added)
            rmprocs(added)
        end
    end
end

@testitem "nonlinear FEAST handles a quadratic matrix polynomial" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using LinearAlgebra
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged

    roots1 = ComplexF64[
        0.6 + 0.05im,
        0.75,
        0.9 - 0.03im,
        2.0,
        2.3 + 0.1im,
        2.6 - 0.1im,
    ]
    roots2 = ComplexF64[-1.0, -1.2 + 0.2im, -1.4 - 0.1im, 3.0, 3.2, 3.4]
    D1 = Diagonal(roots1 .+ roots2)
    D0 = Diagonal(roots1 .* roots2)
    V = Matrix{ComplexF64}(I, length(roots1), length(roots1))
    V[1, 2] = 0.4
    V[2, 3] = -0.2im
    Vinv = inv(V)
    T(z) = V * (z^2 * I - z * D1 + D0) * Vinv
    c, r = 0.75 + 0.0im, 0.25
    expected = roots1[in_contour(roots1, c, r)]

    for (seed, store) in ((777, true), (778, false))
        λ, _, res = nlfeast!(
            T,
            initial_subspace(length(roots1), length(expected) + 1, seed),
            16,
            20;
            c=c,
            r=r,
            ϵ=1e-10,
            store=store,
        )

        inside = in_contour(λ, c, r)
        assert_eigenvalues_found(λ[inside], expected; atol=1e-9)
        assert_converged(res[inside]; atol=1e-9)
    end
end

@testitem "nonlinear FEAST handles butterfly polynomial" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged, butterfly_polynomial_matrices

    A = butterfly_polynomial_matrices()
    T(z) = z^4 * A[5] + z^3 * A[4] + z^2 * A[3] + z * A[2] + A[1]
    c, r = 1.0 + 1.0im, 0.5
    reference, _, reference_res = companion(A)
    expected = reference[in_contour(reference, c, r) .& (reference_res .< 1e-8)]

    λ, _, res = nlfeast!(
        T,
        initial_subspace(size(A[1], 1), length(expected) + 4, 901),
        64,
        3;
        c=c,
        r=r,
        ϵ=1e-8,
        store=true,
        spurious=5e-3,
    )

    inside = in_contour(λ, c, r)
    assert_eigenvalues_found(λ[inside], expected; atol=1e-8)
    assert_converged(res[inside]; atol=1e-8)
end

@testitem "nonlinear FEAST handles gun cavity problem" tags=[:slow, :nep] setup=[FEASTTestSetup] begin
    using FEASTSolver
    using NonlinearEigenproblems: nep_gallery, compute_Mder
    using .FEASTTestSetup: initial_subspace, assert_converged

    nep = nep_gallery("nlevp_native_gun")
    T(z) = compute_Mder(nep, z)
    c, r = 140000.0 + 0.0im, 30000.0

    λ, _, res = nlfeast!(
        T,
        initial_subspace(size(nep, 1), 32, 9901),
        8,
        3;
        c=c,
        r=r,
        ϵ=1e-8,
        store=false,
        spurious=1e-5,
    )

    inside = in_contour(λ, c, r)
    residuals_inside = res[inside]
    @test count(inside) >= 16
    @test count(residuals_inside .< 1e-6) >= 16
    assert_converged(residuals_inside; atol=1e-4)
end

@testitem "dual generalized FEAST handles a small non-normal problem" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using LinearAlgebra
    using MatrixDepot
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged

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

@testitem "MatrixDepot Poisson problem" setup=[FEASTTestSetup] begin
    using FEASTSolver
    using LinearAlgebra
    using MatrixDepot
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged

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

@run_package_tests filter=testitem_filter verbose=true
