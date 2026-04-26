@testitem "distributed FEAST: standard dense worker paths" tags=[:distributed] setup=[FEASTTestSetup] begin
    using FEASTSolver
    using Distributed
    using LinearAlgebra
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged, ensure_workers, external_workers

    added = ensure_workers(2)
    try
        worker_ids = external_workers()[1:2]
        A = Matrix(Diagonal(1.0:12.0))
        expected = complex.(1.0:4.0)

        for (store, seed) in ((false, 111), (true, 112))
            λ, _, res = distributed_feast!(
                initial_subspace(12, 4, seed),
                A;
                nodes=8,
                iter=10,
                c=2.5,
                r=1.6,
                ϵ=1e-12,
                store=store,
                worker_ids=worker_ids,
                worker_blas_threads=1,
            )

            assert_eigenvalues_found(λ, expected; atol=1e-10)
            assert_converged(res; atol=1e-10)
        end

        stats = DenseDistributedFeastStats()
        plan = DenseDistributedFeastPlan(
            A,
            4;
            nodes=8,
            c=2.5,
            r=1.6,
            store=true,
            worker_ids=worker_ids,
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

@testitem "distributed FEAST: generalized dense variants" tags=[:distributed] setup=[FEASTTestSetup] begin
    using FEASTSolver
    using Distributed
    using LinearAlgebra
    using .FEASTTestSetup: initial_subspace, assert_eigenvalues_found, assert_converged, ensure_workers, external_workers

    added = ensure_workers(2)
    try
        worker_ids = external_workers()[1:2]
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
            worker_ids=worker_ids,
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
            worker_ids=worker_ids,
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
            worker_ids=worker_ids,
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

@testitem "distributed NLFEAST: dense and sparse worker paths" tags=[:distributed] setup=[FEASTTestSetup] begin
    using FEASTSolver
    using Distributed
    using LinearAlgebra
    using SparseArrays
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

        T = feast_gallery(
            "polynomial",
            [spdiagm(0 => complex.(-1.0:-1.0:-n)), sparse(I, n, n)],
        )
        λ, _, res = distributed_nlfeast!(
            T,
            initial_subspace(n, 4, 406),
            8,
            10;
            c=2.5,
            r=1.6,
            ϵ=1e-12,
            store=false,
            materialize_nodes=false,
            worker_ids=worker_ids,
            worker_blas_threads=1,
        )

        inside = in_contour(λ, 2.5, 1.6)
        assert_eigenvalues_found(λ[inside], expected_linear; atol=1e-10)
        assert_converged(res[inside]; atol=1e-10)
    finally
        if !isempty(added)
            rmprocs(added)
        end
    end
end

@testitem "distributed NLFEAST: sparse setup remains sparse" tags=[:distributed] setup=[FEASTTestSetup] begin
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
