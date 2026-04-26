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

    function expected_inside_from_dense(A, c, r)
        exact = eigvals(Matrix(A))
        complex.(exact[in_contour(exact, c, r)])
    end

    function grcar_matrix(n::Integer, k::Integer=3; T=Float64)
        A = zeros(T, n, n)
        for j in 1:n
            for i in max(1, j - k):j
                A[i, j] = one(T)
            end
        end
        for i in 2:n
            A[i, i-1] = -one(T)
        end
        A
    end

    function poisson_matrix(n::Integer; T=Float64)
        S = spdiagm(
            -1 => fill(-one(T), n - 1),
             0 => fill(2one(T), n),
             1 => fill(-one(T), n - 1),
        )
        I_n = spdiagm(0 => ones(T, n))
        kron(I_n, S) + kron(S, I_n)
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
